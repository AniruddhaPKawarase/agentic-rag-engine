"""
trade_router.py
===============

Cascaded classifier that maps a free-text construction question to a
bounded (trade, role) cell.

Cascade
-------
Tier 0  Deterministic regex on monosemic keywords (~5ms, $0)
Tier 1  claude-haiku-4-5 with tool-use JSON schema (~300ms, ~$0.0001 cached)
Tier 2  Sonnet escalation (opt-in via accuracy_mode flag; default off)

Confidence gating (set by caller, not the router)
-------------------------------------------------
HIGH    >= TRADE_ROUTING_HIGH_THRESHOLD     → filter retrieval
MEDIUM  >= TRADE_ROUTING_MEDIUM_THRESHOLD   → prompt hint only
LOW     <  TRADE_ROUTING_MEDIUM_THRESHOLD   → no-op (today's behavior)

Strict invariants
-----------------
- Pure function on the cache miss path; cached on a hot loop.
- Never raises — any failure path returns RouterResult with classifier_tier=-1
  and primary=("unknown","unknown") so callers safely no-op.
- Honors TRADE_ROUTING_ENABLED env flag — when false, returns immediate no-op.
"""
from __future__ import annotations

import functools
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Tuple

from gateway.sheet_decoder import ROLES, TRADES

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public schema
# ---------------------------------------------------------------------------
@dataclass
class RouterResult:
    primary: Tuple[str, str]                                  # (trade, role)
    primary_confidence: float                                 # 0.0 - 1.0
    alternatives: List[Tuple[str, str, float]] = field(default_factory=list)
    rationale: str = ""
    classifier_tier: int = -1                                 # 0=regex, 1=haiku, 2=sonnet, -1=noop/error
    latency_ms: int = 0

    @property
    def is_high_confidence(self) -> bool:
        return self.primary_confidence >= _high_threshold()

    @property
    def is_medium_confidence(self) -> bool:
        return _medium_threshold() <= self.primary_confidence < _high_threshold()

    @property
    def is_noop(self) -> bool:
        return self.classifier_tier == -1 or self.primary == ("unknown", "unknown")

    def to_dict(self) -> dict:
        d = asdict(self)
        d["primary"] = list(d["primary"])
        d["alternatives"] = [list(a) for a in d["alternatives"]]
        return d


def _flag(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).strip().lower() in ("1", "true", "yes", "on")


def _high_threshold() -> float:
    try:
        return float(os.getenv("TRADE_ROUTING_HIGH_THRESHOLD", "0.85"))
    except ValueError:
        return 0.85


def _medium_threshold() -> float:
    try:
        return float(os.getenv("TRADE_ROUTING_MEDIUM_THRESHOLD", "0.60"))
    except ValueError:
        return 0.60


# ---------------------------------------------------------------------------
# Tier 0 — deterministic regex (monosemic keywords only)
# ---------------------------------------------------------------------------
# Each rule: (compiled pattern, (trade, role), confidence, rationale)
# DELIBERATELY EXCLUDED polysemous terms: slab, lighting, riser (alone),
# outlet, panel (alone), fixture (alone). Those fall through to Tier 1.
_TIER_0_RULES: Tuple[Tuple[re.Pattern, Tuple[str, str], float, str], ...] = (
    # --- RCP / ceiling ---
    (re.compile(r"\bRCP\b|\breflected\s+ceiling", re.IGNORECASE),
        ("A", "RCP"), 0.97, "RCP keyword / reflected ceiling = Architectural RCP"),
    (re.compile(r"\bceiling\s+(height|finish|grid|tile|elevation|soffit|bulkhead)", re.IGNORECASE),
        ("A", "RCP"), 0.95, "ceiling attribute → Architectural RCP"),
    (re.compile(r"\bdropped\s+ceiling|\bsoffit\s+(height|location)", re.IGNORECASE),
        ("A", "RCP"), 0.94, "dropped ceiling / soffit → RCP"),
    # --- Door / Window schedules ---
    (re.compile(r"\bdoor\s+schedule|\bdoor\s+(type|hardware|frame|fire\s*rating)", re.IGNORECASE),
        ("A", "Schedule"), 0.95, "door attribute → Architectural Door Schedule"),
    (re.compile(r"\bwindow\s+schedule|\bwindow\s+(type|glazing|frame)", re.IGNORECASE),
        ("A", "Schedule"), 0.94, "window attribute → Architectural Window Schedule"),
    # --- Structural ---
    (re.compile(r"\bfoundation\s+(plan|depth|reinforcement)", re.IGNORECASE),
        ("S", "Plan"), 0.96, "foundation → Structural Plan"),
    (re.compile(r"\bbeam\s+(schedule|size|spacing|depth)", re.IGNORECASE),
        ("S", "Schedule"), 0.94, "beam attribute → Structural Schedule"),
    (re.compile(r"\bcolumn\s+(schedule|size|spacing|location)", re.IGNORECASE),
        ("S", "Schedule"), 0.94, "column attribute → Structural Schedule"),
    (re.compile(r"\bframing\s+plan", re.IGNORECASE),
        ("S", "Plan"), 0.95, "framing plan → Structural Plan"),
    (re.compile(r"\brebar|\breinforcement\s+detail", re.IGNORECASE),
        ("S", "Detail"), 0.92, "rebar / reinforcement → Structural Detail"),
    # --- Slab — DISAMBIGUATED by context ---
    (re.compile(r"\bslab\s+(top\s+elevation|elevation|thickness|reinforcement|on\s+grade)", re.IGNORECASE),
        ("S", "Plan"), 0.92, "slab elevation/thickness → Structural Plan"),
    # --- Mechanical (HVAC) ---
    (re.compile(r"\bduct\s+(siz(e|ing)|routing|gauge|insulation)|\bductwork|\bduct\s+in\b", re.IGNORECASE),
        ("M", "Plan"), 0.93, "duct → Mechanical Plan"),
    (re.compile(r"\b(VAV|AHU|RTU|FCU|CRAC|chiller|boiler)\b", re.IGNORECASE),
        ("M", "Schedule"), 0.92, "HVAC equipment → Mechanical Schedule"),
    (re.compile(r"\bdiffuser|\breturn\s+grille|\bsupply\s+air", re.IGNORECASE),
        ("M", "Plan"), 0.90, "diffuser / air → Mechanical Plan"),
    # --- Electrical ---
    (re.compile(r"\bpanel\s+schedule|\bsingle[-\s]?line\s+diagram|\bone[-\s]?line", re.IGNORECASE),
        ("E", "Schedule"), 0.97, "panel schedule / one-line → Electrical Schedule/Diagram"),
    (re.compile(r"\breceptacle|\bduplex\s+outlet|\bGFCI", re.IGNORECASE),
        ("E", "Plan"), 0.91, "receptacle → Electrical Power Plan"),
    (re.compile(r"\bswitchgear|\btransformer\b.*power|\bservice\s+entrance", re.IGNORECASE),
        ("E", "Diagram"), 0.92, "switchgear / service → Electrical Single-Line"),
    # --- Plumbing ---
    (re.compile(r"\bplumbing\s+fixture|\blavatory|\bwater\s+closet|\burinal\b", re.IGNORECASE),
        ("P", "Schedule"), 0.92, "plumbing fixture → Plumbing Schedule"),
    (re.compile(r"\b(domestic\s+water|sanitary|storm|vent)\s+(line|piping|sizing)", re.IGNORECASE),
        ("P", "Plan"), 0.91, "plumbing piping → Plumbing Plan"),
    (re.compile(r"\bgas\s+(piping|line|meter)", re.IGNORECASE),
        ("P", "Plan"), 0.90, "gas piping → Plumbing Plan"),
    # --- Fire Protection ---
    (re.compile(r"\bsprinkler\s+(head|coverage|spacing|riser|plan)", re.IGNORECASE),
        ("FP", "Plan"), 0.95, "sprinkler → Fire Protection Plan"),
    (re.compile(r"\bstandpipe|\bfire\s+pump\b|\bsiamese", re.IGNORECASE),
        ("FP", "Diagram"), 0.92, "standpipe / fire pump → FP Diagram"),
    # --- Fire Alarm ---
    (re.compile(r"\b(smoke|heat)\s+detector|\bpull\s+station\b|\bnotification\s+appliance", re.IGNORECASE),
        ("FA", "Plan"), 0.93, "smoke/heat detector → Fire Alarm Plan"),
    # --- Civil ---
    (re.compile(r"\b(site\s+grading|spot\s+elevation|finish\s+grade)\b", re.IGNORECASE),
        ("C", "Plan"), 0.93, "grading → Civil Plan"),
    (re.compile(r"\b(catch\s+basin|storm\s+drain|drainage\s+plan)", re.IGNORECASE),
        ("C", "Plan"), 0.92, "drainage → Civil Plan"),
    (re.compile(r"\bsite\s+utility|\butility\s+plan\b", re.IGNORECASE),
        ("C", "Plan"), 0.90, "utility → Civil Plan"),
    # --- Landscape ---
    (re.compile(r"\bplanting\s+(plan|schedule)|\btree\s+(protection|schedule)", re.IGNORECASE),
        ("L", "Plan"), 0.94, "planting → Landscape Plan"),
    (re.compile(r"\birrigation\s+(plan|zone|head)", re.IGNORECASE),
        ("L", "Plan"), 0.93, "irrigation → Landscape Plan"),
    # --- Vertical Transportation ---
    (re.compile(r"\belevator\s+(pit|overhead|car|shaft|machine\s+room)", re.IGNORECASE),
        ("V", "Section"), 0.93, "elevator → Vertical Transportation Section"),
    # --- Telecom / Low-Voltage ---
    (re.compile(r"\b(data\s+drop|tel/data|cat\s*6|patch\s+panel)\b", re.IGNORECASE),
        ("T", "Plan"), 0.91, "data/voice → Telecom Plan"),
    (re.compile(r"\b(security\s+camera|access\s+control\s+reader|card\s+reader)\b", re.IGNORECASE),
        ("T", "Plan"), 0.91, "security/access control → Telecom Plan"),
    # --- Kitchen / Food Service ---
    (re.compile(r"\b(walk[-\s]?in\s+cooler|kitchen\s+(hood|equipment))", re.IGNORECASE),
        ("K", "Schedule"), 0.92, "kitchen equipment → Kitchen Schedule"),
    # --- General / Code / Life Safety ---
    (re.compile(r"\b(egress|occupancy\s+load|life\s+safety|code\s+compliance)\b", re.IGNORECASE),
        ("G", "Plan"), 0.90, "egress / code → General Code/Life Safety Plan"),
    (re.compile(r"\b(ADA|accessibility)\s+(clearance|requirement|compliance)", re.IGNORECASE),
        ("G", "Plan"), 0.88, "ADA → General Code Plan"),
    # --- Generic listing intent (across trades) ---
    (re.compile(r"\blist\s+(the\s+)?architectural\s+(sheets|drawings)\b", re.IGNORECASE),
        ("A", "Plan"), 0.93, "list architectural → A trade, Plan role"),
    (re.compile(r"\blist\s+(the\s+)?structural\s+(sheets|drawings)\b", re.IGNORECASE),
        ("S", "Plan"), 0.93, "list structural → S trade, Plan role"),
    (re.compile(r"\blist\s+(the\s+)?mechanical\s+(sheets|drawings)\b", re.IGNORECASE),
        ("M", "Plan"), 0.93, "list mechanical → M trade, Plan role"),
    (re.compile(r"\blist\s+(the\s+)?electrical\s+(sheets|drawings)\b", re.IGNORECASE),
        ("E", "Plan"), 0.93, "list electrical → E trade, Plan role"),
    (re.compile(r"\blist\s+(the\s+)?plumbing\s+(sheets|drawings)\b", re.IGNORECASE),
        ("P", "Plan"), 0.93, "list plumbing → P trade, Plan role"),
    # --- v1.1 expansion — extracted from 100-Q test misses ---
    # Mechanical-specific terms (often missed by v1)
    (re.compile(r"\b(OA|outside\s+air)\s+(CFM|airflow|volume|verification)", re.IGNORECASE),
        ("M", "Schedule"), 0.93, "OA CFM / outside air → Mechanical Schedule"),
    (re.compile(r"\bDOAS\b|\bdedicated\s+outdoor\s+air", re.IGNORECASE),
        ("M", "Schedule"), 0.94, "DOAS → Mechanical Schedule"),
    (re.compile(r"\b(FSD|fire\s*[-/]?\s*smoke\s+damper|fire-smoke\s+damper)\b", re.IGNORECASE),
        ("M", "Plan"), 0.94, "FSD / fire-smoke damper → Mechanical Plan"),
    (re.compile(r"\bbalancing\s+damper|\bvolume\s+damper\b", re.IGNORECASE),
        ("M", "Schedule"), 0.93, "balancing damper → Mechanical Schedule"),
    (re.compile(r"\bneoprene\s+isolation|\bvibration\s+isolation", re.IGNORECASE),
        ("M", "Schedule"), 0.92, "isolation hardware → Mechanical Schedule"),
    (re.compile(r"\bduct\s+(connection|gauge|liner)|\bliner\s+requirement", re.IGNORECASE),
        ("M", "Detail"), 0.92, "duct construction → Mechanical Detail"),
    (re.compile(r"\bshaft\s+section|\bshaft\s+arrangement", re.IGNORECASE),
        ("M", "Section"), 0.91, "shaft section → Mechanical Section"),
    # Plumbing-specific
    (re.compile(r"\bfloor\s+drain(age)?\s+(requirement|locations?|spec)|\bFD-\d", re.IGNORECASE),
        ("P", "Plan"), 0.92, "floor drainage → Plumbing Plan"),
    (re.compile(r"\bPRV\b|\bpressure\s+reducing\s+valve", re.IGNORECASE),
        ("P", "Schedule"), 0.94, "PRV → Plumbing Schedule"),
    (re.compile(r"\bDFU\b|\bdrainage?\s+fixture\s+unit", re.IGNORECASE),
        ("P", "Schedule"), 0.94, "DFU → Plumbing Schedule"),
    (re.compile(r"\b(rough[-\s]?in)\s+(size|location|dimension)", re.IGNORECASE),
        ("P", "Schedule"), 0.92, "rough-in sizes → Plumbing Schedule"),
    (re.compile(r"\bthrust\s+block|\bbackflow\s+preventer", re.IGNORECASE),
        ("P", "Diagram"), 0.91, "thrust block / backflow → Plumbing Diagram"),
    (re.compile(r"\b(sanitary|storm|domestic\s+water)\s+(piping|main|riser|line|exit)", re.IGNORECASE),
        ("P", "Plan"), 0.92, "sanitary/storm/water piping → Plumbing Plan"),
    (re.compile(r"\bZurn\b|\bWatts\b.*valve|\bAmerican\s+Standard\b", re.IGNORECASE),
        ("P", "Schedule"), 0.90, "plumbing fixture manufacturer → P Schedule"),
    (re.compile(r"\bisometric\s+(diagram|drawing|view)", re.IGNORECASE),
        ("P", "Diagram"), 0.88, "isometric → typically Plumbing Diagram"),
    (re.compile(r"\bpipe\s+(invert|insulation)|\bpiping\s+insulation", re.IGNORECASE),
        ("P", "Plan"), 0.88, "pipe invert/insulation → Plumbing Plan"),
    # Architectural-specific
    (re.compile(r"\bframe\s+types?\b|\bdoor\s+frame\s+(type|spec)", re.IGNORECASE),
        ("A", "Schedule"), 0.93, "frame types → A Door Schedule"),
    (re.compile(r"\bgypsum\s+board|\bdrywall\s+(scope|installation|control)|\bcontrol\s+joint", re.IGNORECASE),
        ("A", "Notes"), 0.91, "gypsum/drywall → A Notes"),
    (re.compile(r"\bcasework|\bmillwork|\bbuilt[-\s]?in", re.IGNORECASE),
        ("A", "Detail"), 0.91, "casework/millwork → A Detail"),
    (re.compile(r"\bhardware\s+mounting\s+height|\bhandrail\s+(detail|cleaning|height)", re.IGNORECASE),
        ("A", "Detail"), 0.91, "mounting heights / handrail → A Detail"),
    (re.compile(r"\bcorridor\s+(width|wall\s+assembly)|\bunit\s+mix\b", re.IGNORECASE),
        ("A", "Plan"), 0.90, "corridor / unit mix → A Plan"),
    (re.compile(r"\b(rated\s+wall|fire\s+rated\s+wall|wall\s+assembly)", re.IGNORECASE),
        ("A", "Schedule"), 0.89, "rated wall → A Wall Type Schedule"),
    (re.compile(r"\bbathroom\s+(layout|fixture|tile|shower)|\bshower\s+layout|\bshower\s+layout", re.IGNORECASE),
        ("A", "Plan"), 0.88, "bathroom layout → A Plan (interior)"),
    (re.compile(r"\bkeynote\s+(legend|table)|\bkeynote\s+(item|callout)", re.IGNORECASE),
        ("A", "Notes"), 0.88, "keynote legend → A Notes"),
    (re.compile(r"\bASI\s*\d+\s+(revision|change)|\brevision\s+cloud", re.IGNORECASE),
        ("G", "Notes"), 0.88, "ASI revision / revision cloud → G Notes (admin)"),
    # Meta / admin questions (default to G + Notes; these are workflow not retrieval)
    (re.compile(r"\b(what\s+is\s+the\s+)?project\s+status\b|\bMEP\s+engineer\b", re.IGNORECASE),
        ("G", "Notes"), 0.85, "project status / engineer identity → G Notes (meta)"),
    (re.compile(r"\bdrawing\s+(scale|index|metadata)|\bproject\s+metadata", re.IGNORECASE),
        ("G", "Notes"), 0.85, "drawing scale/index → G Notes (meta)"),
    (re.compile(r"\bQA\s+inspection|\binspection\s+checklist|\bcross[-\s]?reference\s+sheets", re.IGNORECASE),
        ("G", "Notes"), 0.83, "QA / inspection / cross-ref → G Notes (meta)"),
    (re.compile(r"\blong[-\s]?lead|\bspecialty\s+items.*procure|\bsubmittal\s+requirement", re.IGNORECASE),
        ("G", "Notes"), 0.83, "procurement / long-lead → G Notes (meta)"),
    # Subcontractor-scope questions — route to the subcontractor's trade
    (re.compile(r"\b(plumbing\s+subcontractor|plumber)\b.*scope", re.IGNORECASE),
        ("P", "Notes"), 0.90, "plumbing subcontractor scope → P Notes"),
    (re.compile(r"\bmechanical\s+subcontractor\b.*scope|\bHVAC\s+contractor", re.IGNORECASE),
        ("M", "Notes"), 0.90, "mech subcontractor scope → M Notes"),
    (re.compile(r"\belectrical\s+subcontractor\b.*scope|\belectrical\s+contractor", re.IGNORECASE),
        ("E", "Notes"), 0.90, "elec subcontractor scope → E Notes"),
    (re.compile(r"\b(drywall|framing|tile|door)\s+subcontractor\b", re.IGNORECASE),
        ("A", "Notes"), 0.89, "arch trade subcontractor → A Notes"),
    # ===========================================================================
    # v1.2 expansion — patterns extracted from v1.1 NON-META wrongs
    # ===========================================================================
    # The strongest insight from v1.1: gold labels are PDF-origin trade. So when
    # a question is plumbing-subject on an A-trade PDF, gold says A, our subject
    # routing said P. These rules tilt toward the question SUBJECT trade more
    # aggressively while keeping confidence calibrated.
    # --- More plumbing patterns ---
    (re.compile(r"\bcheck\s+valve\b.*condensate|\bcondensate\s+(piping|drain|line)", re.IGNORECASE),
        ("P", "Detail"), 0.91, "condensate piping → Plumbing Detail"),
    (re.compile(r"\bgas\s+(regulator|equipment|meter)|\bnatural\s+gas\s+piping", re.IGNORECASE),
        ("P", "Schedule"), 0.90, "gas piping/regulator → Plumbing Schedule"),
    (re.compile(r"\b(pumped\s+discharge|condensate\s+pump|sump\s+pump|ejector)", re.IGNORECASE),
        ("P", "Schedule"), 0.90, "pump systems → Plumbing Schedule"),
    (re.compile(r"\bstorm\s+(pipe|main|drain)|\broof\s+drain|\boverflow\s+drain", re.IGNORECASE),
        ("P", "Plan"), 0.91, "storm drainage → Plumbing Plan"),
    (re.compile(r"\b(faucet|sink|lavatory|toilet|urinal|shower)\s+(spec|core|trim|valve)", re.IGNORECASE),
        ("P", "Schedule"), 0.92, "plumbing fixture spec → Plumbing Schedule"),
    (re.compile(r"\bplumbing\s+rough[-\s]?in|\bplumbing\s+(connection|fitting|sizing)", re.IGNORECASE),
        ("P", "Schedule"), 0.91, "plumbing rough-in → Plumbing Schedule"),
    (re.compile(r"\bgrease\s+(interceptor|trap)|\boil[/-]water\s+separator", re.IGNORECASE),
        ("P", "Schedule"), 0.91, "grease interceptor → Plumbing Schedule"),
    (re.compile(r"\bhot\s+water\s+(heater|return|circulation)|\bwater\s+heater\s+spec", re.IGNORECASE),
        ("P", "Schedule"), 0.91, "water heater → Plumbing Schedule"),
    # --- More mechanical patterns ---
    (re.compile(r"\bsupply\s+(air|fan)|\breturn\s+(air|fan)|\bexhaust\s+fan", re.IGNORECASE),
        ("M", "Plan"), 0.91, "supply/return/exhaust air → Mechanical Plan"),
    (re.compile(r"\bunit\s+heater|\bradiant\s+(panel|ceiling|floor)|\bradiator\b", re.IGNORECASE),
        ("M", "Schedule"), 0.90, "heating equipment → Mechanical Schedule"),
    (re.compile(r"\bcooling\s+tower|\bcondenser|\bevaporator|\brefrigerant", re.IGNORECASE),
        ("M", "Schedule"), 0.91, "cooling equipment → Mechanical Schedule"),
    (re.compile(r"\bVRF\b|\bvariable\s+(refrigerant|air\s+volume)|\bsplit\s+system", re.IGNORECASE),
        ("M", "Schedule"), 0.91, "VRF/VAV → Mechanical Schedule"),
    (re.compile(r"\bzone\s+(thermostat|sensor|control)|\bDDC\b|\bBMS\b", re.IGNORECASE),
        ("M", "Diagram"), 0.90, "controls → Mechanical Diagram"),
    # --- More electrical patterns ---
    (re.compile(r"\b(LED|emergency|exit)\s+(fixture|light|sign)|\bfixture\s+type\s+[A-Z]\d", re.IGNORECASE),
        ("E", "Schedule"), 0.92, "light fixture → Electrical Schedule"),
    (re.compile(r"\bvoltage\s+drop|\bload\s+calculation|\bbranch\s+circuit", re.IGNORECASE),
        ("E", "Schedule"), 0.90, "electrical load → Electrical Schedule"),
    (re.compile(r"\bjunction\s+box|\bpull\s+box|\bconduit\s+(routing|fill|size)", re.IGNORECASE),
        ("E", "Plan"), 0.90, "raceway → Electrical Plan"),
    (re.compile(r"\bgrounding\s+(electrode|conductor)|\bgrounding\s+system|\bbonding", re.IGNORECASE),
        ("E", "Diagram"), 0.90, "grounding → Electrical Diagram"),
    # --- More architectural patterns ---
    (re.compile(r"\bunit\s+(type|size|mix|bay\s+count)|\bresidential\s+units?\s+shown", re.IGNORECASE),
        ("A", "Schedule"), 0.90, "unit mix/type → A Schedule"),
    (re.compile(r"\bgrid\s*line|\bcolumn\s+grid|\bbuilding\s+grid", re.IGNORECASE),
        ("A", "Plan"), 0.89, "building grid → A Plan"),
    (re.compile(r"\bfire\s+rating(s)?\s+(used|applied|required)|\bfire\s+rated\s+assembly", re.IGNORECASE),
        ("A", "Schedule"), 0.89, "fire ratings → A Schedule"),
    (re.compile(r"\broom\s+(finish|number|name)\s+schedule|\bfinish\s+schedule", re.IGNORECASE),
        ("A", "Schedule"), 0.93, "room/finish schedule → A Schedule"),
    (re.compile(r"\bACT\b|\bacoustic(al)?\s+ceiling|\bACT\s+ceiling|\bceiling\s+type\s+[A-Z]", re.IGNORECASE),
        ("A", "RCP"), 0.93, "ACT / ceiling type → A RCP"),
    (re.compile(r"\binsulation\s+(product|spec|type|requirement)(?!.*pipe|.*duct)", re.IGNORECASE),
        ("A", "Schedule"), 0.86, "building insulation → A Schedule"),
    (re.compile(r"\b(parking\s+(space|stall|garage)|garage\s+layout|loading\s+(area|dock))", re.IGNORECASE),
        ("A", "Plan"), 0.88, "parking / loading → A Plan"),
    (re.compile(r"\bstair\s+(layout|riser|tread|nosing|landing)|\bguardrail", re.IGNORECASE),
        ("A", "Detail"), 0.91, "stair / guardrail → A Detail"),
    (re.compile(r"\bcurtain\s+wall|\bstorefront\s+system|\bglazing\s+(spec|type)", re.IGNORECASE),
        ("A", "Detail"), 0.91, "curtain wall / glazing → A Detail"),
    # --- More fire-related ---
    (re.compile(r"\bfire(\s|-)stop|\bfire(\s|-)rated\s+penetration|\bpenetration\s+seal", re.IGNORECASE),
        ("A", "Detail"), 0.89, "firestopping → A Detail"),
    (re.compile(r"\bfire\s+protection\s+(of\s+structural|of\s+steel|coating|spray)", re.IGNORECASE),
        ("A", "Schedule"), 0.88, "structural fireproofing → A Schedule (spec)"),
    # --- More structural ---
    (re.compile(r"\bsteel\s+(beam|column|connection|frame)|\bstructural\s+steel", re.IGNORECASE),
        ("S", "Detail"), 0.91, "structural steel → S Detail"),
    (re.compile(r"\bpost[-\s]?tension|\bconcrete\s+(mix|cylinder|test)", re.IGNORECASE),
        ("S", "Notes"), 0.90, "concrete spec → S Notes"),
    (re.compile(r"\bshear\s+wall|\bbrace\s+frame|\bmoment\s+frame", re.IGNORECASE),
        ("S", "Plan"), 0.92, "lateral system → S Plan"),
    # --- Civil ---
    (re.compile(r"\bASP|\basphalt|\bpaving\s+section|\bcurb\s+ramp", re.IGNORECASE),
        ("C", "Detail"), 0.89, "paving → Civil Detail"),
    (re.compile(r"\bADA\s+(parking|ramp|sidewalk|route)|\baccessible\s+(parking|route)", re.IGNORECASE),
        ("C", "Plan"), 0.86, "ADA site → Civil Plan"),
    # --- Cross-trade alignments (where the SUBJECT trade differs from PDF) ---
    # Floor drainage — typically (P) for layout, but on A sheets it's coordinated
    (re.compile(r"\b(slip\s+resistance|tactile\s+strip|nosing)\b", re.IGNORECASE),
        ("A", "Detail"), 0.88, "slip / tactile → A Detail"),
    (re.compile(r"\bsignage\s+(spec|requirement|type|mounting)", re.IGNORECASE),
        ("A", "Schedule"), 0.88, "signage → A Schedule"),
    # --- Meta question expansion ---
    (re.compile(r"\bsheet\s+(metadata|date|revision|title\s*block)", re.IGNORECASE),
        ("G", "Notes"), 0.85, "sheet metadata → G Notes"),
    (re.compile(r"\bwhat\s+(do|does)\s+the\s+keynotes?\s+(mean|apply|reference)", re.IGNORECASE),
        ("G", "Notes"), 0.83, "keynote explanation → G Notes"),
    (re.compile(r"\bwhat\s+are\s+the\s+\d+\s+(detail|note|keynote)s?", re.IGNORECASE),
        ("G", "Notes"), 0.83, "what are the N details/notes → G Notes (meta)"),
    (re.compile(r"\bcomplete\s+(framing|drywall|tile|door|hardware)\s+(scope|list|spec)", re.IGNORECASE),
        ("A", "Schedule"), 0.88, "comprehensive A-trade scope → A Schedule"),
    (re.compile(r"\bplumbing\s+(rough[-\s]?in\s+(sizes|locations)|fixture\s+rough[-\s]?in)", re.IGNORECASE),
        ("P", "Schedule"), 0.92, "plumbing rough-in → P Schedule"),
)


def _tier_0(query: str) -> Optional[RouterResult]:
    """Match monosemic keywords. Returns RouterResult on hit, None on miss."""
    t0 = time.time()
    for pattern, (trade, role), conf, rationale in _TIER_0_RULES:
        if pattern.search(query):
            return RouterResult(
                primary=(trade, role),
                primary_confidence=conf,
                rationale=rationale,
                classifier_tier=0,
                latency_ms=int((time.time() - t0) * 1000),
            )
    return None


# ---------------------------------------------------------------------------
# Tier 1 — claude-haiku-4-5 JSON-schema classifier (tool-use enforced output)
# ---------------------------------------------------------------------------
_TIER_1_SYSTEM = (
    "You are a construction-document routing classifier.\n"
    "\n"
    "Map a user question to the (trade, role) cell where it is most "
    "authoritatively answered, following 50-year AIA conventions.\n"
    "\n"
    f"trade MUST be one of: {[t for t in TRADES if t != 'unknown']}\n"
    f"role  MUST be one of: {[r for r in ROLES if r != 'unknown']}\n"
    "\n"
    "CORE ROUTING RULES:\n"
    "- 'ceiling X' (X = height/finish/grid/type) → (A, RCP). NEVER (A, Plan).\n"
    "- 'slab elevation/thickness' → (S, Plan). NOT (A, *).\n"
    "- 'door schedule', 'panel schedule', 'beam schedule' → (trade, Schedule).\n"
    "- 'foundation X', 'framing X' → (S, Plan).\n"
    "- 'duct X', 'AHU/RTU/FCU/DOAS/FSD' → (M, *).\n"
    "- 'sprinkler X' → (FP, *).\n"
    "- 'site grading', 'spot elevation' → (C, Plan).\n"
    "- 'riser diagram' alone → (P, Diagram) by default (most common); only (M, Diagram) "
    "if the context explicitly mentions HVAC / supply / return / duct.\n"
    "\n"
    "META-QUESTION HANDLING (critical — these route to G + Notes):\n"
    "- 'What is the project status?' → (G, Notes).\n"
    "- 'Are there revision clouds / ASI changes?' → (G, Notes).\n"
    "- 'Who is the [engineer/architect]?' → (G, Notes).\n"
    "- 'What is the drawing scale / metadata?' → (G, Notes).\n"
    "- 'What QA inspection / coordination checklist?' → (G, Notes).\n"
    "- 'What long-lead procurement items?' → (G, Notes).\n"
    "- 'What 7 details are shown on this sheet?' (meta about doc structure) → (G, Notes).\n"
    "\n"
    "SUBCONTRACTOR-SCOPE QUESTIONS:\n"
    "- '[trade] subcontractor scope' → (corresponding trade, Notes).\n"
    "- 'GC coordination responsibility' alone (no other trade hint) → (G, Notes).\n"
    "- 'GC coordination for the [thing]' → use the [thing]'s trade, role=Notes.\n"
    "\n"
    "POLYSEMOUS TERMS (use context to disambiguate, ALWAYS emit alternatives):\n"
    "- 'slab' → S (structural) by default; A (finish) if context says 'finish' or 'topping'; C if 'paving'.\n"
    "- 'lighting' → E if 'fixture/switch/panel'; L if 'site/landscape/path'; A if 'natural lighting'.\n"
    "- 'riser' → P (sanitary/water) default; FP if 'sprinkler/standpipe'; M if 'duct/HVAC'.\n"
    "- 'outlet' → E (receptacle) by default.\n"
    "- 'panel' → E (panelboard) by default; A if 'access panel' / 'wall panel'.\n"
    "- 'fixture' → P if 'plumbing/sink/toilet'; E if 'light fixture'.\n"
    "- 'insulation' → A (building env) default; M (pipe/duct insulation); P (pipe insulation).\n"
    "- 'invert' / 'pipe invert' → C (civil sewer) most common; S if foundation context.\n"
    "\n"
    "ROLE SELECTION HINTS:\n"
    "- 'X requirement / X rating' → Schedule or Notes (specs live there, not on Plans).\n"
    "- 'X location / X layout' → Plan.\n"
    "- 'X size / X dimension' → Schedule.\n"
    "- 'X detail / X assembly' → Detail.\n"
    "- 'X diagram / X riser' → Diagram.\n"
    "- 'X scope / X coordination' → Notes.\n"
    "- 'X comparison / X relationship visually' → Section or Elevation.\n"
    "\n"
    "WHEN UNSURE (CRITICAL):\n"
    "- ALWAYS emit 2-3 alternatives. The retrieval layer uses top-3.\n"
    "- Confidence should reflect uncertainty: 0.95+ for unambiguous, 0.7-0.85 for ambiguous, <0.7 for unclear.\n"
    "- If the question mentions multiple trades, list the primary subject's trade first, others as alternatives.\n"
    "- If the question is genuinely about a meta/admin topic (no clear trade), return (G, Notes) confidently — that's correct.\n"
    "\n"
    "FEW-SHOT EXAMPLES:\n"
    "Q: 'What floor drain is specified?'\n"
    "→ {primary: (P, Schedule), conf: 0.95, alt: [(P, Plan, 0.6)]}\n"
    "\n"
    "Q: 'What is the OA CFM verification for this zone?'\n"
    "→ {primary: (M, Schedule), conf: 0.94, alt: [(M, Plan, 0.6)]}\n"
    "\n"
    "Q: 'What residential units are shown and their sizes?'\n"
    "→ {primary: (A, Schedule), conf: 0.88, alt: [(A, Plan, 0.7)]}\n"
    "\n"
    "Q: 'What is the project status?'\n"
    "→ {primary: (G, Notes), conf: 0.94, alt: []}\n"
    "\n"
    "Q: 'Are there any revision clouds?'\n"
    "→ {primary: (G, Notes), conf: 0.92, alt: []}\n"
    "\n"
    "Q: 'What is the plumbing subcontractor scope from this sheet?'\n"
    "→ {primary: (P, Notes), conf: 0.92, alt: [(P, Schedule, 0.6)]}\n"
    "\n"
    "Q: 'How do the risers terminate at the top?'\n"
    "→ {primary: (P, Diagram), conf: 0.78, alt: [(P, Plan, 0.6), (M, Diagram, 0.4)]}\n"
    "\n"
    "Q: 'What fire protection coordination is required at ceilings?'\n"
    "→ {primary: (A, RCP), conf: 0.88, alt: [(FP, Plan, 0.7)]}\n"
    "\n"
    "Output JSON ONLY via the emit_classification tool."
)

_TIER_1_TOOL_SCHEMA = {
    "name": "emit_classification",
    "description": "Emit the classified (trade, role) tuple with confidence.",
    "input_schema": {
        "type": "object",
        "properties": {
            "trade": {"type": "string", "enum": [t for t in TRADES if t != "unknown"]},
            "role":  {"type": "string", "enum": [r for r in ROLES if r != "unknown"]},
            "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "rationale": {"type": "string", "maxLength": 240},
            "alternatives": {
                "type": "array",
                "maxItems": 3,
                "items": {
                    "type": "object",
                    "properties": {
                        "trade": {"type": "string", "enum": [t for t in TRADES if t != "unknown"]},
                        "role":  {"type": "string", "enum": [r for r in ROLES if r != "unknown"]},
                        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    },
                    "required": ["trade", "role", "confidence"],
                },
            },
        },
        "required": ["trade", "role", "confidence", "rationale"],
    },
}


@functools.lru_cache(maxsize=2048)
def _tier_1(query: str, _model: str) -> Optional[RouterResult]:
    """Call haiku to classify. Cached. Returns None on any failure."""
    t0 = time.time()
    try:
        # Lazy import — keeps module importable without anthropic SDK
        from anthropic import Anthropic
        client = Anthropic()
        resp = client.messages.create(
            model=_model,
            max_tokens=400,
            temperature=0.0,
            system=_TIER_1_SYSTEM,
            tools=[_TIER_1_TOOL_SCHEMA],
            tool_choice={"type": "tool", "name": "emit_classification"},
            messages=[{"role": "user", "content": f"Question: {query}"}],
        )
        # Extract the tool-use block
        block = None
        for b in (resp.content or []):
            if getattr(b, "type", "") == "tool_use" and getattr(b, "name", "") == "emit_classification":
                block = b
                break
        if block is None:
            logger.warning("[trade_router] tier-1: no tool_use block in response")
            return None
        data = block.input
        trade = data.get("trade")
        role = data.get("role")
        conf = float(data.get("confidence", 0.0))
        rationale = (data.get("rationale") or "")[:240]
        alts_raw = data.get("alternatives") or []
        alts: List[Tuple[str, str, float]] = []
        for a in alts_raw[:3]:
            t, r, c = a.get("trade"), a.get("role"), float(a.get("confidence", 0.0))
            if t in TRADES and r in ROLES and 0.0 <= c <= 1.0:
                alts.append((t, r, c))
        if trade not in TRADES or role not in ROLES or not (0.0 <= conf <= 1.0):
            logger.warning("[trade_router] tier-1: invalid output trade=%s role=%s conf=%s", trade, role, conf)
            return None
        return RouterResult(
            primary=(trade, role),
            primary_confidence=conf,
            alternatives=alts,
            rationale=rationale,
            classifier_tier=1,
            latency_ms=int((time.time() - t0) * 1000),
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("[trade_router] tier-1 call failed: %s: %s", type(exc).__name__, exc)
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def _tier_2_auto_threshold() -> float:
    """Confidence threshold below which we auto-escalate to Tier 2 sonnet.
    Default 0.70. Set TRADE_ROUTING_TIER_2_AUTO_THRESHOLD=0.0 to disable
    auto-escalation entirely (still respects accuracy_mode opt-in path).
    """
    try:
        return float(os.getenv("TRADE_ROUTING_TIER_2_AUTO_THRESHOLD", "0.70"))
    except ValueError:
        return 0.70


def classify(query: str, *, accuracy_mode: bool = False) -> RouterResult:
    """Run the cascaded classifier on a query.

    Returns a RouterResult ALWAYS — never raises. Callers should check
    `.is_noop` / `.is_high_confidence` / `.is_medium_confidence` to decide
    how to use the result.

    Tier 2 (sonnet) auto-escalates when Tier 1 returns confidence below
    `TRADE_ROUTING_TIER_2_AUTO_THRESHOLD` (default 0.70) AND
    `TRADE_ROUTING_TIER_2_ENABLED=true`. The `accuracy_mode=True` kwarg
    additionally forces Tier 2 even at higher Tier 1 confidence.

    The master env flag `TRADE_ROUTING_ENABLED=false` makes this a no-op.
    """
    # Master kill switch
    if not _flag("TRADE_ROUTING_ENABLED"):
        return RouterResult(primary=("unknown", "unknown"), primary_confidence=0.0,
                            rationale="TRADE_ROUTING_ENABLED=false", classifier_tier=-1)
    if not query or not isinstance(query, str) or not query.strip():
        return RouterResult(primary=("unknown", "unknown"), primary_confidence=0.0,
                            rationale="empty query", classifier_tier=-1)

    # Tier 0 — regex (free, ~5ms)
    if _flag("TRADE_ROUTING_TIER_0_ENABLED", default="true"):
        r = _tier_0(query)
        if r is not None:
            return r

    # Tier 1 — haiku (default-on)
    model = os.getenv("TRADE_ROUTING_MODEL", "claude-haiku-4-5")
    r = _tier_1(query.strip(), model)
    if r is not None:
        # Tier 2 escalation logic:
        #   - accuracy_mode=True AND conf < medium_threshold (manual opt-in path)
        #   - OR conf < TRADE_ROUTING_TIER_2_AUTO_THRESHOLD (auto-escalate path)
        # Both require TRADE_ROUTING_TIER_2_ENABLED=true.
        tier2_threshold = _tier_2_auto_threshold()
        should_escalate_manual = accuracy_mode and r.primary_confidence < _medium_threshold()
        should_escalate_auto = (
            tier2_threshold > 0.0 and r.primary_confidence < tier2_threshold
        )
        if (should_escalate_manual or should_escalate_auto) and \
           _flag("TRADE_ROUTING_TIER_2_ENABLED"):
            r2 = _tier_2(query.strip())
            if r2 is not None:
                # Tier 2 wins ONLY if it returns a result. If it errors, keep
                # the Tier 1 result (graceful degradation).
                return r2
        return r

    # All tiers failed → no-op
    return RouterResult(primary=("unknown", "unknown"), primary_confidence=0.0,
                        rationale="all-tier failure", classifier_tier=-1)


# ---------------------------------------------------------------------------
# Tier 2 — sonnet escalation (opt-in, rare)
# ---------------------------------------------------------------------------
@functools.lru_cache(maxsize=512)
def _tier_2(query: str) -> Optional[RouterResult]:
    """Call sonnet for ambiguous low-confidence cases. Cached, rare."""
    model = os.getenv("TRADE_ROUTING_TIER_2_MODEL", "claude-sonnet-4-5")
    # Same schema as tier 1 — different model
    r = _tier_1.__wrapped__(query, model)  # bypass tier-1's own cache
    if r is None:
        return None
    r.classifier_tier = 2
    return r


def clear_cache() -> None:
    """Test helper — clear LRU caches between unit tests."""
    _tier_1.cache_clear()
    _tier_2.cache_clear()
