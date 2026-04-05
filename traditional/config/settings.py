import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    WEB_SEARCH_MODEL = os.getenv("WEB_SEARCH_MODEL", "gpt-4.1")

settings = Settings()
