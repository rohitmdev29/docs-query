"""Application settings loaded from environment variables."""
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    groq_api_key: str = ""
    chroma_persist_dir: str = "./data/chroma_db"
    upload_dir: str = "./data/uploads"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"          # ← Ye line add ki — unknown env vars ignore karo


settings = Settings()