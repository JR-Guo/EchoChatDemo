from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="", env_file=".env", extra="ignore")

    host: str = Field(default="0.0.0.0", alias="ECHOCHAT_HOST")
    port: int = Field(default=12345, alias="ECHOCHAT_PORT")

    model_path: Path = Field(alias="ECHOCHAT_MODEL_PATH")
    data_dir: Path = Field(alias="ECHOCHAT_DATA_DIR")

    view_classifier_url: str = Field(
        default="http://127.0.0.1:8995", alias="VIEW_CLASSIFIER_URL"
    )

    shared_password: str = Field(alias="SHARED_PASSWORD")
    session_secret: str = Field(alias="SESSION_SECRET")


@lru_cache
def get_settings() -> Settings:
    return Settings()
