from typing import List, Union
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    PROJECT_NAME: str = "ShrimpMate AI Feeding Service"
    VERSION: str = "2.0.0"
    API_V1_STR: str = ""

    # Server settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = False
    WORKERS: int = 1

    # CORS
    CORS_ORIGINS: Union[str, List[str]] = "*"

    # Model file paths
    MODEL_DIR: str = "weights"

    # Logging timezone (synchronized with NestJS Backend / Vietnam GMT+7)
    TIMEZONE: str = "Asia/Ho_Chi_Minh"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    @field_validator("CORS_ORIGINS", mode="before")
    @classmethod
    def assemble_cors_origins(cls, v: Union[str, List[str]]) -> List[str]:
        if isinstance(v, str):
            if v.strip() == "*":
                return ["*"]
            return [origin.strip() for origin in v.split(",") if origin.strip()]
        return v


settings = Settings()
