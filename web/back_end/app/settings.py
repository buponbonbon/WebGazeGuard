from pydantic_settings import BaseSettings
from pydantic import Field

class Settings(BaseSettings):
    app_env: str = Field(default="dev", alias="APP_ENV")
    jwt_secret: str = Field(default="change-me", alias="JWT_SECRET")
    jwt_expires_min: int = Field(default=60*24*7, alias="JWT_EXPIRES_MIN")  # 7 days
    cors_origins: str = Field(default="http://localhost:5173", alias="CORS_ORIGINS")

    ai_cv_endpoint: str | None = Field(default=None, alias="AI_CV_ENDPOINT")
    ai_nlp_endpoint: str | None = Field(default=None, alias="AI_NLP_ENDPOINT")

    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()
