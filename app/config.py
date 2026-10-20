from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# agent/.env, wherever the process is started from. Real environment
# variables still win over it.
ENV_FILE = Path(__file__).resolve().parent.parent / ".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ENV_FILE,
        extra="ignore",
        protected_namespaces=(),
    )

    api_url: str = "http://localhost:8080"
    api_timeout_s: float = 10.0

    model: str = "deepseek/deepseek-v4.1-flash"
    model_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_api_key: str
    # The service key the Go API knows the agent by (AGENT_API_KEY there).
    # Also what POST /tick here requires. Required: without it nothing works.
    agent_api_key: str
    # A hung provider call would otherwise hold the tick lock forever and
    # every later tick would be skipped. Retries cover a transient 5xx/429.
    model_timeout_s: float = 60.0
    model_max_retries: int = 2

    port: int = 8200

    # Browser origins allowed to call /ask and /track/{token}/ask,
    # comma-separated. The console's origin in production goes here.
    cors_origins: str = "http://localhost:3000,http://localhost:3471"

    # How often the proposal pass runs. The question is how long a pending
    # delivery should sit before the agent notices it.
    tick_seconds: float = 180

    # The customer's question box (POST /track/{token}/ask). Who the customer
    # is told to contact, and how many questions a public link may ask.
    operator_name: str = ""
    operator_phone: str = ""
    # Where customers are sent once a link has asked its share of questions.
    operator_email: str = ""
    customer_questions_per_link: int = 8  # per 10 minutes
    customer_questions_per_minute: int = 60  # across all links


settings = Settings()