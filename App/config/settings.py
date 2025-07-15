from pydantic_settings import SettingsConfigDict, BaseSettings

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env")

    HUGGINGFACEHUB_API_TOKEN: str       
    HF_MODEL: str                        
    EMBED_MODEL: str                        
    TEMPERATURE: float = 0.7
    MAX_NEW_TOKENS: int = 512

settings = Settings()