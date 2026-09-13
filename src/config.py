from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls,
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ):
        # pydantic-settings' default order puts OS env vars ABOVE this
        # project's own .env -- verified this silently breaks the pipeline:
        # a stray, malformed, machine-wide DATABASE_URL env var (unrelated
        # to this project, likely a leftover from something else on this
        # machine) was overriding a correct .env value, and create_engine()
        # crashed with NoSuchModuleError since python-dotenv's load_dotenv()
        # (used elsewhere) also never overrides an existing env var. This
        # repo's own .env should be authoritative for this repo's config,
        # not silently shadowed by unrelated global machine state.
        return (init_settings, dotenv_settings, env_settings, file_secret_settings)

    database_url: str = "postgresql://postgres:postgres@localhost:5432/titiktemu"
    redis_url: str = "redis://localhost:6379"

    mapid_api_base_url: str = "https://api.mapid.io"
    mapid_api_key: str = ""

    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.5-flash"

    study_area_bbox_north: float = -6.1880
    study_area_bbox_south: float = -6.2450
    study_area_bbox_east: float = 106.8300
    study_area_bbox_west: float = 106.7980

    grid_cell_size_m: float = 250.0
    projected_crs: str = "EPSG:32748"


settings = Settings()
