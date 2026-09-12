from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql://postgres:postgres@localhost:5432/titiktemu"
    redis_url: str = "redis://localhost:6379"

    mapid_api_base_url: str = "https://api.mapid.io"
    mapid_api_key: str = ""

    gemini_api_key: str = ""
    gemini_model: str = "gemini-flash-2.5"

    study_area_bbox_north: float = -6.1880
    study_area_bbox_south: float = -6.2450
    study_area_bbox_east: float = 106.8300
    study_area_bbox_west: float = 106.7980

    grid_cell_size_m: float = 250.0
    projected_crs: str = "EPSG:32748"


settings = Settings()
