"""MAPID API client -- Properti Go, Struk Go, Menu Go.

STUB: MAPID's API documentation/access was still pending as of our last
discussion of this ("data belum diberikan... akses belum tersedia"). This
client is structured against a REASONABLE REST convention (not MAPID's
actual documented spec, which we don't have) so the rest of the pipeline
has a stable interface to build against. Replace the request/response
shapes here the moment real API docs are available -- don't assume this
shape is correct."""

import httpx
from src.config import settings


class MapidClient:
    def __init__(self):
        self.base_url = settings.mapid_api_base_url
        self.api_key = settings.mapid_api_key

    def _get(self, path: str, params: dict | None = None) -> dict:
        if not self.api_key:
            raise RuntimeError(
                "MAPID_API_KEY not set -- this client cannot be used until MAPID API "
                "access is available. See ingestion/mapid.py docstring."
            )
        resp = httpx.get(
            f"{self.base_url}{path}",
            headers={"Authorization": f"Bearer {self.api_key}"},
            params=params, timeout=30.0,
        )
        resp.raise_for_status()
        return resp.json()

    def get_properti_go_history(self, grid_id: str) -> dict:
        """Rent price history -- feeds rent_surge_reported. SHAPE UNVERIFIED."""
        return self._get("/properti-go/history", params={"grid_id": grid_id})

    def get_struk_go_transactions(self, grid_id: str) -> dict:
        """Real transaction volume -- feeds the GWR y variable. SHAPE UNVERIFIED."""
        return self._get("/struk-go/transactions", params={"grid_id": grid_id})

    def get_menu_go_pois(self, grid_id: str) -> dict:
        """MAPID's own denser POI feed -- alternative to OSM poi_count. SHAPE UNVERIFIED."""
        return self._get("/menu-go/pois", params={"grid_id": grid_id})
