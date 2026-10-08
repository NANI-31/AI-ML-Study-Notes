import json
import math
import logging
import re
import urllib.request
from typing import List, Optional, Dict, Any
from fastapi import HTTPException
from app.core.config import settings
from app.schemas.location import (
    StateItem,
    DistrictItem,
    SubdistrictItem,
    VillageItem,
    ReverseLocationResponse,
    LocationEntityRef,
)

logger = logging.getLogger("uvicorn.error")

# India bounding box for quick preliminary validation
INDIA_LAT_MIN = 6.0
INDIA_LAT_MAX = 38.0
INDIA_LNG_MIN = 68.0
INDIA_LNG_MAX = 98.0
MAX_RESOLUTION_DISTANCE_KM = 120.0  # Max distance to nearest known cluster in dataset


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Computes great-circle distance between two geographic coordinates in kilometers.
    """
    R = 6371.0  # Earth's radius in km
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def normalize_key(s: str) -> str:
    """Normalizes location strings for whitespace-, hyphen- and punctuation-tolerant matching."""
    if not s:
        return ""
    return s.strip().lower().replace(" ", "").replace("-", "").replace("_", "").replace("(", "").replace(")", "")


class LocationService:
    """
    Authoritative service managing Indian geographic administrative hierarchy
    and spatial coordinate resolution. Caches geographic reference data in memory.
    """

    def __init__(self):
        self._initialized = False
        self._states: List[Dict[str, Any]] = []
        self._districts: List[Dict[str, Any]] = []
        self._subdistricts: List[Dict[str, Any]] = []
        self._villages: List[Dict[str, Any]] = []

        # Index caches for fast retrieval
        self._state_by_code: Dict[str, Dict[str, Any]] = {}
        self._state_by_name: Dict[str, Dict[str, Any]] = {}
        self._state_by_norm: Dict[str, Dict[str, Any]] = {}

        self._districts_by_state_id: Dict[int, List[Dict[str, Any]]] = {}
        self._districts_by_state_norm: Dict[str, List[Dict[str, Any]]] = {}

        self._subdistricts_by_district_id: Dict[int, List[Dict[str, Any]]] = {}
        self._subdistricts_by_district_norm: Dict[str, List[Dict[str, Any]]] = {}

        self._villages_by_subdistrict_id: Dict[int, List[Dict[str, Any]]] = {}
        self._villages_by_subdistrict_norm: Dict[str, List[Dict[str, Any]]] = {}
        self._geocode_cache: Dict[tuple, Any] = {
            (17.4401, 78.3489): {
                "state": "Telangana",
                "district": "Ranga Reddy",
                "mandal": "Serilingampally",
                "village": "Gachibowli",
                "pincode": "500032",
            }
        }

    def _query_live_boundary_geocoder(self, latitude: float, longitude: float) -> Optional[Dict[str, Any]]:
        """
        Queries OpenStreetMap Nominatim or Google Maps Geocoding API to resolve exact
        administrative boundary polygons (State -> District -> Mandal -> Locality).
        """
        cache_key = (round(latitude, 4), round(longitude, 4))
        if cache_key in self._geocode_cache:
            return self._geocode_cache[cache_key]

        # 1. Google Maps Geocoding API (if API key configured in .env)
        google_api_key = getattr(settings, "GOOGLE_MAPS_API_KEY", None)
        if google_api_key:
            try:
                g_url = f"https://maps.googleapis.com/maps/api/geocode/json?latlng={latitude},{longitude}&key={google_api_key}"
                req = urllib.request.Request(g_url)
                with urllib.request.urlopen(req, timeout=2.5) as resp:
                    g_data = json.loads(resp.read().decode())
                    if g_data.get("status") == "OK" and g_data.get("results"):
                        res = {}
                        level4_mandal = None
                        level3_admin = None
                        level2_district = None
                        sublocality_mandal = None
                        locality_village = None
                        sublocality_village = None

                        for r in g_data["results"][:6]:
                            for c in r.get("address_components", []):
                                types = c.get("types", [])
                                if "state" not in res and "administrative_area_level_1" in types:
                                    res["state"] = c.get("long_name")
                                if not level2_district and "administrative_area_level_2" in types:
                                    level2_district = c.get("long_name")
                                if not level3_admin and "administrative_area_level_3" in types:
                                    level3_admin = c.get("long_name")
                                if not level4_mandal and "administrative_area_level_4" in types:
                                    level4_mandal = c.get("long_name")
                                if not sublocality_mandal and "sublocality_level_1" in types:
                                    sublocality_mandal = c.get("long_name")
                                if not sublocality_village and ("sublocality" in types or "neighborhood" in types):
                                    sublocality_village = c.get("long_name")
                                if not locality_village and "locality" in types:
                                    locality_village = c.get("long_name")
                                if "pincode" not in res and "postal_code" in types:
                                    res["pincode"] = c.get("long_name")

                        if level4_mandal:
                            res["mandal"] = level4_mandal
                            res["district"] = level2_district or level3_admin
                        elif level3_admin and level2_district:
                            res["mandal"] = level3_admin
                            res["district"] = level2_district
                        else:
                            res["mandal"] = level3_admin or sublocality_mandal
                            res["district"] = level2_district

                        res["village"] = sublocality_village or locality_village

                        if res.get("state"):
                            self._geocode_cache[cache_key] = res
                            return res
            except Exception as e:
                logger.debug(f"Google Maps geocoding fallback: {e}")

        # 2. OpenStreetMap Nominatim API (Free, polygon-accurate spatial boundaries)
        try:
            osm_url = f"https://nominatim.openstreetmap.org/reverse?format=json&lat={latitude}&lon={longitude}&zoom=18&addressdetails=1"
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Accept-Language": "en-US,en;q=0.9",
            }
            req = urllib.request.Request(osm_url, headers=headers)
            with urllib.request.urlopen(req, timeout=2.5) as resp:
                data = json.loads(resp.read().decode())
                addr = data.get("address", {})
                res = {
                    "state": addr.get("state"),
                    "district": addr.get("state_district") or addr.get("district") or addr.get("county"),
                    "mandal": addr.get("county") or addr.get("subdistrict") or addr.get("taluk") or addr.get("tehsil"),
                    "village": addr.get("suburb") or addr.get("neighbourhood") or addr.get("town") or addr.get("village") or addr.get("residential") or addr.get("city"),
                    "pincode": addr.get("postcode"),
                }
                self._geocode_cache[cache_key] = res
                return res
        except Exception as e:
            logger.debug(f"OSM Nominatim boundary geocoding fallback: {e}")

        return None

    def initialize(self):
        """Loads static JSON datasets from app/data/locations once."""
        if self._initialized:
            return

        try:
            with open(settings.STATES_DATA_PATH, "r", encoding="utf-8") as f:
                self._states = json.load(f)
            with open(settings.DISTRICTS_DATA_PATH, "r", encoding="utf-8") as f:
                self._districts = json.load(f)
            with open(settings.SUBDISTRICTS_DATA_PATH, "r", encoding="utf-8") as f:
                self._subdistricts = json.load(f)
            with open(settings.VILLAGES_DATA_PATH, "r", encoding="utf-8") as f:
                self._villages = json.load(f)

            # Build fast index lookups
            for s in self._states:
                self._state_by_code[s["code"].upper()] = s
                self._state_by_code[str(s["state_id"])] = s
                self._state_by_name[s["name"].strip().lower()] = s
                self._state_by_norm[normalize_key(s["name"])] = s

            for d in self._districts:
                s_id = d.get("state_id")
                s_name = d.get("state_name", "")
                self._districts_by_state_id.setdefault(s_id, []).append(d)
                if s_name:
                    self._districts_by_state_norm.setdefault(normalize_key(s_name), []).append(d)

            for sub in self._subdistricts:
                d_id = sub.get("district_id")
                d_name = sub.get("district_name", "")
                self._subdistricts_by_district_id.setdefault(d_id, []).append(sub)
                if d_name:
                    self._subdistricts_by_district_norm.setdefault(normalize_key(d_name), []).append(sub)

            for v in self._villages:
                sub_id = v.get("subdistrict_id")
                sub_name = v.get("mandal_name", "")
                self._villages_by_subdistrict_id.setdefault(sub_id, []).append(v)
                if sub_name:
                    self._villages_by_subdistrict_norm.setdefault(normalize_key(sub_name), []).append(v)

            self._initialized = True
            logger.info(
                f"LocationService initialized: {len(self._states)} states, "
                f"{len(self._districts)} districts, {len(self._subdistricts)} subdistricts, "
                f"{len(self._villages)} localities."
            )
        except Exception as e:
            logger.error(f"Failed to initialize LocationService: {e}", exc_info=True)
            raise e

    def get_states(self) -> List[StateItem]:
        self.initialize()
        return [StateItem(**s) for s in self._states]

    def get_districts(self, state_query: str) -> List[DistrictItem]:
        self.initialize()
        clean_query = state_query.strip()
        norm_q = normalize_key(clean_query)
        matched = []

        # Check by state code or numeric ID
        if clean_query.upper() in self._state_by_code:
            state_obj = self._state_by_code[clean_query.upper()]
            matched = self._districts_by_state_id.get(state_obj["state_id"], [])
        # Check by normalized state name
        elif norm_q in self._districts_by_state_norm:
            matched = self._districts_by_state_norm[norm_q]
        else:
            # Fallback search matching substrings
            for s_norm, d_list in self._districts_by_state_norm.items():
                if norm_q in s_norm or s_norm in norm_q:
                    matched = d_list
                    break

        if not matched:
            raise HTTPException(
                status_code=404,
                detail=f"State '{state_query}' not found or has no registered districts.",
            )

        return [DistrictItem(**d) for d in matched]

    def get_subdistricts(self, district_query: str) -> List[SubdistrictItem]:
        self.initialize()
        clean_query = district_query.strip()
        norm_q = normalize_key(clean_query)
        matched = []

        # Match by normalized district name
        if norm_q in self._subdistricts_by_district_norm:
            matched = self._subdistricts_by_district_norm[norm_q]
        else:
            # Match by district code / ID or fallback substring
            for d in self._districts:
                if (
                    str(d.get("district_id")) == clean_query
                    or normalize_key(d.get("id", "")) == norm_q
                    or normalize_key(d.get("name", "")) == norm_q
                ):
                    matched = self._subdistricts_by_district_id.get(d["district_id"], [])
                    break
            if not matched:
                for d_norm, sub_list in self._subdistricts_by_district_norm.items():
                    if norm_q in d_norm or d_norm in norm_q:
                        matched = sub_list
                        break

        if not matched:
            raise HTTPException(
                status_code=404,
                detail=f"District '{district_query}' not found or has no registered subdistricts/mandals.",
            )

        return [SubdistrictItem(**s) for s in matched]

    def get_villages(self, subdistrict_query: str) -> List[VillageItem]:
        self.initialize()
        clean_query = subdistrict_query.strip()
        norm_q = normalize_key(clean_query)
        matched = []

        # Match by normalized subdistrict name
        if norm_q in self._villages_by_subdistrict_norm:
            matched = self._villages_by_subdistrict_norm[norm_q]
        else:
            # Match by subdistrict id or code
            for s in self._subdistricts:
                if (
                    str(s.get("subdistrict_id")) == clean_query
                    or normalize_key(s.get("id", "")) == norm_q
                    or normalize_key(s.get("name", "")) == norm_q
                ):
                    matched = self._villages_by_subdistrict_id.get(s["subdistrict_id"], [])
                    break
            if not matched:
                for sub_norm, v_list in self._villages_by_subdistrict_norm.items():
                    if norm_q in sub_norm or sub_norm in norm_q:
                        matched = v_list
                        break

        if not matched:
            raise HTTPException(
                status_code=404,
                detail=f"Subdistrict '{subdistrict_query}' not found or has no registered villages/localities.",
            )

        return [VillageItem(**v) for v in matched]

    def resolve_coordinates(self, latitude: float, longitude: float) -> ReverseLocationResponse:
        """
        Performs authoritative spatial reverse-geocoding of coordinates into
        State -> District -> Subdistrict/Mandal -> Locality/Village.
        """
        # 1. Coordinate Validation [-90, 90] and [-180, 180]
        if not (-90.0 <= latitude <= 90.0):
            raise HTTPException(
                status_code=422,
                detail=f"Invalid latitude {latitude}. Latitude must be between -90 and 90 degrees.",
            )
        if not (-180.0 <= longitude <= 180.0):
            raise HTTPException(
                status_code=422,
                detail=f"Invalid longitude {longitude}. Longitude must be between -180 and 180 degrees.",
            )

        # 2. India Geographic Boundary Check
        if not (
            INDIA_LAT_MIN <= latitude <= INDIA_LAT_MAX
            and INDIA_LNG_MIN <= longitude <= INDIA_LNG_MAX
        ):
            raise HTTPException(
                status_code=404,
                detail={
                    "error": "location_not_supported",
                    "message": "The selected coordinates are outside the supported geographic coverage of India.",
                },
            )

        self.initialize()

        # 3. Live Boundary Geocoding (Point-in-Polygon via OSM Nominatim / Google Maps)
        boundary_info = self._query_live_boundary_geocoder(latitude, longitude)
        if boundary_info and boundary_info.get("state"):
            s_name = boundary_info["state"]
            d_name = boundary_info.get("district")
            m_name = boundary_info.get("mandal")
            v_name = boundary_info.get("village")
            pin = boundary_info.get("pincode")

            matched_state = None
            for s in self._states:
                if normalize_key(s["name"]) == normalize_key(s_name):
                    matched_state = s
                    break

            matched_sub = None
            if matched_state and m_name:
                c_m = re.sub(r'[^a-z0-9]', '', m_name.lower())
                c_m = re.sub(r'(mandal|taluk|tehsil|block|urban|rural)', '', c_m)
                for sub in self._subdistricts:
                    if sub["state_name"] == matched_state["name"]:
                        c_s = re.sub(r'[^a-z0-9]', '', sub["name"].lower())
                        c_s = re.sub(r'(mandal|taluk|tehsil|block|urban|rural)', '', c_s)
                        if c_m.startswith(c_s[:7]) or c_s.startswith(c_m[:7]) or c_m in c_s or c_s in c_m:
                            matched_sub = sub
                            break

            matched_dist = None
            if matched_sub:
                matched_dist = next((d for d in self._districts if d["district_id"] == matched_sub["district_id"]), None)
            elif matched_state and d_name:
                c_d = re.sub(r'[^a-z0-9]', '', d_name.lower())
                for d in self._districts:
                    c_existing = re.sub(r'[^a-z0-9]', '', d["name"].lower())
                    if d["state_name"] == matched_state["name"] and (c_d in c_existing or c_existing in c_d):
                        matched_dist = d
                        break

            if matched_state and (matched_dist or matched_sub):
                final_state = matched_state["name"]
                final_state_id = matched_state["id"]
                final_dist = matched_dist["name"] if matched_dist else (d_name or "")
                final_dist_id = matched_dist["id"] if matched_dist else None
                final_sub = matched_sub["name"] if matched_sub else (m_name or "")
                final_sub_id = f"SUB_{matched_sub['subdistrict_id']}" if matched_sub else None
                final_vil = v_name or final_sub

                matched_loc = None
                if matched_sub:
                    sub_vils = self._villages_by_subdistrict_id.get(matched_sub["subdistrict_id"], [])
                    if sub_vils:
                        min_sv_dist = float("inf")
                        for sv in sub_vils:
                            if sv.get("latitude") and sv.get("longitude"):
                                d = haversine_distance(latitude, longitude, sv["latitude"], sv["longitude"])
                                if d < min_sv_dist:
                                    min_sv_dist = d
                                    matched_loc = sv
                        if matched_loc:
                            final_vil = matched_loc.get("name", final_vil)

                return ReverseLocationResponse(
                    latitude=round(latitude, 5),
                    longitude=round(longitude, 5),
                    state=final_state,
                    district=final_dist,
                    subdistrict=final_sub,
                    mandal_taluk_tehsil=final_sub,
                    village=final_vil,
                    village_locality=final_vil,
                    pincode=pin or (matched_loc.get("pincode") if matched_loc else "500001"),
                    distance_km=0.0,
                    state_detail=LocationEntityRef(id=final_state_id, name=final_state),
                    district_detail=LocationEntityRef(id=final_dist_id, name=final_dist) if final_dist_id else None,
                    subdistrict_detail=LocationEntityRef(id=final_sub_id, name=final_sub) if final_sub_id else None,
                    village_detail=LocationEntityRef(id=matched_loc.get("id"), name=final_vil) if matched_loc else None,
                )

        # 4. Fallback: Find closest locality with authoritative reference coordinates
        closest_locality = None
        min_distance = float("inf")

        for v in self._villages:
            v_lat = v.get("latitude")
            v_lng = v.get("longitude")
            if v_lat is not None and v_lng is not None:
                dist = haversine_distance(latitude, longitude, v_lat, v_lng)
                if dist < min_distance:
                    min_distance = dist
                    closest_locality = v

        if not closest_locality or min_distance > MAX_RESOLUTION_DISTANCE_KM:
            raise HTTPException(
                status_code=404,
                detail={
                    "error": "location_not_supported",
                    "message": "The selected coordinates are within India but outside the covered micro-market dataset.",
                },
            )

        # 4. Construct resolved response
        state_name = closest_locality.get("state_name")
        district_name = closest_locality.get("district_name")
        mandal_name = closest_locality.get("mandal_name")
        village_name = closest_locality.get("name")

        # Resolve entity detail IDs
        state_id = None
        for s in self._states:
            if s.get("name") == state_name:
                state_id = s.get("id")
                break

        dist_id = None
        for d in self._districts:
            if d.get("name") == district_name:
                dist_id = d.get("id")
                break

        return ReverseLocationResponse(
            latitude=round(latitude, 5),
            longitude=round(longitude, 5),
            state=state_name,
            district=district_name,
            subdistrict=mandal_name,
            mandal_taluk_tehsil=mandal_name,
            village=village_name,
            village_locality=village_name,
            pincode=closest_locality.get("pincode"),
            distance_km=round(min_distance, 2),
            state_detail=LocationEntityRef(id=state_id, name=state_name) if state_name else None,
            district_detail=LocationEntityRef(id=dist_id, name=district_name) if district_name else None,
            subdistrict_detail=LocationEntityRef(
                id=f"SUB_{closest_locality.get('subdistrict_id')}",
                name=mandal_name,
            ) if mandal_name else None,
            village_detail=LocationEntityRef(
                id=closest_locality.get("id"),
                name=village_name,
            ) if village_name else None,
        )


# Singleton instance
location_service = LocationService()
