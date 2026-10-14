import json
import logging
from typing import List, Optional, Dict, Any
from app.core.config import settings
from app.schemas.map import (
    MapPriceItem,
    MapPriceResponse,
    MapBoundaryFeature,
    MapBoundaryResponse,
    MapGeometry,
)

logger = logging.getLogger("uvicorn.error")


class MapService:
    """
    Manages observed real estate dataset statistics and spatial map boundary visualization.
    Maintains strict separation between observed dataset statistics and ML model predictions.
    """

    def __init__(self):
        self._initialized = False
        self._localities_stats: Dict[str, Any] = {}
        self._districts_stats: Dict[str, Any] = {}
        self._villages: List[Dict[str, Any]] = []

    def initialize(self):
        if self._initialized:
            return

        try:
            try:
                with open(settings.LOCATION_STATISTICS_PATH, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self._localities_stats = data.get("localities", {})
                    self._districts_stats = data.get("districts", {})
            except FileNotFoundError:
                logger.warning(f"Location statistics file not found at '{settings.LOCATION_STATISTICS_PATH}'.")

            try:
                with open(settings.VILLAGES_DATA_PATH, "r", encoding="utf-8") as f:
                    self._villages = json.load(f)
            except FileNotFoundError:
                logger.warning(f"Villages reference file not found at '{settings.VILLAGES_DATA_PATH}'.")

            self._initialized = True
            logger.info(
                f"MapService initialized with {len(self._localities_stats)} locality stats "
                f"and {len(self._districts_stats)} district stats."
            )
        except Exception as e:
            logger.error(f"Failed to initialize MapService: {e}", exc_info=True)
            raise e

    def get_map_prices(
        self,
        state: Optional[str] = None,
        district: Optional[str] = None,
        subdistrict: Optional[str] = None,
        village: Optional[str] = None,
        min_lat: Optional[float] = None,
        max_lat: Optional[float] = None,
        min_lng: Optional[float] = None,
        max_lng: Optional[float] = None,
        level: Optional[str] = None,
        zoom: Optional[int] = None,
    ) -> MapPriceResponse:
        """
        Retrieves aggregated observed dataset statistics for interactive map visualization.
        Applies bounding box and administrative region filtering.
        """
        self.initialize()

        # Determine level based on zoom or explicit parameter
        target_level = level or "locality"
        if zoom is not None and not level:
            if zoom <= 7:
                target_level = "district"
            else:
                target_level = "locality"

        items: List[MapPriceItem] = []

        if target_level == "district":
            for key, stat in self._districts_stats.items():
                s_name = stat.get("state", "")
                d_name = stat.get("district", "")

                # Administrative filters
                if state and state.strip().lower() not in s_name.lower():
                    continue
                if district and district.strip().lower() not in d_name.lower():
                    continue

                items.append(
                    MapPriceItem(
                        location_id=key,
                        location_name=d_name,
                        level="district",
                        state=s_name,
                        district=d_name,
                        property_count=stat.get("property_count", 0),
                        median_price_inr=stat.get("median_price_inr", 0.0),
                        median_price_per_sqft=stat.get("median_price_per_sqft", 0.0),
                        minimum_price_inr=stat.get("minimum_price_inr"),
                        maximum_price_inr=stat.get("maximum_price_inr"),
                        latitude=stat.get("latitude"),
                        longitude=stat.get("longitude"),
                    )
                )
        else:
            # Locality / Village level
            for key, stat in self._localities_stats.items():
                s_name = stat.get("state", "")
                d_name = stat.get("district", "")
                sub_name = stat.get("subdistrict", "")
                v_name = stat.get("village", "")
                lat = stat.get("latitude")
                lng = stat.get("longitude")

                # Administrative filtering
                if state and state.strip().lower() not in s_name.lower():
                    continue
                if district and district.strip().lower() not in d_name.lower():
                    continue
                if subdistrict and subdistrict.strip().lower() not in sub_name.lower():
                    continue
                if village and village.strip().lower() not in v_name.lower():
                    continue

                # Bounding box filtering
                if lat is not None and lng is not None:
                    if min_lat is not None and lat < min_lat:
                        continue
                    if max_lat is not None and lat > max_lat:
                        continue
                    if min_lng is not None and lng < min_lng:
                        continue
                    if max_lng is not None and lng > max_lng:
                        continue

                items.append(
                    MapPriceItem(
                        location_id=key,
                        location_name=v_name,
                        level="locality",
                        state=s_name,
                        district=d_name,
                        subdistrict=sub_name,
                        village=v_name,
                        property_count=stat.get("property_count", 0),
                        median_price_inr=stat.get("median_price_inr", 0.0),
                        median_price_per_sqft=stat.get("median_price_per_sqft", 0.0),
                        minimum_price_inr=stat.get("minimum_price_inr"),
                        maximum_price_inr=stat.get("maximum_price_inr"),
                        latitude=lat,
                        longitude=lng,
                    )
                )

        return MapPriceResponse(
            items=items,
            level=target_level,
            total_locations=len(items),
        )

    def get_map_boundaries(
        self,
        level: str = "locality",
        state_id: Optional[str] = None,
        district_id: Optional[str] = None,
    ) -> MapBoundaryResponse:
        """
        Generates standard GeoJSON FeatureCollection representation for map rendering.
        """
        self.initialize()
        features: List[MapBoundaryFeature] = []

        if level == "locality":
            for v in self._villages:
                s_name = v.get("state_name", "")
                d_name = v.get("district_name", "")
                lat = v.get("latitude")
                lng = v.get("longitude")

                if state_id and state_id.strip().lower() not in s_name.lower():
                    continue
                if district_id and district_id.strip().lower() not in d_name.lower():
                    continue

                if lat is not None and lng is not None:
                    features.append(
                        MapBoundaryFeature(
                            type="Feature",
                            geometry=MapGeometry(type="Point", coordinates=[lng, lat]),
                            properties={
                                "id": v.get("id"),
                                "name": v.get("name"),
                                "mandal": v.get("mandal_name"),
                                "district": d_name,
                                "state": s_name,
                                "pincode": v.get("pincode"),
                                "urban_rural": v.get("urban_rural"),
                                "base_price_sqft": v.get("base_price_sqft"),
                            },
                        )
                    )

        return MapBoundaryResponse(
            type="FeatureCollection",
            features=features,
            level=level,
            count=len(features),
        )


map_service = MapService()
