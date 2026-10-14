import React, { useEffect } from 'react';
import {
  MapContainer,
  TileLayer,
  Marker,
  Popup,
  useMap,
  useMapEvents,
  CircleMarker,
} from 'react-leaflet';
import L from 'leaflet';
import { useAppDispatch, useAppSelector } from '../../hooks/reduxHooks';
import { setCenter, setZoom, setSelectedCoordinates } from '../../store/slices/mapSlice';
import { setFullLocationHierarchy } from '../../store/slices/locationSlice';
import { updateMultipleFields } from '../../store/slices/propertySlice';
import geolocationService from '../../services/geolocationService';
import locationService from '../../services/locationService';
import PricePopup from './PricePopup';
import MapControls from './MapControls';
import MapLegend from './MapLegend';

// Fix Leaflet's default icon URL issues in Vite bundler
delete L.Icon.Default.prototype._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon-2x.png',
  iconUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon.png',
  shadowUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-shadow.png',
});

// Component to dynamically update map center/zoom when Redux state changes
const MapController = ({ center, zoom }) => {
  const map = useMap();
  useEffect(() => {
    if (center) {
      map.setView(center, zoom, { animate: true });
    }
  }, [center, zoom, map]);
  return null;
};

// Map click event listener component
const MapClickHandler = () => {
  const dispatch = useAppDispatch();

  useMapEvents({
    async click(e) {
      const { lat, lng } = e.latlng;
      const res = await geolocationService.reverseGeocodeOnline(lat, lng);

      if (res && res.locality) {
        const loc = res.locality;
        dispatch(
          setSelectedCoordinates({
            latitude: lat,
            longitude: lng,
            name: loc.name,
            district: loc.district_name,
            state: loc.state_name,
            basePriceSqft: loc.base_price_sqft,
            distanceKm: res.distanceKm,
          })
        );

        dispatch(
          setFullLocationHierarchy({
            stateName: loc.state_name,
            districtName: loc.district_name,
            mandalName: loc.mandal_name,
            villageName: loc.name,
            localityObj: loc,
          })
        );

        dispatch(
          updateMultipleFields({
            state: loc.state_name,
            district: loc.district_name,
            mandal_taluk_tehsil: loc.mandal_name,
            village_locality: loc.name,
            pincode: loc.pincode,
            urban_rural: loc.urban_rural || 'Urban',
            distance_to_airport_km: loc.airport_dist_base || 25,
            distance_to_railway_station_km: loc.railway_dist_base || 10,
          })
        );
      } else {
        dispatch(
          setSelectedCoordinates({
            latitude: lat,
            longitude: lng,
            name: 'Custom Coordinates',
          })
        );
      }
    },
  });

  return null;
};

export const IndiaMap = ({
  height = 'h-[500px]',
  showLegend = true,
  showLocalities = true,
  onLocalitySelect,
  className = '',
}) => {
  const dispatch = useAppDispatch();
  const { center, zoom, selectedCoordinates } = useAppSelector((state) => state.map);
  const { selectedState, selectedDistrict } = useAppSelector((state) => state.location);

  const localities = React.useMemo(() => {
    if (selectedDistrict?.name) {
      const distLocs = locationService.getLocalitiesByDistrict(selectedDistrict.name);
      if (distLocs.length > 0) return distLocs;
    }
    if (selectedState?.name) {
      const stateLocs = locationService.getLocalitiesByState(selectedState.name);
      if (stateLocs.length > 0) return stateLocs;
    }
    return locationService.getAllLocalities();
  }, [selectedState, selectedDistrict]);

  const getMarkerColor = (rate) => {
    if (!rate || rate < 5000) return '#10b981'; // emerald-500
    if (rate <= 10000) return '#f59e0b'; // amber-500
    return '#f43f5e'; // rose-500
  };

  const handleLocalityMarkerClick = (loc) => {
    dispatch(setCenter([loc.latitude, loc.longitude]));
    dispatch(setZoom(13));
    dispatch(
      setSelectedCoordinates({
        latitude: loc.latitude,
        longitude: loc.longitude,
        name: loc.name,
        district: loc.district_name,
        state: loc.state_name,
        basePriceSqft: loc.base_price_sqft,
      })
    );
    dispatch(
      setFullLocationHierarchy({
        stateName: loc.state_name,
        districtName: loc.district_name,
        mandalName: loc.mandal_name,
        villageName: loc.name,
        localityObj: loc,
      })
    );
    dispatch(
      updateMultipleFields({
        state: loc.state_name,
        district: loc.district_name,
        mandal_taluk_tehsil: loc.mandal_name,
        village_locality: loc.name,
        pincode: loc.pincode,
        urban_rural: loc.urban_rural || 'Urban',
        distance_to_airport_km: loc.airport_dist_base || 25,
        distance_to_railway_station_km: loc.railway_dist_base || 10,
      })
    );
    if (onLocalitySelect) onLocalitySelect(loc);
  };

  return (
    <div className={`relative w-full ${height} rounded-2xl overflow-hidden shadow-sm border border-slate-200/80 bg-slate-100 ${className}`}>
      <MapContainer
        center={center}
        zoom={zoom}
        scrollWheelZoom={true}
        className="w-full h-full"
      >
        <TileLayer
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        />

        <MapController center={center} zoom={zoom} />
        <MapClickHandler />
        <MapControls />

        {/* Locality Data Points */}
        {showLocalities &&
          localities.map((loc) => (
            <CircleMarker
              key={loc.id}
              center={[loc.latitude, loc.longitude]}
              radius={6}
              pathOptions={{
                fillColor: getMarkerColor(loc.base_price_sqft),
                fillOpacity: 0.85,
                color: '#ffffff',
                weight: 1.5,
              }}
              eventHandlers={{
                click: () => handleLocalityMarkerClick(loc),
              }}
            >
              <Popup>
                <PricePopup locality={loc} onSelect={onLocalitySelect} />
              </Popup>
            </CircleMarker>
          ))}

        {/* Selected Marker pin */}
        {selectedCoordinates && (
          <Marker position={[selectedCoordinates.latitude, selectedCoordinates.longitude]}>
            <Popup autoPan={true}>
              <div className="p-1">
                <div className="font-semibold text-xs text-slate-800">
                  {selectedCoordinates.name || 'Selected Location'}
                </div>
                <div className="text-[10px] text-slate-500 mt-0.5">
                  Lat: {selectedCoordinates.latitude.toFixed(4)}, Lng: {selectedCoordinates.longitude.toFixed(4)}
                </div>
                {selectedCoordinates.basePriceSqft && (
                  <div className="text-xs font-semibold text-emerald-600 mt-1">
                    ~ ₹{selectedCoordinates.basePriceSqft.toLocaleString('en-IN')}/sqft baseline
                  </div>
                )}
              </div>
            </Popup>
          </Marker>
        )}
      </MapContainer>

      {/* Map Legend */}
      {showLegend && (
        <div className="absolute bottom-4 left-4 z-[1000]">
          <MapLegend />
        </div>
      )}
    </div>
  );
};

export default IndiaMap;
