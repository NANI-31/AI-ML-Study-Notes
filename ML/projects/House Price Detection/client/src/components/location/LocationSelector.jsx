import React from 'react';
import { useAppDispatch, useAppSelector } from '../../hooks/reduxHooks';
import {
  setSelectedState,
  setSelectedDistrict,
  setSelectedMandal,
  setSelectedVillage,
} from '../../store/slices/locationSlice';
import { updateMultipleFields } from '../../store/slices/propertySlice';
import { setCenter, setZoom, setSelectedCoordinates } from '../../store/slices/mapSlice';
import locationService from '../../services/locationService';
import Select from '../common/Select';
import { MapPin, Navigation } from 'lucide-react';

export const LocationSelector = ({
  className = '',
  compact = false,
  showTitle = true,
}) => {
  const dispatch = useAppDispatch();
  const {
    states,
    districts,
    mandals,
    villages,
    selectedState,
    selectedDistrict,
    selectedMandal,
    selectedVillage,
  } = useAppSelector((state) => state.location);

  const handleStateChange = (e) => {
    const stateVal = e.target.value;
    const stateObj = states.find((s) => String(s.id) === String(stateVal) || s.name === stateVal) || null;
    dispatch(setSelectedState(stateObj));

    if (stateObj) {
      dispatch(
        updateMultipleFields({
          state: stateObj.name,
          district: '',
          mandal_taluk_tehsil: '',
          village_locality: '',
        })
      );

      // Pan map to state's localities
      const stateLocs = locationService.getLocalitiesByState(stateObj.name);
      if (stateLocs.length > 0 && stateLocs[0].latitude) {
        dispatch(setCenter([stateLocs[0].latitude, stateLocs[0].longitude]));
        dispatch(setZoom(7));
      }
    }
  };

  const handleDistrictChange = (e) => {
    const distVal = e.target.value;
    const distObj = districts.find(
      (d) => String(d.id) === String(distVal) ||
             String(d.district_id) === String(distVal) ||
             d.name === distVal
    ) || null;
    dispatch(setSelectedDistrict(distObj));

    if (distObj) {
      dispatch(
        updateMultipleFields({
          district: distObj.name,
          mandal_taluk_tehsil: '',
          village_locality: '',
          urban_rural: distObj.tier === 'Tier-1' ? 'Urban' : 'Semi-Urban',
        })
      );

      // Pan map to district's localities
      const distLocs = locationService.getLocalitiesByDistrict(distObj.name);
      if (distLocs.length > 0 && distLocs[0].latitude) {
        dispatch(setCenter([distLocs[0].latitude, distLocs[0].longitude]));
        dispatch(setZoom(11));
      }
    }
  };

  const handleMandalChange = (e) => {
    const mandalVal = e.target.value;
    const mandalObj = mandals.find(
      (m) => String(m.id) === String(mandalVal) ||
             String(m.subdistrict_id) === String(mandalVal) ||
             m.name === mandalVal
    ) || null;
    dispatch(setSelectedMandal(mandalObj));

    if (mandalObj) {
      dispatch(
        updateMultipleFields({
          mandal_taluk_tehsil: mandalObj.name,
          village_locality: '',
          urban_rural: mandalObj.urban_rural || 'Semi-Urban',
          distance_to_airport_km: mandalObj.airport_dist_base || 50,
          distance_to_railway_station_km: mandalObj.railway_dist_base || 10,
          distance_to_metro_km: mandalObj.has_metro ? 3.0 : 65.0,
          distance_to_it_hub_km: mandalObj.urban_rural === 'Urban' ? 10.0 : 30.0,
          distance_to_city_center_km: 2.5,
          distance_to_main_road_km: 0.5,
          distance_to_highway_km: 1.5,
        })
      );
    }
  };

  const handleVillageChange = (e) => {
    const villageVal = e.target.value;
    const villageObj = villages.find(
      (v) => String(v.id) === String(villageVal) ||
             String(v.locality_id) === String(villageVal) ||
             v.name === villageVal
    ) || null;
    dispatch(setSelectedVillage(villageObj));

    if (villageObj) {
      dispatch(
        updateMultipleFields({
          village_locality: villageObj.name,
          pincode: villageObj.pincode,
          urban_rural: villageObj.urban_rural || 'Semi-Urban',
          distance_to_airport_km: villageObj.airport_dist_base || 50,
          distance_to_railway_station_km: villageObj.railway_dist_base || 10,
          distance_to_metro_km: villageObj.has_metro ? 3.0 : 70.0,
          distance_to_it_hub_km: villageObj.urban_rural === 'Urban' ? 8.0 : 35.0,
          distance_to_school_km: villageObj.school_dist_base || 1.5,
          distance_to_hospital_km: villageObj.hospital_dist_base || 2.0,
          distance_to_shopping_center_km: villageObj.shopping_dist_base || 1.5,
          distance_to_city_center_km: 1.5,
          distance_to_main_road_km: 0.3,
          distance_to_highway_km: 1.0,
          latitude: villageObj.latitude,
          longitude: villageObj.longitude,
        })
      );

      // Center map on this locality
      if (villageObj.latitude && villageObj.longitude) {
        dispatch(setCenter([villageObj.latitude, villageObj.longitude]));
        dispatch(setZoom(13));
        dispatch(
          setSelectedCoordinates({
            latitude: villageObj.latitude,
            longitude: villageObj.longitude,
            name: villageObj.name,
            district: villageObj.district_name,
            state: villageObj.state_name,
            basePriceSqft: villageObj.base_price_sqft,
          })
        );
      }
    }
  };

  return (
    <div className={`space-y-4 ${className}`}>
      {showTitle && (
        <div className="flex items-center gap-2 pb-2 border-b border-slate-100">
          <MapPin className="w-4 h-4 text-emerald-600" />
          <h4 className="text-sm font-semibold text-slate-800">
            Administrative Hierarchy
          </h4>
        </div>
      )}

      <div className={`grid gap-3.5 ${compact ? 'grid-cols-1' : 'grid-cols-1 sm:grid-cols-2 lg:grid-cols-4'}`}>
        {/* 1. State Selector */}
        <Select
          label="1. State / Union Territory"
          placeholder="Select State..."
          value={selectedState?.id || ''}
          onChange={handleStateChange}
          options={states.map((s) => ({ value: s.id, label: s.name }))}
          required
        />

        {/* 2. District Selector */}
        <Select
          label="2. District"
          placeholder={selectedState ? 'Select District...' : 'Select State first'}
          value={selectedDistrict?.id || ''}
          onChange={handleDistrictChange}
          options={districts.map((d) => ({
            value: d.id,
            label: `${d.name} (${d.tier || 'Dist'})`,
          }))}
          disabled={!selectedState || districts.length === 0}
          required
        />

        {/* 3. Mandal / Tehsil Selector */}
        <Select
          label="3. Mandal / Taluk / Tehsil"
          placeholder={selectedDistrict ? 'Select Mandal...' : 'Select District first'}
          value={selectedMandal?.id || ''}
          onChange={handleMandalChange}
          options={mandals.map((m) => ({
            value: m.id,
            label: `${m.name} (${m.type || 'Subdistrict'})`,
          }))}
          disabled={!selectedDistrict || mandals.length === 0}
        />

        {/* 4. Village / Locality Selector */}
        <Select
          label="4. Village / Locality"
          placeholder={selectedMandal ? 'Select Locality...' : 'Select Mandal first'}
          value={selectedVillage?.id || ''}
          onChange={handleVillageChange}
          options={villages.map((v) => ({
            value: v.id,
            label: `${v.name} (PIN: ${v.pincode})`,
          }))}
          disabled={!selectedMandal || villages.length === 0}
        />
      </div>

      {selectedVillage && (
        <div className="flex items-center gap-2 p-2.5 rounded-xl bg-emerald-50/70 border border-emerald-100 text-xs text-emerald-800">
          <Navigation className="w-3.5 h-3.5 text-emerald-600 shrink-0" />
          <span>
            Selected: <strong>{selectedVillage.name}</strong>, {selectedMandal?.name}, {selectedDistrict?.name}, {selectedState?.name}
          </span>
        </div>
      )}
    </div>
  );
};

export default LocationSelector;
