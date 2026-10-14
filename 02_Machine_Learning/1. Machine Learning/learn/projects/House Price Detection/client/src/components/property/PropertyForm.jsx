import React, { useState } from 'react';
import { useAppDispatch, useAppSelector } from '../../hooks/reduxHooks';
import { updateFormField, updateMultipleFields, resetPropertyForm } from '../../store/slices/propertySlice';
import { predictPrice } from '../../store/slices/predictionSlice';
import { validatePropertyForm } from '../../utils/validation';
import propertyService from '../../services/propertyService';
import LocationSelector from '../location/LocationSelector';
import Input from '../common/Input';
import Select from '../common/Select';
import Button from '../common/Button';
import Card from '../common/Card';
import {
  Sparkles,
  RotateCcw,
  Building2,
  MapPin,
} from 'lucide-react';

export const PropertyForm = ({ onPredictionSuccess, className = '' }) => {
  const dispatch = useAppDispatch();
  const formData = useAppSelector((state) => state.property.formData);
  const loading = useAppSelector((state) => state.prediction.loading);

  const [errors, setErrors] = useState({});
  const [activeTab, setActiveTab] = useState('dimensions'); // 'dimensions' | 'amenities' | 'proximities'

  const handleFieldChange = (e) => {
    const { name, value, type, checked } = e.target;
    const finalVal = type === 'checkbox' ? (checked ? 1 : 0) : value;
    dispatch(updateFormField({ field: name, value: finalVal }));

    // Smart defaults when property category changes
    if (name === 'property_type') {
      if (value === 'Independent House' || value === 'Villa' || value === 'Residential Plot') {
        const updates = {};
        if (Number(formData.floor_number) > 1) {
          updates.floor_number = 0;
        }
        if (Number(formData.total_floors) > 2) {
          updates.total_floors = value === 'Residential Plot' ? 1 : 2;
        }
        if (Object.keys(updates).length > 0) {
          dispatch(updateMultipleFields(updates));
        }
      } else if (value === 'Apartment') {
        if (Number(formData.total_floors) <= 2) {
          dispatch(updateMultipleFields({ total_floors: 5, floor_number: 1 }));
        }
      }
    }

    // Clear error for edited field
    if (errors[name]) {
      setErrors((prev) => {
        const next = { ...prev };
        delete next[name];
        return next;
      });
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();

    const { isValid, errors: validationErrors } = validatePropertyForm(formData);
    if (!isValid) {
      setErrors(validationErrors);
      // Switch to tab with first error
      if (
        validationErrors.built_up_area_sqft ||
        validationErrors.carpet_area_sqft ||
        validationErrors.bedrooms ||
        validationErrors.bathrooms ||
        validationErrors.floor_number
      ) {
        setActiveTab('dimensions');
      } else if (validationErrors.distance_to_city_center_km) {
        setActiveTab('proximities');
      }
      return;
    }

    setErrors({});
    const resultAction = await dispatch(predictPrice(formData));
    if (predictPrice.fulfilled.match(resultAction) && onPredictionSuccess) {
      onPredictionSuccess(resultAction.payload);
    }
  };

  const handleReset = () => {
    dispatch(resetPropertyForm());
    setErrors({});
  };

  return (
    <form onSubmit={handleSubmit} className={`space-y-6 ${className}`}>
      {/* 1. Location Selection Block */}
      <Card
        title="1. Location Information"
        subtitle="Select micro-market in India or click anywhere on the map"
        icon={MapPin}
      >
        <LocationSelector compact={false} showTitle={false} />
      </Card>

      {/* 2. Property Specifications Tabs */}
      <Card
        title="2. Property Attributes"
        subtitle="Configure physical dimensions and structural details"
        icon={Building2}
      >
        {/* Navigation Tabs */}
        <div className="flex border-b border-slate-200 mb-6 gap-2">
          <button
            type="button"
            onClick={() => setActiveTab('dimensions')}
            className={`pb-3 px-3 text-xs font-semibold border-b-2 transition-colors cursor-pointer ${
              activeTab === 'dimensions'
                ? 'border-emerald-600 text-emerald-700'
                : 'border-transparent text-slate-500 hover:text-slate-800'
            }`}
          >
            Dimensions & Layout
          </button>
          <button
            type="button"
            onClick={() => setActiveTab('amenities')}
            className={`pb-3 px-3 text-xs font-semibold border-b-2 transition-colors cursor-pointer ${
              activeTab === 'amenities'
                ? 'border-emerald-600 text-emerald-700'
                : 'border-transparent text-slate-500 hover:text-slate-800'
            }`}
          >
            Amenities & Quality
          </button>
          <button
            type="button"
            onClick={() => setActiveTab('proximities')}
            className={`pb-3 px-3 text-xs font-semibold border-b-2 transition-colors cursor-pointer ${
              activeTab === 'proximities'
                ? 'border-emerald-600 text-emerald-700'
                : 'border-transparent text-slate-500 hover:text-slate-800'
            }`}
          >
            Proximity & Transit
          </button>
        </div>

        {/* Tab 1: Dimensions & Layout */}
        {activeTab === 'dimensions' && (
          <div className="space-y-4 animate-in fade-in duration-150">
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
              <Select
                label="Property Category"
                name="property_type"
                value={formData.property_type}
                onChange={handleFieldChange}
                options={propertyService.propertyTypes}
                error={errors.property_type}
                required
              />

              <Input
                label="Built-Up Area"
                name="built_up_area_sqft"
                type="number"
                unit="sq ft"
                value={formData.built_up_area_sqft}
                onChange={handleFieldChange}
                error={errors.built_up_area_sqft}
                placeholder="e.g. 1250"
                min="100"
                max="50000"
                required
              />

              <Input
                label="Carpet Area"
                name="carpet_area_sqft"
                type="number"
                unit="sq ft"
                value={formData.carpet_area_sqft}
                onChange={handleFieldChange}
                error={errors.carpet_area_sqft}
                placeholder="Defaults to ~80% if empty"
                min="50"
              />

              <Input
                label="Bedrooms (BHK)"
                name="bedrooms"
                type="number"
                value={formData.bedrooms}
                onChange={handleFieldChange}
                error={errors.bedrooms}
                min="0"
                max="10"
                required
              />

              <Input
                label="Bathrooms"
                name="bathrooms"
                type="number"
                value={formData.bathrooms}
                onChange={handleFieldChange}
                error={errors.bathrooms}
                min="1"
                max="10"
                required
              />

              <Input
                label="Attached Balconies"
                name="balconies"
                type="number"
                value={formData.balconies}
                onChange={handleFieldChange}
                min="0"
                max="6"
              />

              <Input
                label="Floor Level"
                name="floor_number"
                type="number"
                value={formData.floor_number}
                onChange={handleFieldChange}
                error={errors.floor_number}
                helperText="0 = Ground floor"
                min="0"
                max="100"
              />

              <Input
                label="Total Building Floors"
                name="total_floors"
                type="number"
                value={formData.total_floors}
                onChange={handleFieldChange}
                error={errors.total_floors}
                min="1"
                max="100"
              />

              <Select
                label="Listing Status"
                name="property_status"
                value={formData.property_status}
                onChange={handleFieldChange}
                options={propertyService.propertyStatuses}
              />
            </div>
          </div>
        )}

        {/* Tab 2: Amenities & Quality */}
        {activeTab === 'amenities' && (
          <div className="space-y-4 animate-in fade-in duration-150">
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
              <Select
                label="Furnishing Status"
                name="furnishing_status"
                value={formData.furnishing_status}
                onChange={handleFieldChange}
                options={propertyService.furnishingStatuses}
              />

              <Select
                label="Construction Quality"
                name="construction_quality"
                value={formData.construction_quality}
                onChange={handleFieldChange}
                options={propertyService.constructionQualities}
              />

              <Select
                label="Property Condition"
                name="property_condition"
                value={formData.property_condition}
                onChange={handleFieldChange}
                options={propertyService.propertyConditions}
              />

              <Input
                label="Property Age"
                name="property_age_years"
                type="number"
                unit="years"
                value={formData.property_age_years}
                onChange={handleFieldChange}
                error={errors.property_age_years}
                helperText="0 = New or Under Construction"
                min="0"
              />

              <Input
                label="Parking Spaces"
                name="parking_spaces"
                type="number"
                value={formData.parking_spaces}
                onChange={handleFieldChange}
                min="0"
                max="10"
              />

              <Select
                label="Facing Direction"
                name="facing_direction"
                value={formData.facing_direction}
                onChange={handleFieldChange}
                options={propertyService.facingDirections}
              />

              <Input
                label="Road Width"
                name="road_width_ft"
                type="number"
                unit="ft"
                value={formData.road_width_ft}
                onChange={handleFieldChange}
                min="10"
              />

              <div className="sm:col-span-2 flex items-center gap-3 p-3.5 rounded-xl bg-slate-50 border border-slate-200 mt-2">
                <input
                  type="checkbox"
                  id="gated_community"
                  name="gated_community"
                  checked={Boolean(formData.gated_community)}
                  onChange={handleFieldChange}
                  className="w-4 h-4 text-emerald-600 rounded-md focus:ring-emerald-500 border-slate-300 cursor-pointer"
                />
                <label
                  htmlFor="gated_community"
                  className="text-xs font-semibold text-slate-800 cursor-pointer"
                >
                  Property is in a Gated Community / Enclave (Security, Clubhouse, Park)
                </label>
              </div>
            </div>
          </div>
        )}

        {/* Tab 3: Proximity & Transit */}
        {activeTab === 'proximities' && (
          <div className="space-y-4 animate-in fade-in duration-150">
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
              <Input
                label="Distance to City Center"
                name="distance_to_city_center_km"
                type="number"
                step="0.5"
                unit="km"
                value={formData.distance_to_city_center_km}
                onChange={handleFieldChange}
                error={errors.distance_to_city_center_km}
                required
              />

              <Input
                label="Distance to Main Road"
                name="distance_to_main_road_km"
                type="number"
                step="0.1"
                unit="km"
                value={formData.distance_to_main_road_km}
                onChange={handleFieldChange}
              />

              <Input
                label="Distance to Highway"
                name="distance_to_highway_km"
                type="number"
                step="0.5"
                unit="km"
                value={formData.distance_to_highway_km}
                onChange={handleFieldChange}
              />

              <Input
                label="Distance to Metro Station"
                name="distance_to_metro_km"
                type="number"
                step="0.5"
                unit="km"
                value={formData.distance_to_metro_km}
                onChange={handleFieldChange}
              />

              <Input
                label="Distance to Railway Station"
                name="distance_to_railway_station_km"
                type="number"
                step="0.5"
                unit="km"
                value={formData.distance_to_railway_station_km}
                onChange={handleFieldChange}
              />

              <Input
                label="Distance to Airport"
                name="distance_to_airport_km"
                type="number"
                step="1"
                unit="km"
                value={formData.distance_to_airport_km}
                onChange={handleFieldChange}
              />

              <Select
                label="Nearby Public Transport"
                name="nearby_public_transport"
                value={formData.nearby_public_transport}
                onChange={handleFieldChange}
                options={propertyService.nearbyTransportRatings}
              />

              <Select
                label="Neighborhood Character"
                name="neighborhood_type"
                value={formData.neighborhood_type}
                onChange={handleFieldChange}
                options={propertyService.neighborhoodTypes}
              />

              <Select
                label="Urban / Rural Setting"
                name="urban_rural"
                value={formData.urban_rural}
                onChange={handleFieldChange}
                options={propertyService.urbanRuralTiers}
              />
            </div>
          </div>
        )}
      </Card>

      {/* 3. Action Submission Bar */}
      <div className="flex flex-col sm:flex-row items-center justify-between gap-4 p-4 rounded-2xl bg-white border border-slate-200/80 shadow-xs">
        <div className="text-xs text-slate-500">
          Sending payload directly to the Scikit-Learn FastAPI inference endpoint.
        </div>

        <div className="flex items-center gap-3 w-full sm:w-auto">
          <Button
            type="button"
            variant="outline"
            onClick={handleReset}
            icon={RotateCcw}
            disabled={loading}
          >
            Reset
          </Button>

          <Button
            type="submit"
            variant="primary"
            icon={Sparkles}
            loading={loading}
            className="w-full sm:w-auto px-6 py-2.5"
          >
            {loading ? 'Estimating Property Price...' : 'Estimate Property Price'}
          </Button>
        </div>
      </div>
    </form>
  );
};

export default PropertyForm;
