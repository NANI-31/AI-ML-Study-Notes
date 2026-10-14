import React from 'react';
import Card from '../common/Card';
import { Home } from 'lucide-react';

export const PropertySummary = ({ property, className = '' }) => {
  if (!property) return null;

  return (
    <Card
      title="Property Specifications Summary"
      subtitle="Parameters evaluated by the Scikit-Learn pipeline"
      icon={Home}
      className={className}
    >
      <div className="grid grid-cols-2 sm:grid-cols-3 gap-4 text-xs">
        <div className="space-y-1">
          <span className="text-slate-400 block">Location</span>
          <span className="font-semibold text-slate-800 block truncate">
            {property.village_locality ? `${property.village_locality}, ` : ''}
            {property.district}, {property.state}
          </span>
        </div>

        <div className="space-y-1">
          <span className="text-slate-400 block">Property Type</span>
          <span className="font-semibold text-slate-800 block">
            {property.property_type || 'Apartment'}
          </span>
        </div>

        <div className="space-y-1">
          <span className="text-slate-400 block">Built-up Area</span>
          <span className="font-semibold text-slate-800 block">
            {property.built_up_area_sqft ? `${Number(property.built_up_area_sqft).toLocaleString('en-IN')} sq ft` : 'N/A'}
          </span>
        </div>

        <div className="space-y-1">
          <span className="text-slate-400 block">Layout</span>
          <span className="font-semibold text-slate-800 block">
            {property.bedrooms || 0} Beds &bull; {property.bathrooms || 0} Baths &bull; {property.balconies || 0} Balcony
          </span>
        </div>

        <div className="space-y-1">
          <span className="text-slate-400 block">Floor Level</span>
          <span className="font-semibold text-slate-800 block">
            Floor {property.floor_number || 0} of {property.total_floors || 1}
          </span>
        </div>

        <div className="space-y-1">
          <span className="text-slate-400 block">Property Age</span>
          <span className="font-semibold text-slate-800 block">
            {property.property_age_years == 0 ? 'Brand New / Under Construction' : `${property.property_age_years} years`}
          </span>
        </div>

        <div className="space-y-1">
          <span className="text-slate-400 block">Furnishing & Quality</span>
          <span className="font-semibold text-slate-800 block">
            {property.furnishing_status} &bull; {property.construction_quality}
          </span>
        </div>

        <div className="space-y-1">
          <span className="text-slate-400 block">Facing & Road</span>
          <span className="font-semibold text-slate-800 block">
            {property.facing_direction} Facing &bull; {property.road_width_ft} ft road
          </span>
        </div>

        <div className="space-y-1">
          <span className="text-slate-400 block">Gated Community</span>
          <span className="font-semibold text-slate-800 block">
            {property.gated_community ? 'Yes (Gated)' : 'No (Standalone)'}
          </span>
        </div>
      </div>
    </Card>
  );
};

export default PropertySummary;
