import React from 'react';
import { formatINR } from '../../utils/formatCurrency';
import { MapPin, ArrowRight } from 'lucide-react';
import { useNavigate } from 'react-router-dom';

export const PricePopup = ({ locality, onSelect }) => {
  const navigate = useNavigate();

  if (!locality) return null;

  const handlePredictClick = () => {
    if (onSelect) onSelect(locality);
    navigate('/predict');
  };

  return (
    <div className="p-1 min-w-[220px]">
      <div className="flex items-center gap-1.5 text-emerald-700 font-semibold text-sm mb-1">
        <MapPin className="w-4 h-4 shrink-0" />
        <span>{locality.name}</span>
      </div>

      <div className="text-xs text-slate-500 mb-2">
        {locality.district_name || locality.district}, {locality.state_name || locality.state}
        {locality.pincode && ` - ${locality.pincode}`}
      </div>

      <div className="p-2 rounded-xl bg-slate-50 border border-slate-100 mb-3 space-y-1">
        <div className="flex justify-between text-xs">
          <span className="text-slate-400">Baseline Rate:</span>
          <span className="font-semibold text-slate-800">
            {locality.base_price_sqft
              ? `${formatINR(locality.base_price_sqft)} / sq ft`
              : 'N/A'}
          </span>
        </div>
        <div className="flex justify-between text-xs">
          <span className="text-slate-400">Classification:</span>
          <span className="font-medium text-slate-700 capitalize">
            {locality.urban_rural || 'Urban'}
          </span>
        </div>
      </div>

      <button
        onClick={handlePredictClick}
        className="w-full flex items-center justify-center gap-1.5 py-1.5 px-3 rounded-lg bg-emerald-600 hover:bg-emerald-700 text-white text-xs font-semibold shadow-xs transition-colors cursor-pointer"
      >
        <span>Estimate Price Here</span>
        <ArrowRight className="w-3.5 h-3.5" />
      </button>
    </div>
  );
};

export default PricePopup;
