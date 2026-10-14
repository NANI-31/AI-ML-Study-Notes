import React from 'react';

export const MapLegend = ({ className = '' }) => {
  return (
    <div
      className={`p-3 rounded-2xl bg-white/95 backdrop-blur-md border border-slate-200/80 shadow-md text-xs ${className}`}
    >
      <div className="font-semibold text-slate-800 mb-2">Locality Base Rates</div>
      <div className="flex items-center gap-3">
        <div className="flex items-center gap-1.5">
          <span className="w-2.5 h-2.5 rounded-full bg-emerald-500 inline-block" />
          <span className="text-slate-600">&lt; ₹5,000 / sqft</span>
        </div>
        <div className="flex items-center gap-1.5">
          <span className="w-2.5 h-2.5 rounded-full bg-amber-500 inline-block" />
          <span className="text-slate-600">₹5k – ₹10k</span>
        </div>
        <div className="flex items-center gap-1.5">
          <span className="w-2.5 h-2.5 rounded-full bg-rose-500 inline-block" />
          <span className="text-slate-600">&gt; ₹10,000 / sqft</span>
        </div>
      </div>
    </div>
  );
};

export default MapLegend;
