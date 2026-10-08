import React from 'react';
import { useMap } from 'react-leaflet';
import { Compass } from 'lucide-react';

export const MapControls = ({ onResetView }) => {
  const map = useMap();

  const handleReset = () => {
    map.setView([20.5937, 78.9629], 5);
    if (onResetView) onResetView();
  };

  return (
    <div className="absolute top-4 right-4 z-[1000] flex flex-col gap-2">
      <button
        onClick={handleReset}
        title="Reset map view to India"
        className="p-2.5 rounded-xl bg-white/95 hover:bg-white text-slate-700 shadow-md border border-slate-200/80 hover:text-emerald-600 transition-all cursor-pointer backdrop-blur-xs"
      >
        <Compass className="w-5 h-5" />
      </button>
    </div>
  );
};

export default MapControls;
