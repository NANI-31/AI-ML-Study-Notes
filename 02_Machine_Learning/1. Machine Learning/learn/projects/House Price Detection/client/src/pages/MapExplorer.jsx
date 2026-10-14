import React from 'react';
import PageContainer from '../components/layout/PageContainer';
import IndiaMap from '../components/map/IndiaMap';
import LocationSelector from '../components/location/LocationSelector';
import Card from '../components/common/Card';
import Button from '../components/common/Button';
import { useAppSelector } from '../hooks/reduxHooks';
import { formatINR } from '../utils/formatCurrency';
import { useNavigate } from 'react-router-dom';
import { Map, MapPin, Sparkles, Navigation, Compass } from 'lucide-react';

export const MapExplorer = () => {
  const navigate = useNavigate();
  const { selectedCoordinates } = useAppSelector((state) => state.map);
  const { selectedState, selectedDistrict, selectedVillage } =
    useAppSelector((state) => state.location);

  const handlePredictForLocation = () => {
    navigate('/predict');
  };

  return (
    <PageContainer maxWidth="max-w-[1500px]" padding="px-4 py-6">
      <div className="flex flex-col lg:flex-row gap-6 items-start">
        {/* Left Sidebar: Controls & Selected Location Stats */}
        <div className="w-full lg:w-[420px] shrink-0 space-y-5">
          <div>
            <div className="flex items-center gap-2 text-emerald-600 text-xs font-semibold mb-1">
              <Map className="w-4 h-4" />
              <span>Geographic Explorer</span>
            </div>
            <h1 className="text-2xl font-bold text-slate-900 tracking-tight">
              Interactive Map of India
            </h1>
            <p className="text-xs text-slate-500 mt-1 leading-relaxed">
              Click on any pin or point on the map to automatically inspect localized rates and load administrative hierarchies.
            </p>
          </div>

          {/* Administrative Hierarchy Dropdowns */}
          <Card title="Quick Location Filter" icon={Compass} padding="p-5">
            <LocationSelector compact={true} showTitle={false} />
          </Card>

          {/* Active Selection Details Card */}
          <Card
            title={selectedVillage?.name || selectedCoordinates?.name || 'Selected Point'}
            subtitle={
              selectedDistrict?.name
                ? `${selectedDistrict.name}, ${selectedState?.name || 'India'}`
                : 'Tap any locality on the map'
            }
            icon={MapPin}
            padding="p-5"
          >
            {selectedVillage || selectedCoordinates ? (
              <div className="space-y-4">
                <div className="grid grid-cols-2 gap-3 text-xs">
                  <div className="p-3 rounded-xl bg-slate-50 border border-slate-100">
                    <span className="text-slate-400 block mb-0.5">Base Rate</span>
                    <span className="font-bold text-slate-900 text-sm">
                      {selectedVillage?.base_price_sqft || selectedCoordinates?.basePriceSqft
                        ? `${formatINR(selectedVillage?.base_price_sqft || selectedCoordinates?.basePriceSqft)} / sq ft`
                        : 'Varies by property'}
                    </span>
                  </div>

                  <div className="p-3 rounded-xl bg-slate-50 border border-slate-100">
                    <span className="text-slate-400 block mb-0.5">Classification</span>
                    <span className="font-semibold text-slate-800 text-sm capitalize">
                      {selectedVillage?.urban_rural || 'Urban'}
                    </span>
                  </div>

                  <div className="p-3 rounded-xl bg-slate-50 border border-slate-100">
                    <span className="text-slate-400 block mb-0.5">PIN Code</span>
                    <span className="font-semibold text-slate-800">
                      {selectedVillage?.pincode || 'Available in dataset'}
                    </span>
                  </div>

                  <div className="p-3 rounded-xl bg-slate-50 border border-slate-100">
                    <span className="text-slate-400 block mb-0.5">Metro Transit</span>
                    <span className="font-semibold text-slate-800">
                      {selectedVillage?.has_metro ? 'Yes (Metro Station)' : 'Feeder / Road'}
                    </span>
                  </div>
                </div>

                <Button
                  onClick={handlePredictForLocation}
                  variant="primary"
                  icon={Sparkles}
                  className="w-full py-2.5"
                >
                  Estimate Property Price Here
                </Button>
              </div>
            ) : (
              <div className="py-6 text-center text-xs text-slate-400">
                <Navigation className="w-8 h-8 mx-auto mb-2 text-slate-300" />
                Click anywhere on the map or use the filters above to inspect real estate data.
              </div>
            )}
          </Card>
        </div>

        {/* Right Section: Large Interactive Leaflet Map */}
        <div className="flex-1 w-full min-h-[550px] lg:min-h-[750px] flex flex-col">
          <IndiaMap height="h-[600px] lg:h-[750px]" showLegend={true} />
        </div>
      </div>
    </PageContainer>
  );
};

export default MapExplorer;
