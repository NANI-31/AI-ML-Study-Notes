import React, { useRef } from 'react';
import PageContainer from '../components/layout/PageContainer';
import PropertyForm from '../components/property/PropertyForm';
import PropertySummary from '../components/property/PropertySummary';
import PredictionResult from '../components/prediction/PredictionResult';
import IndiaMap from '../components/map/IndiaMap';
import ErrorMessage from '../components/common/ErrorMessage';
import Card from '../components/common/Card';
import { useAppSelector, useAppDispatch } from '../hooks/reduxHooks';
import { clearPrediction } from '../store/slices/predictionSlice';
import { Calculator, Sparkles, MapPin } from 'lucide-react';

export const PricePrediction = () => {
  const dispatch = useAppDispatch();
  const { result, submittedProperty, error } = useAppSelector(
    (state) => state.prediction
  );
  const resultRef = useRef(null);

  const handlePredictionSuccess = () => {
    if (resultRef.current) {
      resultRef.current.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
  };

  return (
    <PageContainer maxWidth="max-w-7xl">
      <div className="space-y-8 pb-12">
        {/* Header */}
        <div>
          <div className="flex items-center gap-2 text-emerald-600 text-xs font-semibold mb-1">
            <Calculator className="w-4 h-4" />
            <span>FastAPI ML Inference</span>
          </div>
          <h1 className="text-3xl font-extrabold text-slate-900 tracking-tight">
            Indian House Price Prediction
          </h1>
          <p className="text-xs sm:text-sm text-slate-500 mt-1 max-w-2xl leading-relaxed">
            Configure residential property specifications and obtain instant, calibrated market valuations computed by the Scikit-Learn Gradient Boosting model.
          </p>
        </div>

        {/* Global Error Banner */}
        {error && (
          <ErrorMessage
            title="Prediction Request Error"
            message={error}
            onRetry={() => dispatch(clearPrediction())}
          />
        )}

        {/* Main Grid: Form on Left (8 cols), Results & Map on Right (4 cols) */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
          {/* Left Column: Form */}
          <div className="lg:col-span-7 xl:col-span-7 space-y-6">
            <PropertyForm onPredictionSuccess={handlePredictionSuccess} />
          </div>

          {/* Right Column: Prediction Output & Map Context */}
          <div className="lg:col-span-5 xl:col-span-5 space-y-6 sticky top-24" ref={resultRef}>
            {/* If Prediction exists */}
            {result ? (
              <div className="space-y-6">
                <PredictionResult
                  result={result}
                  submittedProperty={submittedProperty}
                />
                <PropertySummary property={submittedProperty} />
              </div>
            ) : (
              /* Empty / Placeholder Result State */
              <div className="p-8 rounded-3xl border-2 border-dashed border-slate-200 bg-slate-50/50 text-center space-y-3">
                <div className="w-12 h-12 rounded-2xl bg-white shadow-xs border border-slate-200/80 flex items-center justify-center mx-auto text-emerald-600">
                  <Sparkles className="w-6 h-6" />
                </div>
                <h3 className="font-semibold text-slate-800 text-base">
                  Awaiting Property Specifications
                </h3>
                <p className="text-xs text-slate-500 max-w-xs mx-auto leading-relaxed">
                  Fill in the property details on the left and tap &ldquo;Estimate Property Price&rdquo; to execute inference.
                </p>
              </div>
            )}

            {/* Geographic Context Map Preview */}
            <Card
              title="Geographic Context"
              subtitle="Selected location preview"
              icon={MapPin}
              padding="p-4"
            >
              <IndiaMap height="h-[280px]" showLegend={false} />
            </Card>
          </div>
        </div>
      </div>
    </PageContainer>
  );
};

export default PricePrediction;
