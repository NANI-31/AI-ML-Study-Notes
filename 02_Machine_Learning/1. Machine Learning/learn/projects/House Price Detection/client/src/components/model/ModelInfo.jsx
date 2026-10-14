import React from 'react';
import Card from '../common/Card';
import { Cpu, Database, AlertTriangle, Layers } from 'lucide-react';

export const ModelInfo = ({ className = '' }) => {
  return (
    <div className={`space-y-6 ${className}`}>
      <Card
        title="Model Pipeline Specifications"
        subtitle="Gradient Boosting valuation architecture & metadata"
        icon={Cpu}
      >
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
          <div className="p-4 rounded-2xl bg-emerald-50/50 border border-emerald-100/60">
            <span className="text-[11px] font-semibold text-emerald-800 uppercase tracking-wider block mb-1">
              Active Estimator
            </span>
            <span className="text-sm font-bold text-slate-900 block">
              HistGradientBoostingRegressor
            </span>
            <span className="text-xs text-slate-500 mt-1 block">
              TransformedTargetRegressor with log1p/expm1 scaling
            </span>
          </div>

          <div className="p-4 rounded-2xl bg-slate-50 border border-slate-100">
            <span className="text-[11px] font-semibold text-slate-500 uppercase tracking-wider block mb-1">
              Features Transformed
            </span>
            <span className="text-sm font-bold text-slate-900 block">
              36 Features (22 Num, 14 Cat)
            </span>
            <span className="text-xs text-slate-500 mt-1 block">
              Scikit-Learn ColumnTransformer Pipeline
            </span>
          </div>

          <div className="p-4 rounded-2xl bg-slate-50 border border-slate-100">
            <span className="text-[11px] font-semibold text-slate-500 uppercase tracking-wider block mb-1">
              Serving Microservice
            </span>
            <span className="text-sm font-bold text-slate-900 block">
              FastAPI v1.0.0
            </span>
            <span className="text-xs text-slate-500 mt-1 block">
              Singleton In-Memory Execution
            </span>
          </div>
        </div>

        <div className="space-y-3 text-xs text-slate-600 leading-relaxed border-t border-slate-100 pt-4">
          <h4 className="font-semibold text-slate-900 text-sm flex items-center gap-2">
            <Layers className="w-4 h-4 text-emerald-600" />
            Inference Architecture
          </h4>
          <p>
            When a property payload is submitted, the FastAPI backend sends the validated 1-row DataFrame directly to the serialized pipeline artifact. Categorical values (state, district, property type, furnishing, quality) are encoded on-the-fly, missing attributes are imputed via median strategy, and continuous numerical features are standardized.
          </p>
          <p>
            Target prediction output is returned in raw INR currency units, which the service layer normalizes into standard Indian denominations (Lakhs and Crores).
          </p>
        </div>
      </Card>

      <Card
        title="Dataset & Modeling Policy"
        subtitle="Transparency & limitations"
        icon={Database}
      >
        <div className="flex items-start gap-3 p-4 rounded-2xl bg-amber-50/80 border border-amber-200/80 text-xs text-amber-900 leading-relaxed">
          <AlertTriangle className="w-5 h-5 text-amber-600 shrink-0 mt-0.5" />
          <div className="space-y-1">
            <h5 className="font-bold text-amber-900">Synthetic Training Dataset Notice</h5>
            <p>
              This machine-learning model is trained on a generated dataset representing realistic property patterns across 36 Indian States/UTs, 76 Districts, and 219 Micro-markets.
            </p>
            <p>
              Predictions produced by this service do not represent certified bank appraisals, government circle rates, municipal assessments, or binding market listings. All valuations must be treated strictly as algorithmic demonstration estimates.
            </p>
          </div>
        </div>
      </Card>
    </div>
  );
};

export default ModelInfo;
