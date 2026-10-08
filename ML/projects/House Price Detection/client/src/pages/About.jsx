import React from 'react';
import PageContainer from '../components/layout/PageContainer';
import ModelInfo from '../components/model/ModelInfo';
import Card from '../components/common/Card';
import { Info, Code } from 'lucide-react';

export const About = () => {
  return (
    <PageContainer maxWidth="max-w-5xl">
      <div className="space-y-8 pb-12">
        <div>
          <div className="flex items-center gap-2 text-emerald-600 text-xs font-semibold mb-1">
            <Info className="w-4 h-4" />
            <span>Architecture & Specifications</span>
          </div>
          <h1 className="text-3xl font-extrabold text-slate-900 tracking-tight">
            About the Indian House Price Prediction System
          </h1>
          <p className="text-xs sm:text-sm text-slate-500 mt-1 max-w-2xl leading-relaxed">
            Technical overview of the machine learning pipeline, administrative geographic hierarchy, and full-stack API contract.
          </p>
        </div>

        {/* Model Architecture Info */}
        <ModelInfo />

        {/* Full-Stack Architecture Explanation */}
        <Card
          title="Full-Stack System Design"
          subtitle="Decoupled React & FastAPI serving structure"
          icon={Code}
        >
          <div className="space-y-4 text-xs text-slate-600 leading-relaxed">
            <p>
              The system strictly decouples model training, API serving, and client interactions:
            </p>

            <div className="p-4 rounded-2xl bg-slate-900 text-slate-200 font-mono text-xs overflow-x-auto space-y-1">
              <div className="text-emerald-400">// Client-Server Prediction Request Flow:</div>
              <div>React Client (Form / Map)</div>
              <div>&nbsp;&nbsp;&darr; POST /api/v1/predict (Validated Pydantic Payload)</div>
              <div>FastAPI Backend Server</div>
              <div>&nbsp;&nbsp;&darr; Convert payload to 1-row Pandas DataFrame</div>
              <div>&nbsp;&nbsp;&darr; Execute in-memory joblib composite pipeline</div>
              <div>&nbsp;&nbsp;&darr; Target Regressor: HistGradientBoosting &bull; log1p / expm1</div>
              <div>JSON Valuation Response (INR, Lakhs, Crores)</div>
              <div>&nbsp;&nbsp;&darr; Redux Toolkit state slice update</div>
              <div>Prediction Result UI Component</div>
            </div>

            <p>
              The trained model is loaded <strong>once</strong> during server startup lifespan into an in-memory singleton. It is never retrained or reloaded per HTTP request, guaranteeing sub-50ms inference latency.
            </p>
          </div>
        </Card>
      </div>
    </PageContainer>
  );
};

export default About;
