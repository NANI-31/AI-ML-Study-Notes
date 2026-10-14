import React from 'react';
import { Link } from 'react-router-dom';
import PageContainer from '../components/layout/PageContainer';
import IndiaMap from '../components/map/IndiaMap';
import Card from '../components/common/Card';
import Button from '../components/common/Button';
import {
  Map,
  Sparkles,
  TrendingUp,
  ShieldCheck,
  ArrowRight,
  Layers,
  Compass,
} from 'lucide-react';

export const Home = () => {
  return (
    <div className="space-y-16 pb-12">
      {/* 1. Hero Section */}
      <section className="relative overflow-hidden bg-gradient-to-b from-emerald-50/70 via-slate-50/30 to-white pt-16 pb-20 border-b border-slate-100">
        <PageContainer>
          <div className="text-center max-w-3xl mx-auto space-y-6">
            <div className="inline-flex items-center gap-2 px-3.5 py-1.5 rounded-full bg-emerald-100/70 text-emerald-800 text-xs font-semibold shadow-xs">
              <Sparkles className="w-3.5 h-3.5 text-emerald-600" />
              <span>AI-Powered Indian Real Estate Analytics</span>
            </div>

            <h1 className="text-4xl sm:text-6xl font-extrabold tracking-tight text-slate-900 leading-[1.15]">
              Explore Indian Property Prices with{' '}
              <span className="text-transparent bg-clip-text bg-gradient-to-r from-emerald-600 to-teal-600">
                Machine Learning
              </span>
            </h1>

            <p className="text-base sm:text-lg text-slate-600 max-w-2xl mx-auto leading-relaxed">
              Explore locations and estimate residential property prices across Indian States, Districts, and Micro-markets using geographic and structural real estate indicators.
            </p>

            <div className="flex flex-col sm:flex-row items-center justify-center gap-3 pt-2">
              <Link to="/map">
                <Button size="lg" variant="primary" icon={Map} className="w-full sm:w-auto">
                  Explore Interactive Map
                </Button>
              </Link>
              <Link to="/predict">
                <Button size="lg" variant="outline" icon={Sparkles} className="w-full sm:w-auto">
                  Estimate Property Price
                </Button>
              </Link>
            </div>
          </div>
        </PageContainer>
      </section>

      {/* 2. Interactive Map Preview Section */}
      <PageContainer>
        <div className="space-y-4">
          <div className="flex flex-col sm:flex-row sm:items-end justify-between gap-3">
            <div>
              <span className="text-xs font-semibold text-emerald-600 uppercase tracking-wider block">
                Geographic Intelligence
              </span>
              <h2 className="text-2xl font-bold text-slate-900 mt-1">
                Explore Micro-Markets Across India
              </h2>
            </div>
            <Link
              to="/map"
              className="inline-flex items-center gap-1.5 text-xs font-semibold text-emerald-600 hover:text-emerald-700 transition-colors"
            >
              <span>Open Fullscreen Map</span>
              <ArrowRight className="w-3.5 h-3.5" />
            </Link>
          </div>

          <IndiaMap height="h-[480px]" showLegend={true} />
        </div>
      </PageContainer>

      {/* 3. Value Proposition / Feature Cards */}
      <PageContainer>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <Card
            title="Interactive Map Discovery"
            subtitle="Click any state or district to inspect rates"
            icon={Compass}
            hover
          >
            <p className="text-xs text-slate-600 leading-relaxed mt-2">
              Browse 36 States, 76 Districts, and 219 Micro-markets with localized base rates, metro proximity, and urban/rural classifications.
            </p>
          </Card>

          <Card
            title="Machine Learning Precision"
            subtitle="Gradient Boosting valuation model"
            icon={Layers}
            hover
          >
            <p className="text-xs text-slate-600 leading-relaxed mt-2">
              Evaluate properties against 36 distinct physical, structural, and infrastructure attributes using a Scikit-Learn pipeline.
            </p>
          </Card>

          <Card
            title="Standard Indian Currency"
            subtitle="Normalized valuations in INR, Lakhs & Crores"
            icon={TrendingUp}
            hover
          >
            <p className="text-xs text-slate-600 leading-relaxed mt-2">
              Valuation outputs are automatically converted into Lakhs (₹ 100k) and Crores (₹ 10M) with price per square foot metrics.
            </p>
          </Card>
        </div>
      </PageContainer>

      {/* 4. How It Works Steps */}
      <section className="bg-slate-50 py-16 border-y border-slate-200/80">
        <PageContainer>
          <div className="text-center max-w-xl mx-auto mb-12">
            <span className="text-xs font-semibold text-emerald-600 uppercase tracking-wider block">
              Workflow
            </span>
            <h2 className="text-2xl font-bold text-slate-900 mt-1">
              How Property Valuation Works
            </h2>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            <div className="p-6 rounded-2xl bg-white border border-slate-200 shadow-xs relative">
              <div className="w-9 h-9 rounded-xl bg-emerald-100 text-emerald-700 font-bold flex items-center justify-center mb-4 text-sm">
                1
              </div>
              <h3 className="font-semibold text-slate-900 text-sm mb-1.5">Choose Location</h3>
              <p className="text-xs text-slate-500 leading-relaxed">
                Select your State, District, Mandal, and Locality or tap your desired coordinate on the interactive map.
              </p>
            </div>

            <div className="p-6 rounded-2xl bg-white border border-slate-200 shadow-xs relative">
              <div className="w-9 h-9 rounded-xl bg-emerald-100 text-emerald-700 font-bold flex items-center justify-center mb-4 text-sm">
                2
              </div>
              <h3 className="font-semibold text-slate-900 text-sm mb-1.5">Enter Property Details</h3>
              <p className="text-xs text-slate-500 leading-relaxed">
                Provide built-up area, BHK layout, floor number, furnishing quality, and proximity to transportation hubs.
              </p>
            </div>

            <div className="p-6 rounded-2xl bg-white border border-slate-200 shadow-xs relative">
              <div className="w-9 h-9 rounded-xl bg-emerald-100 text-emerald-700 font-bold flex items-center justify-center mb-4 text-sm">
                3
              </div>
              <h3 className="font-semibold text-slate-900 text-sm mb-1.5">Get Estimated Price</h3>
              <p className="text-xs text-slate-500 leading-relaxed">
                The FastAPI microservice computes instant inference using the trained Scikit-Learn gradient boosting model.
              </p>
            </div>
          </div>
        </PageContainer>
      </section>

      {/* 5. Model Transparency & Synthetic Data Disclaimer */}
      <PageContainer>
        <div className="p-6 rounded-3xl bg-slate-900 text-white flex flex-col md:flex-row items-start md:items-center justify-between gap-6 shadow-lg">
          <div className="space-y-2 max-w-2xl">
            <div className="flex items-center gap-2 text-emerald-400 text-xs font-semibold">
              <ShieldCheck className="w-4 h-4" />
              <span>Transparent AI Disclosure</span>
            </div>
            <h3 className="text-lg font-bold text-white">
              Analytical Model Trained on Synthetic Datasets
            </h3>
            <p className="text-xs text-slate-400 leading-relaxed">
              Price estimates are generated by a machine-learning model and should not be considered official property valuations. Because this model was trained on synthetic data for demonstration, values illustrate algorithmic capabilities rather than real-world appraisals.
            </p>
          </div>

          <Link to="/about">
            <Button variant="outline" size="sm" className="whitespace-nowrap bg-white/10 text-white border-white/20 hover:bg-white/20">
              Read Documentation
            </Button>
          </Link>
        </div>
      </PageContainer>
    </div>
  );
};

export default Home;
