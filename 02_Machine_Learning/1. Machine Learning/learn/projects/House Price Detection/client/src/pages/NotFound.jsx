import React from 'react';
import { Link } from 'react-router-dom';
import PageContainer from '../components/layout/PageContainer';
import Button from '../components/common/Button';
import { Home, Compass } from 'lucide-react';

export const NotFound = () => {
  return (
    <PageContainer>
      <div className="py-24 text-center max-w-md mx-auto space-y-4">
        <div className="w-16 h-16 rounded-3xl bg-emerald-50 text-emerald-600 flex items-center justify-center mx-auto shadow-xs">
          <Compass className="w-8 h-8" />
        </div>
        <h1 className="text-4xl font-extrabold text-slate-900 tracking-tight">404</h1>
        <h2 className="text-lg font-bold text-slate-800">Page Not Found</h2>
        <p className="text-xs text-slate-500 leading-relaxed">
          The requested page does not exist or has been moved. Explore our real estate map or run a price estimation.
        </p>
        <div className="pt-2 flex justify-center gap-3">
          <Link to="/">
            <Button variant="primary" icon={Home}>
              Back to Home
            </Button>
          </Link>
          <Link to="/map">
            <Button variant="outline" icon={Compass}>
              Explore Map
            </Button>
          </Link>
        </div>
      </div>
    </PageContainer>
  );
};

export default NotFound;
