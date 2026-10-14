import React from 'react';
import { Routes, Route } from 'react-router-dom';
import Navbar from './components/layout/Navbar';
import Footer from './components/layout/Footer';
import Home from './pages/Home';
import MapExplorer from './pages/MapExplorer';
import PricePrediction from './pages/PricePrediction';
import About from './pages/About';
import NotFound from './pages/NotFound';

export const App = () => {
  return (
    <div className="min-h-screen flex flex-col bg-slate-50/50 text-slate-800 antialiased font-sans selection:bg-emerald-100 selection:text-emerald-900">
      <Navbar />
      <div className="flex-1 flex flex-col">
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/map" element={<MapExplorer />} />
          <Route path="/predict" element={<PricePrediction />} />
          <Route path="/about" element={<About />} />
          <Route path="*" element={<NotFound />} />
        </Routes>
      </div>
      <Footer />
    </div>
  );
};

export default App;
