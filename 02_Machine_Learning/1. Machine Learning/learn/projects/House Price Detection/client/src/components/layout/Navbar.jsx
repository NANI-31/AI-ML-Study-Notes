import React, { useState } from 'react';
import { NavLink, Link } from 'react-router-dom';
import { Home, Map, Calculator, Info, Menu, X, Sparkles } from 'lucide-react';

export const Navbar = () => {
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  const navLinks = [
    { name: 'Home', path: '/', icon: Home },
    { name: 'Explore Map', path: '/map', icon: Map },
    { name: 'Predict Price', path: '/predict', icon: Calculator },
    { name: 'About Model', path: '/about', icon: Info },
  ];

  const linkClass = ({ isActive }) =>
    `inline-flex items-center gap-2 px-3.5 py-2 rounded-xl text-sm font-medium transition-all duration-150 ${
      isActive
        ? 'bg-emerald-50 text-emerald-700 font-semibold shadow-xs'
        : 'text-slate-600 hover:text-slate-900 hover:bg-slate-50'
    }`;

  const mobileLinkClass = ({ isActive }) =>
    `flex items-center gap-3 px-4 py-3 rounded-xl text-sm font-medium transition-colors ${
      isActive
        ? 'bg-emerald-50 text-emerald-700 font-semibold'
        : 'text-slate-700 hover:bg-slate-50'
    }`;

  return (
    <header className="sticky top-0 z-40 bg-white/90 backdrop-blur-md border-b border-slate-200/80">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          {/* Brand Logo */}
          <Link to="/" className="flex items-center gap-2.5 group">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-emerald-600 to-teal-500 flex items-center justify-center text-white shadow-xs group-hover:shadow-md transition-shadow">
              <Home className="w-5 h-5" />
            </div>
            <div>
              <span className="text-base font-bold tracking-tight text-slate-900 block leading-tight">
                Bharat Property
              </span>
              <span className="text-[10px] font-semibold uppercase tracking-wider text-emerald-600 block">
                Price Intelligence
              </span>
            </div>
          </Link>

          {/* Desktop Navigation */}
          <nav className="hidden md:flex items-center gap-1.5">
            {navLinks.map((link) => {
              const Icon = link.icon;
              return (
                <NavLink key={link.path} to={link.path} className={linkClass}>
                  <Icon className="w-4 h-4" />
                  {link.name}
                </NavLink>
              );
            })}
          </nav>

          {/* Action CTA & Backend Status */}
          <div className="hidden md:flex items-center gap-3">
            <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-emerald-50/80 border border-emerald-200/60 text-emerald-700 text-xs font-medium">
              <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
              ML v1.0 Ready
            </div>

            <Link
              to="/predict"
              className="inline-flex items-center gap-2 px-4 py-2 rounded-xl text-xs font-semibold text-white bg-emerald-600 hover:bg-emerald-700 shadow-xs hover:shadow transition-all"
            >
              <Sparkles className="w-3.5 h-3.5" />
              Estimate Price
            </Link>
          </div>

          {/* Mobile Menu Toggle Button */}
          <div className="flex items-center md:hidden">
            <button
              onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
              className="p-2 rounded-xl text-slate-600 hover:text-slate-900 hover:bg-slate-100 focus:outline-none"
              aria-label="Toggle Navigation"
            >
              {mobileMenuOpen ? <X className="w-6 h-6" /> : <Menu className="w-6 h-6" />}
            </button>
          </div>
        </div>
      </div>

      {/* Mobile Menu Dropdown */}
      {mobileMenuOpen && (
        <div className="md:hidden border-t border-slate-100 bg-white px-4 pt-3 pb-5 space-y-1.5 shadow-lg animate-in slide-in-from-top-2 duration-150">
          {navLinks.map((link) => {
            const Icon = link.icon;
            return (
              <NavLink
                key={link.path}
                to={link.path}
                onClick={() => setMobileMenuOpen(false)}
                className={mobileLinkClass}
              >
                <Icon className="w-5 h-5 text-emerald-600" />
                {link.name}
              </NavLink>
            );
          })}
          <div className="pt-3 border-t border-slate-100">
            <Link
              to="/predict"
              onClick={() => setMobileMenuOpen(false)}
              className="w-full flex items-center justify-center gap-2 px-4 py-2.5 rounded-xl text-sm font-semibold text-white bg-emerald-600"
            >
              <Sparkles className="w-4 h-4" />
              Estimate Property Price
            </Link>
          </div>
        </div>
      )}
    </header>
  );
};

export default Navbar;
