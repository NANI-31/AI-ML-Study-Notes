import React from 'react';
import { HelpCircle } from 'lucide-react';

export const EmptyState = ({
  icon: Icon = HelpCircle,
  title = 'No records found',
  description = 'Try selecting another option or adjusting your filters.',
  action,
  className = '',
}) => {
  return (
    <div
      className={`flex flex-col items-center justify-center p-8 text-center rounded-2xl border-2 border-dashed border-slate-200 bg-slate-50/50 ${className}`}
    >
      <div className="p-3 rounded-2xl bg-white shadow-xs border border-slate-100 text-slate-400 mb-3">
        <Icon className="w-6 h-6" />
      </div>
      <h4 className="text-sm font-semibold text-slate-800">{title}</h4>
      <p className="text-xs text-slate-500 max-w-sm mt-1 mb-4 leading-relaxed">
        {description}
      </p>
      {action && <div>{action}</div>}
    </div>
  );
};

export default EmptyState;
