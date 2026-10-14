import React from 'react';

export const Card = ({
  children,
  className = '',
  title,
  subtitle,
  icon: Icon,
  action,
  padding = 'p-6',
  hover = false,
  ...props
}) => {
  return (
    <div
      className={`bg-white rounded-2xl border border-slate-200/80 shadow-xs ${
        hover ? 'hover:shadow-md transition-shadow duration-200' : ''
      } ${padding} ${className}`}
      {...props}
    >
      {(title || subtitle || Icon || action) && (
        <div className="flex items-center justify-between pb-4 mb-4 border-b border-slate-100">
          <div className="flex items-center gap-3">
            {Icon && (
              <div className="p-2 rounded-xl bg-emerald-50 text-emerald-600 border border-emerald-100/50">
                <Icon className="w-5 h-5" />
              </div>
            )}
            <div>
              {title && <h3 className="font-semibold text-slate-900 text-base">{title}</h3>}
              {subtitle && <p className="text-xs text-slate-500 mt-0.5">{subtitle}</p>}
            </div>
          </div>
          {action && <div>{action}</div>}
        </div>
      )}
      {children}
    </div>
  );
};

export default Card;
