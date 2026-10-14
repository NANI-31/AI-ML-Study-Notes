import React from 'react';

export const LoadingSpinner = ({
  size = 'md', // 'sm' | 'md' | 'lg'
  text = '',
  className = '',
}) => {
  const sizes = {
    sm: 'w-4 h-4 border-2',
    md: 'w-7 h-7 border-3',
    lg: 'w-10 h-10 border-4',
  };

  return (
    <div className={`flex flex-col items-center justify-center p-4 gap-2.5 ${className}`}>
      <div
        className={`${sizes[size]} border-slate-200 border-t-emerald-600 rounded-full animate-spin`}
      />
      {text && <p className="text-xs font-medium text-slate-500 animate-pulse">{text}</p>}
    </div>
  );
};

export default LoadingSpinner;
