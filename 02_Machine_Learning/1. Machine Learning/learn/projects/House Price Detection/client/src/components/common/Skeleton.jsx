import React from 'react';

export const Skeleton = ({
  className = '',
  variant = 'rectangular', // 'rectangular' | 'circular' | 'text'
}) => {
  const variants = {
    rectangular: 'rounded-xl',
    circular: 'rounded-full',
    text: 'rounded-md h-4',
  };

  return (
    <div
      className={`animate-pulse bg-slate-200/80 ${variants[variant]} ${className}`}
    />
  );
};

export default Skeleton;
