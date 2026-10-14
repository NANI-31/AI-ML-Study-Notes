import React from 'react';

export const PageContainer = ({
  children,
  className = '',
  maxWidth = 'max-w-7xl',
  padding = 'px-4 sm:px-6 lg:px-8 py-8',
}) => {
  return (
    <main className={`flex-1 ${maxWidth} mx-auto w-full ${padding} ${className}`}>
      {children}
    </main>
  );
};

export default PageContainer;
