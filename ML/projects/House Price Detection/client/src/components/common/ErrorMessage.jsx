import React from 'react';
import { AlertCircle, RefreshCw } from 'lucide-react';
import Button from './Button';

export const ErrorMessage = ({
  title = 'Something went wrong',
  message,
  onRetry,
  className = '',
}) => {
  return (
    <div
      className={`p-4 rounded-2xl bg-red-50/80 border border-red-200/80 flex items-start gap-3.5 ${className}`}
    >
      <div className="p-1.5 rounded-lg bg-red-100 text-red-600 shrink-0 mt-0.5">
        <AlertCircle className="w-5 h-5" />
      </div>
      <div className="flex-1">
        <h4 className="text-sm font-semibold text-red-900">{title}</h4>
        {message && <p className="text-xs text-red-700 mt-1 leading-relaxed">{message}</p>}
        {onRetry && (
          <div className="mt-3">
            <Button
              size="sm"
              variant="outline"
              onClick={onRetry}
              icon={RefreshCw}
              className="text-red-700 border-red-300 hover:bg-red-100/50"
            >
              Try Again
            </Button>
          </div>
        )}
      </div>
    </div>
  );
};

export default ErrorMessage;
