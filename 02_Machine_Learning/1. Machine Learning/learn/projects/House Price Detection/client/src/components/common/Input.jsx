import React from 'react';

export const Input = ({
  label,
  id,
  name,
  type = 'text',
  value,
  onChange,
  placeholder,
  error,
  helperText,
  required = false,
  disabled = false,
  icon: Icon,
  unit,
  className = '',
  ...props
}) => {
  const inputId = id || name;

  return (
    <div className={`flex flex-col gap-1.5 ${className}`}>
      {label && (
        <label
          htmlFor={inputId}
          className="text-xs font-semibold text-slate-700 flex items-center justify-between"
        >
          <span>
            {label}
            {required && <span className="text-emerald-600 ml-1">*</span>}
          </span>
          {unit && <span className="text-slate-400 font-normal">({unit})</span>}
        </label>
      )}

      <div className="relative rounded-xl shadow-xs">
        {Icon && (
          <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none text-slate-400">
            <Icon className="w-4 h-4" />
          </div>
        )}
        <input
          id={inputId}
          name={name}
          type={type}
          value={value}
          onChange={onChange}
          disabled={disabled}
          placeholder={placeholder}
          required={required}
          className={`block w-full rounded-xl border text-sm transition-colors duration-150 focus:outline-none focus:ring-2 focus:ring-emerald-500/20 disabled:bg-slate-50 disabled:text-slate-400 ${
            Icon ? 'pl-9' : 'pl-3.5'
          } ${unit ? 'pr-12' : 'pr-3.5'} py-2.5 ${
            error
              ? 'border-red-400 bg-red-50/20 text-red-900 focus:border-red-500'
              : 'border-slate-200 text-slate-900 hover:border-slate-300 focus:border-emerald-500 bg-white'
          }`}
          {...props}
        />
        {unit && (
          <div className="absolute inset-y-0 right-0 pr-3.5 flex items-center pointer-events-none text-xs text-slate-400 font-medium">
            {unit}
          </div>
        )}
      </div>

      {error ? (
        <p className="text-xs text-red-600 mt-0.5">{error}</p>
      ) : helperText ? (
        <p className="text-xs text-slate-400 mt-0.5">{helperText}</p>
      ) : null}
    </div>
  );
};

export default Input;
