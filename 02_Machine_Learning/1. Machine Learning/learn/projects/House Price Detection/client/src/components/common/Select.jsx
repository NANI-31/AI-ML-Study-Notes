import React from 'react';

export const Select = ({
  label,
  id,
  name,
  value,
  onChange,
  options = [],
  placeholder = 'Select an option',
  error,
  helperText,
  required = false,
  disabled = false,
  className = '',
  ...props
}) => {
  const selectId = id || name;

  return (
    <div className={`flex flex-col gap-1.5 ${className}`}>
      {label && (
        <label
          htmlFor={selectId}
          className="text-xs font-semibold text-slate-700 flex items-center"
        >
          {label}
          {required && <span className="text-emerald-600 ml-1">*</span>}
        </label>
      )}

      <div className="relative rounded-xl shadow-xs">
        <select
          id={selectId}
          name={name}
          value={value ?? ''}
          onChange={onChange}
          disabled={disabled}
          required={required}
          className={`block w-full appearance-none rounded-xl border text-sm transition-colors duration-150 pl-3.5 pr-10 py-2.5 bg-white focus:outline-none focus:ring-2 focus:ring-emerald-500/20 disabled:bg-slate-50 disabled:text-slate-400 disabled:cursor-not-allowed ${
            error
              ? 'border-red-400 bg-red-50/20 text-red-900 focus:border-red-500'
              : 'border-slate-200 text-slate-900 hover:border-slate-300 focus:border-emerald-500'
          }`}
          {...props}
        >
          {placeholder && <option value="">{placeholder}</option>}
          {options.map((option, idx) => {
            if (typeof option === 'string' || typeof option === 'number') {
              return (
                <option key={idx} value={option}>
                  {option}
                </option>
              );
            }
            return (
              <option key={option.value || option.id || idx} value={option.value ?? option.id}>
                {option.label || option.name}
              </option>
            );
          })}
        </select>

        <div className="absolute inset-y-0 right-0 flex items-center pr-3 pointer-events-none text-slate-400">
          <svg className="w-4 h-4 fill-current" viewBox="0 0 20 20">
            <path
              fillRule="evenodd"
              d="M5.293 7.293a1 1 0 011.414 0L10 10.586l3.293-3.293a1 1 0 111.414 1.414l-4 4a1 1 0 01-1.414 0l-4-4a1 1 0 010-1.414z"
              clipRule="evenodd"
            />
          </svg>
        </div>
      </div>

      {error ? (
        <p className="text-xs text-red-600 mt-0.5">{error}</p>
      ) : helperText ? (
        <p className="text-xs text-slate-400 mt-0.5">{helperText}</p>
      ) : null}
    </div>
  );
};

export default Select;
