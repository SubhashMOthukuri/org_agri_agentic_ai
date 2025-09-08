import React, { forwardRef } from 'react';
import { clsx } from 'clsx';

interface InputProps extends React.InputHTMLAttributes<HTMLInputElement> {
  label?: string;
  error?: string;
  helperText?: string;
  leftIcon?: React.ComponentType<any>;
  rightIcon?: React.ComponentType<any>;
  onRightIconClick?: () => void;
}

export const Input = forwardRef<HTMLInputElement, InputProps>(
  ({ 
    label, 
    error, 
    helperText, 
    leftIcon: LeftIcon, 
    rightIcon: RightIcon, 
    onRightIconClick,
    className = '',
    ...props 
  }, ref) => {
    return (
      <div className="w-full">
        {label && (
          <label className="block text-sm font-medium text-gray-700 mb-1">
            {label}
          </label>
        )}
        <div className="relative">
          {LeftIcon && (
            <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
              <LeftIcon className="h-5 w-5 text-gray-400" />
            </div>
          )}
          <input
            ref={ref}
            className={clsx(
              'block w-full rounded-lg border-gray-300 shadow-sm focus:border-green-500 focus:ring-green-500 sm:text-sm',
              LeftIcon && 'pl-10',
              RightIcon && 'pr-10',
              error && 'border-red-300 focus:border-red-500 focus:ring-red-500',
              className
            )}
            {...props}
          />
          {RightIcon && (
            <div 
              className={clsx(
                'absolute inset-y-0 right-0 pr-3 flex items-center',
                onRightIconClick ? 'cursor-pointer' : 'pointer-events-none'
              )}
              onClick={onRightIconClick}
            >
              <RightIcon className="h-5 w-5 text-gray-400" />
            </div>
          )}
        </div>
        {error && (
          <p className="mt-1 text-sm text-red-600">{error}</p>
        )}
        {helperText && !error && (
          <p className="mt-1 text-sm text-gray-500">{helperText}</p>
        )}
      </div>
    );
  }
);

Input.displayName = 'Input';
