import React from 'react';
import { motion } from 'framer-motion';
import { clsx } from 'clsx';
import { hoverHandlers, animations } from '../../utils/animations';
import { use90Hz } from '../../hooks/use90Hz';

interface ButtonProps {
  children: React.ReactNode;
  variant?: 'primary' | 'secondary' | 'outline' | 'ghost' | 'danger';
  size?: 'sm' | 'md' | 'lg';
  loading?: boolean;
  disabled?: boolean;
  onClick?: () => void;
  className?: string;
  icon?: React.ComponentType<any>;
  type?: 'button' | 'submit' | 'reset';
}

export const Button: React.FC<ButtonProps> = ({
  children,
  variant = 'primary',
  size = 'md',
  loading = false,
  disabled = false,
  onClick,
  className = '',
  icon: Icon,
  type = 'button'
}) => {
  const { controls, hoverAnimation, tapAnimation, resetAnimation } = use90Hz();
  
  const baseClasses = 'inline-flex items-center justify-center font-medium rounded-lg focus:outline-none focus:ring-2 focus:ring-offset-2';
  
  const variantClasses = {
    primary: 'bg-green-600 text-white hover:bg-green-700 focus:ring-green-500 shadow-sm',
    secondary: 'bg-gray-100 text-gray-900 hover:bg-gray-200 focus:ring-gray-500',
    outline: 'border border-gray-300 text-gray-700 hover:bg-gray-50 focus:ring-gray-500 bg-white',
    ghost: 'text-gray-600 hover:bg-gray-100 focus:ring-gray-500',
    danger: 'bg-red-600 text-white hover:bg-red-700 focus:ring-red-500 shadow-sm'
  };

  const sizeClasses = {
    sm: 'px-3 py-1.5 text-sm',
    md: 'px-4 py-2 text-sm',
    lg: 'px-6 py-3 text-base'
  };

  const handleMouseEnter = () => {
    if (!disabled && !loading) {
      hoverAnimation(1.02);
    }
  };

  const handleMouseLeave = () => {
    if (!disabled && !loading) {
      resetAnimation();
    }
  };

  const handleMouseDown = () => {
    if (!disabled && !loading) {
      tapAnimation(0.98);
    }
  };

  const handleMouseUp = () => {
    if (!disabled && !loading) {
      resetAnimation();
    }
  };

  return (
    <motion.button
      animate={controls}
      onMouseEnter={handleMouseEnter}
      onMouseLeave={handleMouseLeave}
      onMouseDown={handleMouseDown}
      onMouseUp={handleMouseUp}
      onClick={onClick}
      disabled={disabled || loading}
      type={type}
      className={clsx(
        baseClasses,
        variantClasses[variant],
        sizeClasses[size],
        disabled || loading ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer',
        className
      )}
      style={{
        willChange: 'transform', // Optimize for 90Hz
        transform: 'translateZ(0)', // Force hardware acceleration
      }}
    >
      {loading && (
        <motion.div
          animate={animations.spinner.animate}
          className="mr-2 h-4 w-4 border-2 border-current border-t-transparent rounded-full"
        />
      )}
      {Icon && !loading && <Icon className="mr-2 h-4 w-4" />}
      {children}
    </motion.button>
  );
};
