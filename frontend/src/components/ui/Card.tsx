import React from 'react';
import { motion } from 'framer-motion';
import { clsx } from 'clsx';
import { hoverHandlers, animations } from '../../utils/animations';
import { use90Hz } from '../../hooks/use90Hz';

interface CardProps {
  children: React.ReactNode;
  className?: string;
  hover?: boolean;
  padding?: 'sm' | 'md' | 'lg';
}

export const Card: React.FC<CardProps> = ({ 
  children, 
  className = '', 
  hover = true, 
  padding = 'md' 
}) => {
  const { controls, hoverAnimation, resetAnimation } = use90Hz();
  
  const paddingClasses = {
    sm: 'p-4',
    md: 'p-6',
    lg: 'p-8'
  };

  const handleMouseEnter = () => {
    if (hover) {
      hoverAnimation(1.02);
    }
  };

  const handleMouseLeave = () => {
    if (hover) {
      resetAnimation();
    }
  };

  return (
    <motion.div
      animate={controls}
      onMouseEnter={handleMouseEnter}
      onMouseLeave={handleMouseLeave}
      className={clsx(
        'bg-white rounded-xl border border-gray-200 shadow-card',
        paddingClasses[padding],
        className
      )}
      style={{
        willChange: 'transform, box-shadow', // Optimize for 90Hz
        transform: 'translateZ(0)', // Force hardware acceleration
      }}
    >
      {children}
    </motion.div>
  );
};

export const CardHeader: React.FC<{ children: React.ReactNode; className?: string }> = ({ 
  children, 
  className = '' 
}) => (
  <div className={clsx('mb-4', className)}>
    {children}
  </div>
);

export const CardContent: React.FC<{ children: React.ReactNode; className?: string }> = ({ 
  children, 
  className = '' 
}) => (
  <div className={clsx('', className)}>
    {children}
  </div>
);

export const CardTitle: React.FC<{ children: React.ReactNode; className?: string }> = ({ 
  children, 
  className = '' 
}) => (
  <h3 className={clsx('text-lg font-semibold text-gray-900', className)}>
    {children}
  </h3>
);

export const CardDescription: React.FC<{ children: React.ReactNode; className?: string }> = ({ 
  children, 
  className = '' 
}) => (
  <p className={clsx('text-sm text-gray-500 mt-1', className)}>
    {children}
  </p>
);
