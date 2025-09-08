import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { use90HzPerformance } from '../../hooks/use90Hz';

interface PerformanceMonitorProps {
  show?: boolean;
  position?: 'top-right' | 'top-left' | 'bottom-right' | 'bottom-left';
}

export const PerformanceMonitor: React.FC<PerformanceMonitorProps> = ({ 
  show = false, 
  position = 'top-right' 
}) => {
  const { fps, is90Hz, is60Hz, isLowFPS } = use90HzPerformance();
  const [isVisible, setIsVisible] = useState(show);

  useEffect(() => {
    setIsVisible(show);
  }, [show]);

  if (!isVisible) return null;

  const positionClasses = {
    'top-right': 'top-4 right-4',
    'top-left': 'top-4 left-4',
    'bottom-right': 'bottom-4 right-4',
    'bottom-left': 'bottom-4 left-4',
  };

  const getFPSColor = () => {
    if (is90Hz) return 'text-green-600';
    if (is60Hz) return 'text-yellow-600';
    if (isLowFPS) return 'text-red-600';
    return 'text-gray-600';
  };

  const getFPSStatus = () => {
    if (is90Hz) return '90Hz Ready';
    if (is60Hz) return '60Hz';
    if (isLowFPS) return 'Low FPS';
    return 'Unknown';
  };

  return (
    <AnimatePresence>
      <motion.div
        initial={{ opacity: 0, scale: 0.8 }}
        animate={{ opacity: 1, scale: 1 }}
        exit={{ opacity: 0, scale: 0.8 }}
        className={`fixed ${positionClasses[position]} z-50`}
      >
        <div className="bg-black/80 backdrop-blur-sm rounded-lg px-3 py-2 text-white text-xs font-mono">
          <div className="flex items-center space-x-2">
            <div className={`w-2 h-2 rounded-full ${
              is90Hz ? 'bg-green-500' : 
              is60Hz ? 'bg-yellow-500' : 
              isLowFPS ? 'bg-red-500' : 'bg-gray-500'
            }`} />
            <span className={getFPSColor()}>
              {fps}fps
            </span>
            <span className="text-gray-400">
              {getFPSStatus()}
            </span>
          </div>
        </div>
      </motion.div>
    </AnimatePresence>
  );
};

// Performance optimization tips component
export const PerformanceTips: React.FC = () => {
  const { is90Hz, is60Hz, isLowFPS } = use90HzPerformance();
  const [showTips, setShowTips] = useState(false);

  const getTip = () => {
    if (is90Hz) {
      return {
        title: "90Hz Optimized",
        message: "Your display is running at 90Hz for smooth interactions",
        color: "green"
      };
    }
    if (is60Hz) {
      return {
        title: "60Hz Display",
        message: "Consider enabling 90Hz mode for smoother animations",
        color: "yellow"
      };
    }
    if (isLowFPS) {
      return {
        title: "Performance Issue",
        message: "Close other applications to improve performance",
        color: "red"
      };
    }
    return null;
  };

  const tip = getTip();
  if (!tip) return null;

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className={`fixed top-4 left-1/2 transform -translate-x-1/2 z-50 ${
        tip.color === 'green' ? 'bg-green-50 border-green-200 text-green-800' :
        tip.color === 'yellow' ? 'bg-yellow-50 border-yellow-200 text-yellow-800' :
        'bg-red-50 border-red-200 text-red-800'
      } border rounded-lg px-4 py-2 max-w-md`}
    >
      <div className="flex items-center space-x-2">
        <div className={`w-2 h-2 rounded-full ${
          tip.color === 'green' ? 'bg-green-500' :
          tip.color === 'yellow' ? 'bg-yellow-500' :
          'bg-red-500'
        }`} />
        <div>
          <div className="font-semibold text-sm">{tip.title}</div>
          <div className="text-xs">{tip.message}</div>
        </div>
        <button
          onClick={() => setShowTips(false)}
          className="text-gray-400 hover:text-gray-600"
        >
          ×
        </button>
      </div>
    </motion.div>
  );
};
