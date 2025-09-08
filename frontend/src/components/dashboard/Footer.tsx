import React from 'react';
import { motion } from 'framer-motion';
import { CheckCircleIcon, ExclamationTriangleIcon, XCircleIcon } from '@heroicons/react/24/outline';

export const Footer: React.FC = () => {
  const systemStatus = 'healthy';
  const lastUpdate = new Date();
  const metrics = {
    apiLatency: 45,
    dbLatency: 12,
    wsConnections: 3
  };

  const getStatusConfig = (status: string) => {
    switch (status) {
      case 'healthy':
        return { color: 'green', icon: CheckCircleIcon, text: 'All Systems Operational' };
      case 'warning':
        return { color: 'yellow', icon: ExclamationTriangleIcon, text: 'Minor Issues Detected' };
      case 'error':
        return { color: 'red', icon: XCircleIcon, text: 'System Issues Detected' };
      default:
        return { color: 'gray', icon: CheckCircleIcon, text: 'Unknown Status' };
    }
  };

  const config = getStatusConfig(systemStatus);
  const timeAgo = formatTimeAgo(lastUpdate);

  return (
    <footer className="bg-white border-t border-gray-200 px-6 py-4">
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-6">
          <div className="flex items-center space-x-2">
            <div className="h-6 w-6 bg-gradient-to-br from-green-500 to-green-700 rounded-lg flex items-center justify-center">
              <span className="text-white font-bold text-xs">🌾</span>
            </div>
            <span className="text-sm font-medium text-gray-900">AgriAI Dashboard</span>
          </div>
          
          <div className="flex items-center space-x-4">
            <div className="flex items-center space-x-2">
              <config.icon className={`h-4 w-4 text-${config.color}-500`} />
              <span className={`text-sm font-medium text-${config.color}-700`}>
                {config.text}
              </span>
            </div>
            <div className="text-xs text-gray-500">
              Last update: {timeAgo}
            </div>
          </div>
        </div>

        <div className="flex items-center space-x-6">
          <div className="flex items-center space-x-4 text-xs text-gray-500">
            <span>API: {metrics.apiLatency}ms</span>
            <span>DB: {metrics.dbLatency}ms</span>
            <span>WS: {metrics.wsConnections} active</span>
          </div>
          
          <div className="flex items-center space-x-4">
            <FooterLink href="/privacy">Privacy</FooterLink>
            <FooterLink href="/terms">Terms</FooterLink>
            <FooterLink href="/support">Support</FooterLink>
            <span className="text-xs text-gray-500">© 2024</span>
          </div>
        </div>
      </div>
    </footer>
  );
};

const FooterLink: React.FC<{ href: string; children: React.ReactNode }> = ({ href, children }) => (
  <motion.a
    href={href}
    whileHover={{ scale: 1.05 }}
    className="text-xs text-gray-500 hover:text-gray-700 transition-colors"
  >
    {children}
  </motion.a>
);

const formatTimeAgo = (date: Date): string => {
  const now = new Date();
  const diffInSeconds = Math.floor((now.getTime() - date.getTime()) / 1000);
  
  if (diffInSeconds < 60) {
    return `${diffInSeconds}s ago`;
  } else if (diffInSeconds < 3600) {
    const minutes = Math.floor(diffInSeconds / 60);
    return `${minutes}m ago`;
  } else if (diffInSeconds < 86400) {
    const hours = Math.floor(diffInSeconds / 3600);
    return `${hours}h ago`;
  } else {
    const days = Math.floor(diffInSeconds / 86400);
    return `${days}d ago`;
  }
};
