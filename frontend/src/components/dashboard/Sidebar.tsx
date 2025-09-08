import React from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  HomeIcon,
  CloudIcon,
  ChartBarIcon,
  BugAntIcon,
  CpuChipIcon,
  Cog6ToothIcon,
  Bars3Icon
} from '@heroicons/react/24/outline';

type DashboardTab = 'overview' | 'weather' | 'market' | 'pest' | 'yield' | 'settings';

interface SidebarProps {
  activeTab: DashboardTab;
  onTabChange: (tab: DashboardTab) => void;
  collapsed: boolean;
  onToggleCollapse: () => void;
}

const navigationItems = [
  { id: 'overview', label: 'Dashboard', icon: HomeIcon },
  { id: 'weather', label: 'Weather', icon: CloudIcon },
  { id: 'market', label: 'Market', icon: ChartBarIcon },
  { id: 'pest', label: 'Pest Control', icon: BugAntIcon },
  { id: 'yield', label: 'Yield Prediction', icon: CpuChipIcon },
  { id: 'settings', label: 'Settings', icon: Cog6ToothIcon },
] as const;

export const Sidebar: React.FC<SidebarProps> = ({ 
  activeTab, 
  onTabChange, 
  collapsed, 
  onToggleCollapse 
}) => {
  return (
    <motion.aside
      animate={{ width: collapsed ? 80 : 280 }}
      className="bg-white border-r border-gray-200 flex flex-col h-full"
    >
      {/* Header */}
      <div className="p-4 border-b border-gray-100">
        <div className="flex items-center justify-between">
          <AnimatePresence>
            {!collapsed && (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                className="flex items-center space-x-3"
              >
                <div className="h-8 w-8 bg-gradient-to-br from-green-500 to-green-700 rounded-lg flex items-center justify-center">
                  <span className="text-white font-bold text-sm">🌾</span>
                </div>
                <div>
                  <h2 className="text-lg font-bold text-gray-900">AgriAI</h2>
                  <p className="text-xs text-gray-500">Dashboard</p>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
          <button
            onClick={onToggleCollapse}
            className="p-2 hover:bg-gray-100 rounded-lg transition-colors"
          >
            <Bars3Icon className="h-5 w-5 text-gray-600" />
          </button>
        </div>
      </div>

      {/* Navigation */}
      <nav className="flex-1 p-4 space-y-2">
        {navigationItems.map((item) => (
          <NavigationItem
            key={item.id}
            item={item}
            isActive={activeTab === item.id}
            collapsed={collapsed}
            onClick={() => onTabChange(item.id as DashboardTab)}
          />
        ))}
      </nav>

      {/* Quick Actions */}
      <div className="p-4 border-t border-gray-100">
        <QuickActionsPanel collapsed={collapsed} />
      </div>
    </motion.aside>
  );
};

const NavigationItem: React.FC<{
  item: typeof navigationItems[0];
  isActive: boolean;
  collapsed: boolean;
  onClick: () => void;
}> = ({ item, isActive, collapsed, onClick }) => {
  return (
    <motion.button
      whileHover={{ scale: 1.02 }}
      whileTap={{ scale: 0.98 }}
      onClick={onClick}
      className={`
        w-full flex items-center space-x-3 px-3 py-2 rounded-lg transition-all duration-200
        ${isActive 
          ? 'bg-green-50 text-green-700 border border-green-200' 
          : 'text-gray-600 hover:bg-gray-50 hover:text-gray-900'
        }
      `}
    >
      <item.icon className={`h-5 w-5 ${isActive ? 'text-green-600' : 'text-gray-500'}`} />
      <AnimatePresence>
        {!collapsed && (
          <motion.span
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="font-medium"
          >
            {item.label}
          </motion.span>
        )}
      </AnimatePresence>
      {isActive && !collapsed && (
        <motion.div
          layoutId="activeIndicator"
          className="ml-auto h-2 w-2 bg-green-500 rounded-full"
        />
      )}
    </motion.button>
  );
};

const QuickActionsPanel: React.FC<{ collapsed: boolean }> = ({ collapsed }) => {
  const quickActions = [
    { label: 'Generate Report', icon: '📊' },
    { label: 'Export Data', icon: '📤' },
    { label: 'System Status', icon: '🔧' },
  ];

  return (
    <div className="space-y-2">
      <AnimatePresence>
        {!collapsed && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-2"
          >
            Quick Actions
          </motion.div>
        )}
      </AnimatePresence>
      
      {quickActions.map((action, index) => (
        <motion.button
          key={action.label}
          whileHover={{ scale: 1.02 }}
          className="w-full flex items-center space-x-3 px-3 py-2 text-sm text-gray-600 hover:bg-gray-50 rounded-lg transition-colors"
        >
          <span className="text-lg">{action.icon}</span>
          <AnimatePresence>
            {!collapsed && (
              <motion.span
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
              >
                {action.label}
              </motion.span>
            )}
          </AnimatePresence>
        </motion.button>
      ))}
    </div>
  );
};
