import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { User } from '../../types/api';
import { Button } from '../ui/Button';
import { Input } from '../ui/Input';
import { Badge } from '../ui/Badge';
import { 
  BellIcon, 
  MagnifyingGlassIcon, 
  Bars3Icon,
  UserCircleIcon,
  Cog6ToothIcon,
  QuestionMarkCircleIcon,
  ArrowRightOnRectangleIcon,
  ChevronDownIcon
} from '@heroicons/react/24/outline';

interface HeaderProps {
  user: User | null;
  onLogout: () => void;
  onToggleSidebar: () => void;
}

export const Header: React.FC<HeaderProps> = ({ user, onLogout, onToggleSidebar }) => {
  const [searchQuery, setSearchQuery] = useState('');
  const [showNotifications, setShowNotifications] = useState(false);
  const [showUserMenu, setShowUserMenu] = useState(false);

  const notifications = [
    { id: 1, type: 'pest', message: 'Aphids detected in Field A-3', time: '2 min ago', severity: 'high' },
    { id: 2, type: 'weather', message: 'Heavy rain expected in 2 hours', time: '15 min ago', severity: 'medium' },
    { id: 3, type: 'market', message: 'Wheat prices increased by 5.3%', time: '1 hour ago', severity: 'low' },
  ];

  const unreadCount = notifications.length;

  return (
    <header className="bg-white border-b border-gray-200 px-6 py-4">
      <div className="flex items-center justify-between">
        {/* Left side */}
        <div className="flex items-center space-x-4">
          <Button
            variant="ghost"
            size="sm"
            onClick={onToggleSidebar}
            className="lg:hidden"
          >
            <Bars3Icon className="h-5 w-5" />
          </Button>
          
          <div className="flex items-center space-x-3">
            <div className="h-8 w-8 bg-gradient-to-br from-green-500 to-green-700 rounded-lg flex items-center justify-center">
              <span className="text-white font-bold text-sm">🌾</span>
            </div>
            <div className="hidden sm:block">
              <h1 className="text-xl font-bold text-gray-900">AgriAI</h1>
              <p className="text-xs text-gray-500">Agricultural Intelligence</p>
            </div>
          </div>
        </div>

        {/* Center - Search */}
        <div className="flex-1 max-w-2xl mx-8 hidden md:block">
          <div className="relative">
            <MagnifyingGlassIcon className="absolute left-3 top-1/2 transform -translate-y-1/2 h-5 w-5 text-gray-400" />
            <Input
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Search farms, crops, weather..."
              className="pl-10 pr-4"
            />
          </div>
        </div>

        {/* Right side */}
        <div className="flex items-center space-x-4">
          {/* Notifications */}
          <div className="relative">
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setShowNotifications(!showNotifications)}
              className="relative"
            >
              <BellIcon className="h-5 w-5" />
              {unreadCount > 0 && (
                <motion.div
                  initial={{ scale: 0 }}
                  animate={{ scale: 1 }}
                  className="absolute -top-1 -right-1 bg-red-500 text-white text-xs rounded-full h-5 w-5 flex items-center justify-center font-bold"
                >
                  {unreadCount > 9 ? '9+' : unreadCount}
                </motion.div>
              )}
            </Button>

            <AnimatePresence>
              {showNotifications && (
                <motion.div
                  initial={{ opacity: 0, y: -10, scale: 0.95 }}
                  animate={{ opacity: 1, y: 0, scale: 1 }}
                  exit={{ opacity: 0, y: -10, scale: 0.95 }}
                  className="absolute right-0 top-full mt-2 w-80 bg-white rounded-xl shadow-xl border border-gray-200 z-50"
                >
                  <div className="p-4 border-b border-gray-100">
                    <h3 className="font-semibold text-gray-900">Notifications</h3>
                  </div>
                  <div className="max-h-96 overflow-y-auto">
                    {notifications.map((notification) => (
                      <div
                        key={notification.id}
                        className="p-4 border-b border-gray-100 hover:bg-gray-50 cursor-pointer"
                      >
                        <div className="flex items-start space-x-3">
                          <div className={`h-2 w-2 rounded-full mt-2 ${
                            notification.severity === 'high' ? 'bg-red-500' :
                            notification.severity === 'medium' ? 'bg-yellow-500' : 'bg-green-500'
                          }`} />
                          <div className="flex-1">
                            <p className="text-sm text-gray-900">{notification.message}</p>
                            <p className="text-xs text-gray-500 mt-1">{notification.time}</p>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          </div>

          {/* User Menu */}
          <div className="relative">
            <Button
              variant="ghost"
              onClick={() => setShowUserMenu(!showUserMenu)}
              className="flex items-center space-x-2"
            >
              <div className="h-8 w-8 bg-gradient-to-br from-blue-500 to-purple-600 rounded-full flex items-center justify-center">
                <span className="text-white font-semibold text-sm">
                  {user?.username?.charAt(0).toUpperCase() || 'U'}
                </span>
              </div>
              <div className="hidden md:block text-left">
                <p className="text-sm font-medium text-gray-900">{user?.username || 'User'}</p>
                <p className="text-xs text-gray-500">{user?.role || 'Farmer'}</p>
              </div>
              <ChevronDownIcon className="h-4 w-4 text-gray-400" />
            </Button>

            <AnimatePresence>
              {showUserMenu && (
                <motion.div
                  initial={{ opacity: 0, y: -10, scale: 0.95 }}
                  animate={{ opacity: 1, y: 0, scale: 1 }}
                  exit={{ opacity: 0, y: -10, scale: 0.95 }}
                  className="absolute right-0 top-full mt-2 w-64 bg-white rounded-xl shadow-xl border border-gray-200 z-50"
                >
                  <div className="p-4 border-b border-gray-100">
                    <div className="flex items-center space-x-3">
                      <div className="h-12 w-12 bg-gradient-to-br from-blue-500 to-purple-600 rounded-full flex items-center justify-center">
                        <span className="text-white font-semibold text-lg">
                          {user?.username?.charAt(0).toUpperCase() || 'U'}
                        </span>
                      </div>
                      <div>
                        <p className="font-semibold text-gray-900">{user?.username || 'User'}</p>
                        <p className="text-sm text-gray-500">{user?.email || 'user@example.com'}</p>
                      </div>
                    </div>
                  </div>
                  <div className="py-2">
                    <MenuItem icon={UserCircleIcon} label="Profile" onClick={() => {}} />
                    <MenuItem icon={Cog6ToothIcon} label="Settings" onClick={() => {}} />
                    <MenuItem icon={QuestionMarkCircleIcon} label="Help & Support" onClick={() => {}} />
                    <div className="border-t border-gray-100 my-2"></div>
                    <MenuItem icon={ArrowRightOnRectangleIcon} label="Sign Out" onClick={onLogout} />
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        </div>
      </div>
    </header>
  );
};

const MenuItem: React.FC<{
  icon: React.ComponentType<any>;
  label: string;
  onClick: () => void;
}> = ({ icon: Icon, label, onClick }) => (
  <button
    onClick={onClick}
    className="w-full flex items-center space-x-3 px-4 py-2 text-sm text-gray-700 hover:bg-gray-50"
  >
    <Icon className="h-5 w-5" />
    <span>{label}</span>
  </button>
);
