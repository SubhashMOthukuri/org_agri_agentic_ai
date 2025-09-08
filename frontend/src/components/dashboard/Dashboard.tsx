import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { useAuth } from '../../contexts/AuthContext';
import { WeatherDashboard } from './WeatherDashboard';
import { MarketIntelligence } from './MarketIntelligence';
import { PestDetection } from './PestDetection';
import { YieldPrediction } from './YieldPrediction';
import { Header } from './Header';
import { Sidebar } from './Sidebar';
import { Footer } from './Footer';

type DashboardTab = 'overview' | 'weather' | 'market' | 'pest' | 'yield' | 'settings';

export const Dashboard: React.FC = () => {
  const { user, logout } = useAuth();
  const [activeTab, setActiveTab] = useState<DashboardTab>('overview');
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);

  const renderContent = () => {
    switch (activeTab) {
      case 'weather':
        return <WeatherDashboard farmId="farm_123" />;
      case 'market':
        return <MarketIntelligence farmId="farm_123" />;
      case 'pest':
        return <PestDetection farmId="farm_123" />;
      case 'yield':
        return <YieldPrediction farmId="farm_123" />;
      case 'overview':
      default:
        return <OverviewDashboard farmId="farm_123" />;
    }
  };

  return (
    <div className="min-h-screen bg-gray-50 flex">
      <Sidebar
        activeTab={activeTab}
        onTabChange={setActiveTab}
        collapsed={sidebarCollapsed}
        onToggleCollapse={() => setSidebarCollapsed(!sidebarCollapsed)}
      />
      
      <div className="flex-1 flex flex-col">
        <Header
          user={user}
          onLogout={logout}
          onToggleSidebar={() => setSidebarCollapsed(!sidebarCollapsed)}
        />
        
        <main className="flex-1 p-6">
          <motion.div
            key={activeTab}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.3 }}
          >
            {renderContent()}
          </motion.div>
        </main>
        
        <Footer />
      </div>
    </div>
  );
};

const OverviewDashboard: React.FC<{ farmId: string }> = ({ farmId }) => {
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold text-gray-900">Farm Overview</h1>
        <p className="text-gray-600 mt-2">Welcome to your agricultural intelligence dashboard</p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <WeatherDashboard farmId={farmId} />
        <MarketIntelligence farmId={farmId} />
        <PestDetection farmId={farmId} />
        <YieldPrediction farmId={farmId} />
      </div>
    </div>
  );
};
