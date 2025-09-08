import React from 'react';
import { AppProvider } from './contexts/AppProvider';
import { useAuth } from './contexts/AuthContext';
import { LoginForm } from './components/auth/LoginForm';
import { Dashboard } from './components/dashboard/Dashboard';

const AppContent: React.FC = () => {
  const { isAuthenticated, isLoading } = useAuth();

  if (isLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gray-50">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-green-600"></div>
      </div>
    );
  }

  return isAuthenticated ? <Dashboard /> : <LoginForm />;
};

function App() {
  return (
    <AppProvider>
      <AppContent />
    </AppProvider>
  );
}

export default App;
