import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { useAuth } from '../../contexts/AuthContext';
import { Button } from '../ui/Button';
import { Input } from '../ui/Input';
import { Card, CardHeader, CardContent, CardTitle, CardDescription } from '../ui/Card';
import { LoginRequest } from '../../types/api';

export const LoginForm: React.FC = () => {
  const { login, isLoading } = useAuth();
  const [formData, setFormData] = useState<LoginRequest>({
    username: '',
    password: ''
  });
  const [error, setError] = useState<string>('');

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');

    try {
      await login(formData);
    } catch (err: any) {
      setError(err.message || 'Login failed. Please try again.');
    }
  };

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setFormData(prev => ({
      ...prev,
      [e.target.name]: e.target.value
    }));
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-gray-50 py-12 px-4 sm:px-6 lg:px-8">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
        className="max-w-md w-full space-y-8"
      >
        <Card>
          <CardHeader className="text-center">
            <div className="mx-auto h-12 w-12 bg-gradient-to-br from-green-500 to-green-700 rounded-lg flex items-center justify-center mb-4">
              <span className="text-white font-bold text-xl">🌾</span>
            </div>
            <CardTitle className="text-2xl">AgriAI Dashboard</CardTitle>
            <CardDescription>
              Sign in to access your agricultural intelligence platform
            </CardDescription>
          </CardHeader>
          <CardContent>
            <form onSubmit={handleSubmit} className="space-y-6">
              <Input
                label="Username"
                name="username"
                type="text"
                value={formData.username}
                onChange={handleChange}
                required
                placeholder="Enter your username"
              />
              
              <Input
                label="Password"
                name="password"
                type="password"
                value={formData.password}
                onChange={handleChange}
                required
                placeholder="Enter your password"
              />

              {error && (
                <motion.div
                  initial={{ opacity: 0, y: -10 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="bg-red-50 border border-red-200 rounded-lg p-3"
                >
                  <p className="text-sm text-red-600">{error}</p>
                </motion.div>
              )}

              <Button
                type="submit"
                className="w-full"
                loading={isLoading}
                disabled={!formData.username || !formData.password}
              >
                Sign In
              </Button>
            </form>

            <div className="mt-6 text-center">
              <p className="text-sm text-gray-600">
                Demo credentials: <br />
                <span className="font-mono text-xs bg-gray-100 px-2 py-1 rounded">
                  admin / admin123
                </span>
              </p>
            </div>
          </CardContent>
        </Card>
      </motion.div>
    </div>
  );
};
