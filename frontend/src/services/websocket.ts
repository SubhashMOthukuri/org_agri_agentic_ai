import { WebSocketMessage } from '../types/api';
import { config } from '../config/environment';

class WebSocketService {
  private ws: WebSocket | null = null;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  private reconnectInterval = 1000;
  private listeners: Map<string, Set<(data: any) => void>> = new Map();

  constructor() {
    this.connect();
  }

  private connect() {
    const wsUrl = config.wsUrl;
    
    try {
      this.ws = new WebSocket(wsUrl);

      this.ws.onopen = () => {
        console.log('WebSocket connected');
        this.reconnectAttempts = 0;
        this.emit('connection', { status: 'connected' });
      };

      this.ws.onmessage = (event) => {
        try {
          const message: WebSocketMessage = JSON.parse(event.data);
          this.emit(message.type, message.data);
        } catch (error) {
          console.error('Error parsing WebSocket message:', error);
        }
      };

      this.ws.onclose = () => {
        console.log('WebSocket disconnected');
        this.emit('connection', { status: 'disconnected' });
        this.handleReconnect();
      };

      this.ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        this.emit('error', error);
      };
    } catch (error) {
      console.error('Failed to create WebSocket connection:', error);
      this.handleReconnect();
    }
  }

  private handleReconnect() {
    if (this.reconnectAttempts < this.maxReconnectAttempts) {
      this.reconnectAttempts++;
      console.log(`Attempting to reconnect... (${this.reconnectAttempts}/${this.maxReconnectAttempts})`);
      
      setTimeout(() => {
        this.connect();
      }, this.reconnectInterval * this.reconnectAttempts);
    } else {
      console.error('Max reconnection attempts reached');
      this.emit('connection', { status: 'failed' });
    }
  }

  private emit(type: string, data: any) {
    const listeners = this.listeners.get(type);
    if (listeners) {
      listeners.forEach(listener => {
        try {
          listener(data);
        } catch (error) {
          console.error(`Error in WebSocket listener for ${type}:`, error);
        }
      });
    }
  }

  // Subscribe to specific message types
  subscribe(type: string, callback: (data: any) => void) {
    if (!this.listeners.has(type)) {
      this.listeners.set(type, new Set());
    }
    this.listeners.get(type)!.add(callback);

    // Return unsubscribe function
    return () => {
      const listeners = this.listeners.get(type);
      if (listeners) {
        listeners.delete(callback);
        if (listeners.size === 0) {
          this.listeners.delete(type);
        }
      }
    };
  }

  // Unsubscribe from specific message types
  unsubscribe(type: string, callback: (data: any) => void) {
    const listeners = this.listeners.get(type);
    if (listeners) {
      listeners.delete(callback);
      if (listeners.size === 0) {
        this.listeners.delete(type);
      }
    }
  }

  // Send message to server
  send(message: any) {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(message));
    } else {
      console.warn('WebSocket is not connected');
    }
  }

  // Get connection status
  getConnectionStatus(): 'connecting' | 'connected' | 'disconnected' | 'failed' {
    if (!this.ws) return 'disconnected';
    
    switch (this.ws.readyState) {
      case WebSocket.CONNECTING:
        return 'connecting';
      case WebSocket.OPEN:
        return 'connected';
      case WebSocket.CLOSING:
      case WebSocket.CLOSED:
        return 'disconnected';
      default:
        return 'failed';
    }
  }

  // Close connection
  close() {
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
  }

  // Reconnect manually
  reconnect() {
    this.close();
    this.reconnectAttempts = 0;
    this.connect();
  }
}

// Create and export a singleton instance
export const websocketService = new WebSocketService();
export default websocketService;
