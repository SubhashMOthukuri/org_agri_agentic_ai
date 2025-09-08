// Environment configuration
export const config = {
  apiBaseUrl: process.env.REACT_APP_API_BASE_URL || 'http://localhost:8000',
  wsUrl: process.env.REACT_APP_WS_URL || 'ws://localhost:8000/ws',
  mapboxToken: process.env.REACT_APP_MAPBOX_TOKEN || '',
  auth0Domain: process.env.REACT_APP_AUTH0_DOMAIN || '',
  auth0ClientId: process.env.REACT_APP_AUTH0_CLIENT_ID || '',
};
