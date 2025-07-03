/** @type {import('next').NextConfig} */
const nextConfig = {
  async rewrites() {
    // Get the backend URL from environment variable with fallback
    const backendUrl = process.env.NEXT_PUBLIC_BACKEND_URL || 'http://127.0.0.1:8000';
    
    return [
      {
        // Source: All paths starting with /accounts/
        source: '/accounts/:path*',
        // Destination: Forward to Django backend
        destination: `${backendUrl}/accounts/:path*`,
      },
      {
        // Also proxy API, admin, and other Django-specific routes
        source: '/api/:path*',
        destination: `${backendUrl}/api/:path*`,
      },
      {
        source: '/admin/:path*',
        destination: `${backendUrl}/admin/:path*`,
      },
      {
        source: '/_header.html',
        destination: `${backendUrl}/_header.html`,
      },
      {
        source: '/knowledge/:path*',
        destination: `${backendUrl}/knowledge/:path*`,
      },
    ];
  },
};

export default nextConfig;
