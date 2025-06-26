/** @type {import('next').NextConfig} */
const nextConfig = {
  async rewrites() {
    return [
      {
        // Source: All paths starting with /accounts/
        source: '/accounts/:path*',
        // Destination: Forward to Django backend
        destination: 'http://127.0.0.1:8000/accounts/:path*',
      },
      {
        // Also proxy API, admin, and other Django-specific routes
        source: '/api/:path*',
        destination: 'http://127.0.0.1:8000/api/:path*',
      },
      {
        source: '/admin/:path*',
        destination: 'http://127.0.0.1:8000/admin/:path*',
      },
       {
        source: '/_header.html',
        destination: 'http://127.0.0.1:8000/_header.html',
      },
    ];
  },
};

export default nextConfig;
