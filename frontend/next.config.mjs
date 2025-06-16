/** @type {import('next').NextConfig} */
const nextConfig = {
  eslint: {
    ignoreDuringBuilds: true,
  },
  typescript: {
    ignoreBuildErrors: true,
  },
  images: {
    unoptimized: true,
  },
  // API 요청을 Django 백엔드와 FastAPI 백엔드로 프록시
  async rewrites() {
    return [
      // CSV 업로드와 테이블 관리, 예측 관련 API는 FastAPI 서버로 라우팅
      {
        source: '/api/upload-csv',
        destination: 'http://localhost:8001/api/upload-csv',
      },
      {
        source: '/api/tables',
        destination: 'http://localhost:8001/api/tables',
      },
      {
        source: '/api/predict',
        destination: 'http://localhost:8001/api/predict',
      },
      // 그 외 API 요청은 Django 백엔드로 라우팅
      {
        source: '/api/:path*',
        destination: 'http://localhost:8000/api/:path*',
      },
    ];
  },
}

export default nextConfig
