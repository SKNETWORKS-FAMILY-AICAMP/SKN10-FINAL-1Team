/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    './pages/**/*.{js,ts,jsx,tsx,mdx}',
    './components/**/*.{js,ts,jsx,tsx,mdx}',
    './app/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      colors: {
        primary: '#2563eb',
        secondary: '#3b82f6',
        dark: '#0f172a',
        light: '#f8fafc'
      }
    },
  },
  plugins: [
    require('@tailwindcss/typography'),
  ],
} 