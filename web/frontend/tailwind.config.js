/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        'bg-primary':    '#eef2ff',
        'text-primary':  '#0f172a',
        'text-secondary':'#1e40af',
        'text-muted':    '#64748b',
        'accent-blue':   '#2563eb',
        'accent-indigo': '#4f46e5',
        'accent-red':    '#dc2626',
        'accent-green':  '#16a34a',
        'accent-yellow': '#d97706',
        'accent-purple': '#7c3aed',
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
        mono: ['JetBrains Mono', 'monospace'],
      },
      animation: {
        'pulse-slow': 'pulse 3s cubic-bezier(0.4,0,0.6,1) infinite',
        'slide-up':   'slideUp 0.35s ease-out',
        'fade-in':    'fadeIn 0.25s ease-out',
      },
      keyframes: {
        slideUp: {
          from: { opacity: '0', transform: 'translateY(14px)' },
          to:   { opacity: '1', transform: 'translateY(0)' },
        },
        fadeIn: {
          from: { opacity: '0' },
          to:   { opacity: '1' },
        },
      },
    },
  },
  plugins: [],
}
