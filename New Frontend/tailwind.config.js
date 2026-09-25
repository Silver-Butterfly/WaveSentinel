/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        void: '#020617',
        'dark-matter': '#0f172a',
        stardust: '#94a3b8',
        'navy-blue': '#1e293b',
        'electric-blue': '#00f0ff',
        'neon-teal': '#00ffcc',
      },
      fontFamily: {
        heading: ['"Times New Roman"', 'Times', 'serif'],
        body: ['Arial', 'Calibri', 'sans-serif'],
        mono: ['Arial', 'Calibri', 'sans-serif'],
      },
      animation: {
        float: 'float 8s ease-in-out infinite',
      },
      keyframes: {
        float: {
          '0%, 100%': { transform: 'translateY(0px)' },
          '50%': { transform: 'translateY(-20px)' },
        }
      }
    },
  },
  plugins: [],
}
