/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        'edge-primary': '#3b82f6',
        'edge-secondary': '#8b5cf6',
      }
    },
  },
  plugins: [],
}
