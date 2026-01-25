/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        'form-valid': '#22c55e',
        'form-invalid': '#ef4444',
        'form-warning': '#eab308',
      },
    },
  },
  plugins: [],
}
