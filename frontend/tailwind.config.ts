import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
    "./store/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
      },
      boxShadow: {
        brutal: '6px 6px 0px 0px rgba(0,0,0,1)',
        'brutal-sm': '3px 3px 0px 0px rgba(0,0,0,1)',
        'brutal-lg': '10px 10px 0px 0px rgba(0,0,0,1)',
      },
    },
  },
  plugins: [],
};

export default config;
