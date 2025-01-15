/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ["index.html"],
  theme: {
    container: {
      center: true,
      padding: "16px",
    },
    extend: {
      colors: {
        primary: "#0ea5e9",
        dark: "#0f172a",
        secondary: "#64748b",
        tertierary: "#fdba74",
      },
      screens: {
        "2xl": "1280px",
      },
    },
  },
  plugins: [require("@tailwindcss/line-clamp")],
};
