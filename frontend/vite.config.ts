import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

export default defineConfig({
  plugins: [react()],
  build: {
    // Split large vendor libs into their own chunks for better caching.
    rollupOptions: {
      output: {
        // Function form (object form is unsupported by Vite's rolldown bundler).
        // Order matters: match markdown deps before the bare react check, since
        // react-markdown also contains "react".
        manualChunks(id) {
          if (!id.includes("node_modules")) return undefined;
          if (/[\\/]node_modules[\\/](react-markdown|remark-gfm|rehype-highlight|highlight\.js)[\\/]/.test(id))
            return "markdown";
          if (id.includes("@tanstack/react-query")) return "query";
          if (/[\\/]node_modules[\\/](react|react-dom|scheduler)[\\/]/.test(id)) return "react";
          return undefined;
        },
      },
    },
    chunkSizeWarningLimit: 700,
  },
  server: {
    host: true,
    port: 5173,
    proxy: {
      "/api": { target: "http://localhost:8000", changeOrigin: true },
      "/health": { target: "http://localhost:8000", changeOrigin: true },
    },
  },
});
