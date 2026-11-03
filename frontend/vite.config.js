import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { defineConfig } from "vite";

const here = dirname(fileURLToPath(import.meta.url));

export default defineConfig({
  base: "/static/build/",
  server: {
    proxy: {
      "/api": { target: "https://localhost:8443", secure: false },
      "/ws": { target: "https://localhost:8443", secure: false, ws: true },
      "/static/vendor": { target: "https://localhost:8443", secure: false },
      "/static/ext": { target: "https://localhost:8443", secure: false },
      "/static/viewer-engine.js": { target: "https://localhost:8443", secure: false },
    },
  },
  build: {
    outDir: resolve(here, "../backend/web/build"),
    emptyOutDir: true,
    rollupOptions: {
      output: {
        manualChunks(id) {
          if (id.includes("/node_modules/three/") && !id.includes("/examples/")) return "three-vendor";
          if (id.includes("/examples/jsm/loaders/FBXLoader") || id.includes("/examples/jsm/libs/fflate")) return "fbx-loader";
        },
      },
      input: {
        main: resolve(here, "index.html"),
        camera: resolve(here, "cam.html"),
      },
    },
  },
});
