import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { defineConfig } from "vite";

const here = dirname(fileURLToPath(import.meta.url));

export default defineConfig({
  base: "/static/build/",
  build: {
    outDir: resolve(here, "../backend/web/build"),
    emptyOutDir: true,
    rollupOptions: {
      input: {
        main: resolve(here, "index.html"),
        camera: resolve(here, "cam.html"),
      },
    },
  },
});
