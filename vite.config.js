import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    host: true,
    // Camera + microphone require a secure context. `vite --host` over plain http
    // only works on localhost; use `npm run dev -- --https` or a tunnel for phones.
  },
  build: {
    // onnxruntime-web ships large wasm chunks; silence the size warning.
    chunkSizeWarningLimit: 4000,
  },
  resolve: {
    // Opt into onnxruntime's "external wasm" export condition. Without it the
    // package resolves to a bundled build and Vite emits a 25 MB .wasm asset
    // that is never used, because faceEngine points wasmPaths at a CDN.
    conditions: ["onnxruntime-web-use-extern-wasm"],
  },
  optimizeDeps: {
    exclude: ["onnxruntime-web"],
  },
});
