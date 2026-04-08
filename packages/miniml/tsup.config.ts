import { defineConfig } from 'tsup';

export default defineConfig({
  entry: ['src/index.ts', 'src/worker.ts'],
  format: ['esm'],
  dts: true,
  sourcemap: true,
  clean: true,
  minify: true,
  treeshake: true,
  external: ['./wasm/miniml_core.js'],
});
