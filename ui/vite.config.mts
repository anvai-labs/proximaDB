/*
 * Copyright 2025 ProximaDB
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// The ProximaDB server the dev server proxies to (CRA's former
// `"proxy": "http://localhost:5678"` — the multiplexed REST/gRPC/Flight port).
const API_TARGET = 'http://localhost:5678';

export default defineConfig({
  plugins: [react()],
  // Served from the origin root: the Monaco loader override in
  // SqlQueryTab.tsx is the absolute path `/vs`, and `public/vs` (the
  // self-hosted runtime, TD-UICDN-1) is copied verbatim to `build/vs`.
  base: '/',
  build: {
    // Same output directory CRA used, so `ui/.gitignore` and the TD-UICDN-1
    // verification (`build/vs/`) stay valid.
    outDir: 'build',
    // Monaco's runtime under public/vs is several MB of pre-built AMD code;
    // it is copied, not bundled, so the chunk-size warning only concerns the
    // app bundle itself.
    chunkSizeWarningLimit: 2048,
  },
  server: {
    port: 3000,
    // The only prefix the UI calls with a relative URL (`/api/v1/...` in
    // SqlQueryTab.tsx / GraphVisualizationTab.tsx); everything else is served
    // by Vite itself.
    proxy: {
      '/api': API_TARGET,
    },
  },
  preview: {
    port: 3000,
  },
});
