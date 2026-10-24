import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import { studioPlugin } from './server/plugin.js'

const here = path.dirname(fileURLToPath(import.meta.url))

// Store, queue and Ctrl-C handling are created by the plugin inside the running dev server only
// (server/plugin.js); tests use vitest.config.js and never load this file.
export default defineConfig({
  root: here,
  plugins: [react(), studioPlugin()],
  server: {
    port: 5199,
    strictPort: true,
    host: 'localhost',
    watch: { ignored: ['**/.venv/**', '**/models/**'] },
  },
})
