import { defineConfig } from 'vitest/config'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

// Tests never load vite.config.js: its plugin would build the real store and queue (and its
// recovery would mark the live Studio's runs interrupted). server/isolation.test.js keeps it so.
export default defineConfig({
  root: path.dirname(fileURLToPath(import.meta.url)),
  test: { include: ['**/*.test.js'], environment: 'node' },
})
