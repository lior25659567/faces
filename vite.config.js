import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { framesApi } from './pipeline/frames-api.js'

export default defineConfig({
  plugins: [react(), framesApi()],
  // pipeline/ holds the Python env and model weights (about 1.8 GB); keep the dev server and tests out of it.
  server: { watch: { ignored: ['**/pipeline/.venv/**', '**/pipeline/vendor/**', '**/input/**'] } },
  test: { include: ['src/**/*.test.js', 'pipeline/*.test.js'] },
})
