import { createAutomation } from './automation.js'
import { studioApi } from './api.js'
import { DATA } from './paths.js'
import { createQueue, installShutdown } from './queue.js'
import { createStore } from './store.js'
import { createWork, createWorkWatcher } from './work.js'

// The Studio's Vite plugin. The real store and queue exist only inside a dev server (never at build
// time, never in vitest), and they touch runs on disk only once that server is really listening:
// a second `vite` that loses the port to strictPort must not mark the live Studio's runs interrupted.
export const studioPlugin = ({ dataRoot = DATA, onListening = installShutdown } = {}) => ({
  name: 'exercise-studio',
  apply: 'serve',
  configureServer(server) {
    const store = createStore(dataRoot)
    const queue = createQueue({ store })
    const automation = createAutomation({ store, queue })
    const work = createWork(dataRoot)
    const watcher = createWorkWatcher({ work, store, queue })
    studioApi({ store, queue, work, watcher, automation, onListening: () => onListening(queue) }).configureServer(server)
  },
})
