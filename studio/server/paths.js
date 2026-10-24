import path from 'node:path'
import { fileURLToPath } from 'node:url'

export const REPO = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..', '..')
// STUDIO_DATA points a test (or a throwaway Studio) at a temp copy, never at the user's data.
export const DATA = process.env.STUDIO_DATA ? path.resolve(process.env.STUDIO_DATA) : path.join(REPO, '3d-character', 'studio')
export const PYTHON = path.join(REPO, 'studio', '.venv', 'bin', 'python')
export const BLENDER = '/Applications/Blender.app/Contents/MacOS/Blender'
export const SKIN = path.join(REPO, '3d-character', '3d cartoon boy.glb')
export const CLIPS_DIR = path.join(REPO, '3d-character', 'mixamo')
