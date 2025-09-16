import './style.css'
import * as THREE from 'three'
import { createNoise2D } from 'simplex-noise'
// @ts-ignore - web worker loader via Vite
import MesherWorkerUrl from './mesher.worker.ts?worker&url'

const appRoot = document.querySelector<HTMLDivElement>('#app')!

// Inject overrides to ensure full-screen canvas/HUD regardless of default template styles
const globalStyle = document.createElement('style')
globalStyle.textContent = `
  html, body, #app { margin: 0; padding: 0; width: 100%; height: 100%; overflow: hidden; background: #87ceeb; }
  #app { display: block; max-width: none; padding: 0; }
  canvas { display: block; }
  #hud { position: fixed; top: 0; left: 0; width: 100%; height: 100%; pointer-events: none; }
  #crosshair { position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); color: rgba(255,255,255,0.85); font-size: 24px; user-select: none; }
  #hint { position: absolute; bottom: 20px; left: 50%; transform: translateX(-50%); background: rgba(0,0,0,0.5); color: #fff; padding: 8px 12px; border-radius: 6px; font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial; }
`
document.head.appendChild(globalStyle)

// Renderer
const renderer = new THREE.WebGLRenderer({ antialias: true })
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
renderer.setSize(window.innerWidth, window.innerHeight)
renderer.outputColorSpace = THREE.SRGBColorSpace
appRoot.appendChild(renderer.domElement)

// Scene and Camera
const scene = new THREE.Scene()
scene.background = new THREE.Color(0x87ceeb)
scene.fog = new THREE.Fog(0x87ceeb, 60, 300)

const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000)
camera.position.set(0, 20, 0)

// Lighting
const sun = new THREE.DirectionalLight(0xffffff, 2.0)
sun.position.set(100, 200, 100)
sun.castShadow = false
scene.add(sun)
scene.add(new THREE.AmbientLight(0xffffff, 0.4))

// Day/Night cycle & water
let worldTime = 0
const ambientLight = new THREE.AmbientLight(0xffffff, 0.4)
scene.add(ambientLight)
const waterLevel = 12
const water = new THREE.Mesh(
  new THREE.PlaneGeometry(2048, 2048, 1, 1),
  new THREE.MeshPhongMaterial({ color: 0x3366cc, transparent: true, opacity: 0.55, shininess: 60 })
)
water.rotation.x = -Math.PI / 2
water.position.y = waterLevel + 0.01
water.receiveShadow = false
water.renderOrder = -1
scene.add(water)

// Terrain generation & Voxel world
type Voxel = 0 | 1 | 2 | 3 | 4 | 5

const BLOCK = {
  Air: 0 as Voxel,
  Grass: 1 as Voxel,
  Dirt: 2 as Voxel,
  Stone: 3 as Voxel,
  Log: 4 as Voxel,
  Sand: 5 as Voxel,
}

const blockColors: Record<number, number> = {
  [BLOCK.Air]: 0x000000,
  [BLOCK.Grass]: 0x55aa55,
  [BLOCK.Dirt]: 0x8b5a2b,
  [BLOCK.Stone]: 0x777777,
  [BLOCK.Log]: 0xa07040,
  [BLOCK.Sand]: 0xd8c17a,
}
let selectedBlock: Voxel = BLOCK.Grass

interface ChunkConfig {
  sizeX: number
  sizeY: number
  sizeZ: number
  worldX: number
  worldZ: number
}

class Chunk {
  public mesh: THREE.Mesh | null = null
  private voxels: Uint8Array
  private config: ChunkConfig
  private worker: Worker | null = null

  constructor(config: ChunkConfig, noise2D: (x: number, y: number) => number) {
    this.config = config
    this.voxels = new Uint8Array(config.sizeX * config.sizeY * config.sizeZ)
    this.populate(noise2D)
    this.buildMesh()
  }

  private index(x: number, y: number, z: number): number {
    const { sizeX, sizeY } = this.config
    return x + y * sizeX + z * sizeX * sizeY
  }

  private populate(noise2D: (x: number, y: number) => number) {
    const { sizeX, sizeY, sizeZ, worldX, worldZ } = this.config
    for (let z = 0; z < sizeZ; z++) {
      for (let x = 0; x < sizeX; x++) {
        const wx = worldX + x
        const wz = worldZ + z
        const n = (noise2D(wx / 64, wz / 64) + 1) * 0.5
        const e = (noise2D(wx / 128, wz / 128) + 1) * 0.5
        const h = Math.floor(8 + n * 24 + e * 8) // varied terrain
        const moisture = (noise2D((wx+1000) / 200, (wz+1000) / 200) + 1) * 0.5
        // 0: plains, 1: desert, 2: forest
        let biome = 0
        if (moisture < 0.3) biome = 1
        else if (moisture > 0.65) biome = 2
        for (let y = 0; y < sizeY; y++) {
          let voxelType: Voxel = BLOCK.Air
          if (y <= h) {
            if (y === h) {
              if (biome === 1 || h <= waterLevel + 1) voxelType = BLOCK.Sand
              else voxelType = BLOCK.Grass
            } else if (y < h - 3) voxelType = BLOCK.Stone
            else voxelType = BLOCK.Dirt
          }
          this.voxels[this.index(x, y, z)] = voxelType
        }
      }
    }
  }

  // removed unused isSolid to satisfy TS strict

  private buildMesh() {
    // Offload meshing to worker using greedy mesher
    if (!this.worker) this.worker = new Worker(MesherWorkerUrl, { type: 'module' })
    const { sizeX, sizeY, sizeZ } = this.config
    const message = { sizeX, sizeY, sizeZ, voxels: this.voxels.buffer.slice(0), blockColors: Object.values(blockColors) }
    this.worker.onmessage = (ev: MessageEvent<any>) => {
      const { positions, normals, colors, uvs, indices } = ev.data
      const geometry = new THREE.BufferGeometry()
      geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3))
      geometry.setAttribute('normal', new THREE.Float32BufferAttribute(normals, 3))
      geometry.setAttribute('uv', new THREE.Float32BufferAttribute(uvs, 2))
      geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3))
      geometry.setIndex(new THREE.BufferAttribute(indices, 1))
      geometry.computeBoundingSphere()
      const material = new THREE.MeshStandardMaterial({ vertexColors: true, flatShading: true })
      const newMesh = new THREE.Mesh(geometry, material)
      newMesh.receiveShadow = false
      newMesh.castShadow = false
      ;(newMesh as any).userData.chunk = this
      if (this.mesh && this.mesh.parent) this.mesh.parent.remove(this.mesh)
      this.mesh = newMesh
      if (this.mesh && (this.mesh as any).position) {
        // position will be set by world after rebuild
      }
    }
    ;(this.worker as any).postMessage(message, [message.voxels])
  }

  public setVoxel(localX: number, localY: number, localZ: number, value: Voxel): void {
    const { sizeX, sizeY, sizeZ } = this.config
    if (localX < 0 || localY < 0 || localZ < 0 || localX >= sizeX || localY >= sizeY || localZ >= sizeZ) return
    this.voxels[this.index(localX, localY, localZ)] = value
  }

  public getVoxel(localX: number, localY: number, localZ: number): Voxel {
    const { sizeX, sizeY, sizeZ } = this.config
    if (localX < 0 || localY < 0 || localZ < 0 || localX >= sizeX || localY >= sizeY || localZ >= sizeZ) return 0
    return this.voxels[this.index(localX, localY, localZ)] as Voxel
  }

  public rebuildMesh(): void {
    if (this.mesh) {
      if (this.mesh.parent) this.mesh.parent.remove(this.mesh)
      this.mesh.geometry.dispose()
    }
    this.buildMesh()
  }
}

class World {
  public readonly chunkSize = 16
  public readonly height = 64
  private readonly noise2D: (x: number, y: number) => number
  private readonly scene: THREE.Scene
  private readonly chunks = new Map<string, Chunk>()
  private readonly active = new Set<string>()
  private readonly edits: { x: number, y: number, z: number, v: Voxel }[] = []

  constructor(scene: THREE.Scene) {
    this.scene = scene
    this.noise2D = createNoise2D(Math.random)
    this.loadEdits()
  }

  private key(cx: number, cz: number): string { return `${cx},${cz}` }
  private toChunkCoord(worldCoord: number): number { return Math.floor(worldCoord / this.chunkSize) }
  private positiveMod(n: number, mod: number): number { return ((n % mod) + mod) % mod }

  private loadChunk(cx: number, cz: number): void {
    const k = this.key(cx, cz)
    if (this.chunks.has(k)) return
    const sizeX = this.chunkSize
    const sizeZ = this.chunkSize
    const sizeY = this.height
    const worldX = cx * sizeX
    const worldZ = cz * sizeZ
    const chunk = new Chunk({ sizeX, sizeY, sizeZ, worldX, worldZ }, this.noise2D)
    // Apply any saved edits for this chunk
    for (const e of this.edits) {
      if (
        e.x >= worldX && e.x < worldX + sizeX &&
        e.z >= worldZ && e.z < worldZ + sizeZ &&
        e.y >= 0 && e.y < sizeY
      ) {
        const lx = e.x - worldX
        const lz = e.z - worldZ
        chunk.setVoxel(lx, e.y, lz, e.v)
      }
    }
    chunk.rebuildMesh()
    if (chunk.mesh) {
      chunk.mesh.position.set(worldX, 0, worldZ)
      this.scene.add(chunk.mesh)
    }
    this.chunks.set(k, chunk)

    // Procedural trees: sparse distribution in forest biomes
    const moisture = (this.noise2D((worldX+1000) / 200, (worldZ+1000) / 200) + 1) * 0.5
    const isForest = moisture > 0.65
    if (isForest) {
      for (let i = 0; i < 3; i++) {
        const tx = worldX + Math.floor(Math.random() * sizeX)
        const tz = worldZ + Math.floor(Math.random() * sizeZ)
        // find ground height by scanning up
        for (let y = sizeY-2; y >= 1; y--) {
          const { chunk: ch, localX, localZ } = this.getChunkAtWorld(tx, tz)
          if (!ch) break
          const vBelow = ch.getVoxel(localX, y, localZ)
          const vAbove = ch.getVoxel(localX, y+1, localZ)
          if (vBelow !== BLOCK.Air && vAbove === BLOCK.Air) {
            this.plantTree(tx, y+1, tz)
            break
          }
        }
      }
    }
  }

  private unloadChunk(cx: number, cz: number): void {
    const k = this.key(cx, cz)
    const chunk = this.chunks.get(k)
    if (!chunk) return
    if (chunk.mesh) {
      this.scene.remove(chunk.mesh)
      chunk.mesh.geometry.dispose()
      ;(chunk.mesh.material as THREE.Material).dispose()
    }
    this.chunks.delete(k)
  }

  public updateStreaming(cameraPosition: THREE.Vector3, radius: number = 2): void {
    const centerCX = this.toChunkCoord(cameraPosition.x)
    const centerCZ = this.toChunkCoord(cameraPosition.z)
    const needed = new Set<string>()
    for (let dz = -radius; dz <= radius; dz++) {
      for (let dx = -radius; dx <= radius; dx++) {
        const cx = centerCX + dx
        const cz = centerCZ + dz
        const k = this.key(cx, cz)
        needed.add(k)
        if (!this.chunks.has(k)) this.loadChunk(cx, cz)
      }
    }
    // Unload not-needed
    for (const k of this.chunks.keys()) {
      if (!needed.has(k)) {
        const [sx, sz] = k.split(',')
        this.unloadChunk(parseInt(sx), parseInt(sz))
      }
    }
    this.active.clear()
    for (const k of needed) this.active.add(k)
  }

  public plantTree(worldX: number, baseY: number, worldZ: number): void {
    const height = 4 + Math.floor(Math.random() * 3)
    for (let i = 0; i < height; i++) this.setVoxelAtWorld(worldX, baseY + i, worldZ, BLOCK.Log)
    const topY = baseY + height
    for (let dz = -2; dz <= 2; dz++) {
      for (let dx = -2; dx <= 2; dx++) {
        const dist = Math.abs(dx) + Math.abs(dz)
        if (dist <= 3) this.setVoxelAtWorld(worldX + dx, topY, worldZ + dz, BLOCK.Grass)
      }
    }
  }

  public getChunkAtWorld(worldX: number, worldZ: number): { chunk: Chunk | null, localX: number, localZ: number, cx: number, cz: number } {
    const cx = this.toChunkCoord(worldX)
    const cz = this.toChunkCoord(worldZ)
    const k = this.key(cx, cz)
    const chunk = this.chunks.get(k) ?? null
    const localX = this.positiveMod(Math.floor(worldX - cx * this.chunkSize), this.chunkSize)
    const localZ = this.positiveMod(Math.floor(worldZ - cz * this.chunkSize), this.chunkSize)
    return { chunk, localX, localZ, cx, cz }
  }

  public setVoxelAtWorld(worldX: number, worldY: number, worldZ: number, value: Voxel): void {
    const { chunk, localX, localZ, cx, cz } = this.getChunkAtWorld(worldX, worldZ)
    if (!chunk) { this.loadChunk(cx, cz); return this.setVoxelAtWorld(worldX, worldY, worldZ, value) }
    if (worldY < 0 || worldY >= this.height) return
    chunk.setVoxel(localX, Math.floor(worldY), localZ, value)
    chunk.rebuildMesh()
    if (chunk.mesh) {
      chunk.mesh.position.set(cx * this.chunkSize, 0, cz * this.chunkSize)
      this.scene.add(chunk.mesh)
    }
    this.recordEdit(worldX, Math.floor(worldY), worldZ, value)
  }

  public raycastFrom(_camera: THREE.Camera, raycaster: THREE.Raycaster): THREE.Intersection | null {
    // Collect chunk meshes only
    const meshes: THREE.Object3D[] = []
    for (const ch of this.chunks.values()) if (ch.mesh) meshes.push(ch.mesh)
    const hits = raycaster.intersectObjects(meshes, false)
    return hits.length > 0 ? hits[0] : null
  }

  private saveEdits(): void {
    try {
      localStorage.setItem('voxel_edits', JSON.stringify(this.edits))
    } catch {}
  }

  private loadEdits(): void {
    try {
      const raw = localStorage.getItem('voxel_edits')
      if (raw) {
        const arr = JSON.parse(raw) as { x: number, y: number, z: number, v: Voxel }[]
        this.edits.splice(0, this.edits.length, ...arr)
      }
    } catch {}
  }

  private recordEdit(x: number, y: number, z: number, v: Voxel): void {
    // Deduplicate by same coord
    const idx = this.edits.findIndex(e => e.x === x && e.y === y && e.z === z)
    if (idx >= 0) this.edits[idx].v = v
    else this.edits.push({ x, y, z, v })
    this.saveEdits()
  }
}

const world = new World(scene)
world.updateStreaming(camera.position)

// Simple FPS controls with Pointer Lock
let isLocked = false
let yaw = 0
let pitch = 0
const velocity = new THREE.Vector3()
const direction = new THREE.Vector3()
const moveForward = { state: false }
const moveBackward = { state: false }
const moveLeft = { state: false }
const moveRight = { state: false }

const canvas = renderer.domElement
canvas.addEventListener('click', () => {
  if (!isLocked) canvas.requestPointerLock()
})

document.addEventListener('pointerlockchange', () => {
  isLocked = document.pointerLockElement === canvas
  const hint = document.getElementById('hint')
  if (hint) hint.style.display = isLocked ? 'none' : 'block'
})

document.addEventListener('mousemove', (event) => {
  if (!isLocked) return
  const movementX = event.movementX || 0
  const movementY = event.movementY || 0
  yaw -= movementX * (0.003 * mouseSensitivity)
  pitch -= movementY * (0.003 * mouseSensitivity)
  const maxPitch = Math.PI / 2 - 0.01
  pitch = Math.max(-maxPitch, Math.min(maxPitch, pitch))
})

document.addEventListener('keydown', (e) => {
  switch (e.code) {
    case 'KeyW': moveForward.state = true; break
    case 'KeyS': moveBackward.state = true; break
    case 'KeyA': moveLeft.state = true; break
    case 'KeyD': moveRight.state = true; break
    case 'Digit1': selectedBlock = BLOCK.Grass; updateHotbar(); break
    case 'Digit2': selectedBlock = BLOCK.Dirt; updateHotbar(); break
    case 'Digit3': selectedBlock = BLOCK.Stone; updateHotbar(); break
    case 'Digit4': selectedBlock = BLOCK.Log; updateHotbar(); break
    case 'Digit5': selectedBlock = BLOCK.Sand; updateHotbar(); break
  }
})

document.addEventListener('keyup', (e) => {
  switch (e.code) {
    case 'KeyW': moveForward.state = false; break
    case 'KeyS': moveBackward.state = false; break
    case 'KeyA': moveLeft.state = false; break
    case 'KeyD': moveRight.state = false; break
  }
})

// Prevent context menu on right-click for placement
document.addEventListener('contextmenu', (e) => { if (isLocked) e.preventDefault() })

// Selection highlight & hotbar
const selectionMaterial = new THREE.MeshBasicMaterial({ color: 0xffffff, wireframe: true, transparent: true, opacity: 0.8 })
const selectionMesh = new THREE.Mesh(new THREE.BoxGeometry(1.01, 1.01, 1.01), selectionMaterial)
selectionMesh.visible = false
scene.add(selectionMesh)

const raycaster = new THREE.Raycaster()
raycaster.far = 8

function updateSelection(): void {
  raycaster.setFromCamera(new THREE.Vector2(0, 0), camera)
  const hit = world.raycastFrom(camera, raycaster)
  if (!hit || !hit.face || !(hit.object as any).userData.chunk) { selectionMesh.visible = false; return }
  // access chunk to avoid tree-shaken unused warning in dev tooling
  if ((hit.object as any).userData.chunk) {
    // no-op
  }
  const meshPos = hit.object.position as THREE.Vector3
  const faceNormal = hit.face.normal.clone().applyMatrix3(new THREE.Matrix3().getNormalMatrix((hit.object as THREE.Object3D).matrixWorld))
  // Remove target (inside), place target (outside)
  const localPoint = hit.point.clone().sub(meshPos)
  const removePoint = localPoint.clone().addScaledVector(faceNormal, -0.01)
  const rx = Math.floor(removePoint.x)
  const ry = Math.floor(removePoint.y)
  const rz = Math.floor(removePoint.z)
  selectionMesh.position.set(Math.floor(meshPos.x) + rx + 0.5, ry + 0.5, Math.floor(meshPos.z) + rz + 0.5)
  selectionMesh.visible = ry >= 0 && ry < world.height
}

function interact(button: number): void {
  raycaster.setFromCamera(new THREE.Vector2(0, 0), camera)
  const hit = world.raycastFrom(camera, raycaster)
  if (!hit || !hit.face || !(hit.object as any).userData.chunk) return
  const meshPos = hit.object.position as THREE.Vector3
  const faceNormal = hit.face.normal.clone().applyMatrix3(new THREE.Matrix3().getNormalMatrix((hit.object as THREE.Object3D).matrixWorld)).normalize()
  const localPoint = hit.point.clone().sub(meshPos)
  if (button === 0) {
    // Remove block
    const p = localPoint.clone().addScaledVector(faceNormal, -0.01)
    const wx = Math.floor(meshPos.x) + Math.floor(p.x)
    const wy = Math.floor(p.y)
    const wz = Math.floor(meshPos.z) + Math.floor(p.z)
    world.setVoxelAtWorld(wx, wy, wz, 0)
  } else if (button === 2) {
    // Place block adjacent
    const p = localPoint.clone().addScaledVector(faceNormal, 0.51)
    const wx = Math.floor(meshPos.x) + Math.floor(p.x)
    const wy = Math.floor(p.y)
    const wz = Math.floor(meshPos.z) + Math.floor(p.z)
    world.setVoxelAtWorld(wx, wy, wz, selectedBlock)
  }
}

document.addEventListener('mousedown', (e) => {
  if (!isLocked) return
  if (modeCombat) {
    if (e.button === 0) shootWithAudio()
  } else {
    if (e.button === 0 || e.button === 2) interactWithAudio(e.button)
  }
})

// Combat: hitscan shooting and simple enemies
const enemies: THREE.Mesh[] = []
function spawnEnemy(x: number, y: number, z: number) {
  const m = new THREE.Mesh(new THREE.BoxGeometry(0.8, 1.6, 0.8), new THREE.MeshStandardMaterial({ color: 0xcc4444 }))
  m.position.set(x, y, z)
  scene.add(m)
  enemies.push(m)
}
for (let i = 0; i < 5; i++) spawnEnemy((Math.random() - 0.5) * 40, 20, (Math.random() - 0.5) * 40)

const muzzleFlashMat = new THREE.MeshBasicMaterial({ color: 0xffffaa })
const muzzleFlash = new THREE.Mesh(new THREE.SphereGeometry(0.06), muzzleFlashMat)
muzzleFlash.visible = false
scene.add(muzzleFlash)

function shoot() {
  muzzleFlash.position.copy(camera.position)
  muzzleFlash.visible = true
  setTimeout(() => (muzzleFlash.visible = false), 50)
  const shootRay = new THREE.Raycaster()
  shootRay.setFromCamera(new THREE.Vector2(0, 0), camera)
  // Hit enemies first
  const enemyHits = shootRay.intersectObjects(enemies, false)
  if (enemyHits.length > 0) {
    const h = enemyHits[0]
    const obj = h.object
    obj.parent?.remove(obj)
    const idx = enemies.indexOf(obj as THREE.Mesh)
    if (idx >= 0) enemies.splice(idx, 1)
    return
  }
  // Otherwise interact with world (remove block)
  const hit = world.raycastFrom(camera, shootRay)
  if (hit && hit.face && (hit.object as any).userData.chunk) {
    const meshPos = hit.object.position as THREE.Vector3
    const faceNormal = hit.face.normal.clone().applyMatrix3(new THREE.Matrix3().getNormalMatrix((hit.object as THREE.Object3D).matrixWorld)).normalize()
    const localPoint = hit.point.clone().sub(meshPos)
    const p = localPoint.clone().addScaledVector(faceNormal, -0.01)
    const wx = Math.floor(meshPos.x) + Math.floor(p.x)
    const wy = Math.floor(p.y)
    const wz = Math.floor(meshPos.z) + Math.floor(p.z)
    world.setVoxelAtWorld(wx, wy, wz, 0)
  }
}

function updateCamera(dt: number) {
  // Set rotation from yaw/pitch
  const quaternion = new THREE.Quaternion()
  quaternion.setFromEuler(new THREE.Euler(pitch, yaw, 0, 'YXZ'))
  camera.quaternion.copy(quaternion)

  direction.set(0, 0, 0)
  if (moveForward.state) direction.z -= 1
  if (moveBackward.state) direction.z += 1
  if (moveLeft.state) direction.x -= 1
  if (moveRight.state) direction.x += 1
  if (direction.lengthSq() > 0) direction.normalize()

  // Transform direction into world space using camera yaw only
  const yawQuat = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 1, 0), yaw)
  direction.applyQuaternion(yawQuat)

  const speed = moveSpeed
  velocity.x = direction.x * speed
  velocity.z = direction.z * speed

  camera.position.x += velocity.x * dt
  camera.position.z += velocity.z * dt

  // Simple ground following via raycast downwards
  const raycaster = new THREE.Raycaster(new THREE.Vector3(camera.position.x, 200, camera.position.z), new THREE.Vector3(0, -1, 0))
  const intersects = raycaster.intersectObjects(scene.children, false)
  if (intersects.length > 0) {
    const hit = intersects[0]
    camera.position.y = hit.point.y + 1.8
  }
}

// Animate
let last = performance.now()
function animate() {
  const now = performance.now()
  const dt = Math.min(0.05, (now - last) / 1000)
  last = now
  // Day-night progression
  worldTime += dt * 0.05
  const dayPhase = (Math.sin(worldTime) + 1) * 0.5
  ambientLight.intensity = 0.2 + dayPhase * 0.6
  sun.intensity = 1.0 + dayPhase
  const angle = worldTime * 0.5
  sun.position.set(Math.cos(angle) * 200, 100 + Math.sin(angle) * 150, Math.sin(angle) * 200)
  water.material.opacity = 0.45 + 0.1 * Math.sin(worldTime * 1.5)
  world.updateStreaming(camera.position, renderDistance)
  updateSelection()
  updateCamera(dt)
  drawMinimap()
  renderer.render(scene, camera)
  const frameDelay = Math.max(0, (1000 / fpsCap) - (performance.now() - now))
  setTimeout(() => requestAnimationFrame(animate), frameDelay)
}
animate()

// Resize
window.addEventListener('resize', () => {
  camera.aspect = window.innerWidth / window.innerHeight
  camera.updateProjectionMatrix()
  renderer.setSize(window.innerWidth, window.innerHeight)
})

// Minimal audio cues
let audioCtx: AudioContext | null = null
function ensureAudio() { if (!audioCtx) audioCtx = new (window.AudioContext || (window as any).webkitAudioContext)() }
function playClick() {
  ensureAudio(); if (!audioCtx) return
  const o = audioCtx.createOscillator(); const g = audioCtx.createGain()
  o.frequency.value = 660; g.gain.value = 0.1
  o.connect(g); g.connect(audioCtx.destination)
  o.start(); setTimeout(() => { o.stop(); g.disconnect() }, 80)
}
function playShoot() {
  ensureAudio(); if (!audioCtx) return
  const o = audioCtx.createOscillator(); const g = audioCtx.createGain()
  o.type = 'square'; o.frequency.value = 220; g.gain.value = 0.08
  o.connect(g); g.connect(audioCtx.destination)
  o.start(); setTimeout(() => { o.stop(); g.disconnect() }, 60)
}

// Hook audio into interactions
const interactWithAudio = (button: number) => { playClick(); interact(button) }
const shootWithAudio = () => { playShoot(); shoot() }

// Hotbar UI
const hotbar = document.getElementById('hotbar')
function updateHotbar() {
  if (!hotbar) return
  for (const el of Array.from(hotbar.querySelectorAll('.slot'))) {
    const type = parseInt((el as HTMLElement).dataset.type || '0')
    ;(el as HTMLElement).style.borderColor = type === selectedBlock ? '#fff' : '#888'
  }
}
updateHotbar()

// Settings and modes
let modeCombat = false
const modeEl = document.getElementById('mode') as HTMLElement | null
const settingsEl = document.getElementById('settings') as HTMLElement | null
const sensitivityEl = document.getElementById('sensitivity') as HTMLInputElement | null
const moveSpeedEl = document.getElementById('movespeed') as HTMLInputElement | null
const renderDistEl = document.getElementById('renderdist') as HTMLInputElement | null
const fpsCapEl = document.getElementById('fpscap') as HTMLInputElement | null
let mouseSensitivity = sensitivityEl ? parseFloat(sensitivityEl.value) : 0.5
let moveSpeed = moveSpeedEl ? parseFloat(moveSpeedEl.value) : 20
let renderDistance = renderDistEl ? parseInt(renderDistEl.value) : 2
let fpsCap = fpsCapEl ? parseInt(fpsCapEl.value) : 60

document.addEventListener('keydown', (e) => {
  if (e.code === 'KeyB') { modeCombat = false; if (modeEl) modeEl.textContent = 'Mode: Build' }
  if (e.code === 'KeyF') { modeCombat = true; if (modeEl) modeEl.textContent = 'Mode: Combat' }
  if (e.code === 'KeyP') { if (settingsEl) settingsEl.style.display = settingsEl.style.display === 'none' ? 'block' : 'none' }
})

if (sensitivityEl) sensitivityEl.addEventListener('input', () => { mouseSensitivity = parseFloat(sensitivityEl!.value) })
if (moveSpeedEl) moveSpeedEl.addEventListener('input', () => { moveSpeed = parseFloat(moveSpeedEl!.value) })
if (renderDistEl) renderDistEl.addEventListener('input', () => { renderDistance = parseInt(renderDistEl!.value) })
if (fpsCapEl) fpsCapEl.addEventListener('input', () => { fpsCap = parseInt(fpsCapEl!.value) })

// Minimap setup
const minimap = document.getElementById('minimap') as HTMLCanvasElement | null
const mmCtx = minimap ? minimap.getContext('2d') : null
function drawMinimap() {
  if (!minimap || !mmCtx) return
  const size = minimap.width
  mmCtx.clearRect(0, 0, size, size)
  mmCtx.fillStyle = 'rgba(0,0,0,0.2)'
  mmCtx.fillRect(0, 0, size, size)
  // Simple top-down dots for enemies and player
  const scale = 2
  const cx = size / 2
  const cz = size / 2
  // Player
  mmCtx.fillStyle = '#ffffff'
  mmCtx.fillRect(cx - 2, cz - 2, 4, 4)
  // Enemies
  mmCtx.fillStyle = '#ff5555'
  for (const e of enemies) {
    const dx = (e.position.x - camera.position.x) / scale
    const dz = (e.position.z - camera.position.z) / scale
    const x = cx + dx
    const z = cz + dz
    if (x >= 0 && x < size && z >= 0 && z < size) mmCtx.fillRect(x - 2, z - 2, 4, 4)
  }
}
