import './style.css'
import * as THREE from 'three'
import { createNoise2D } from 'simplex-noise'

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

// Terrain generation & Voxel world
type Voxel = 0 | 1

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
        const h = Math.floor(8 + n * 24) // height 8..32
        for (let y = 0; y < sizeY; y++) {
          const isSolid: Voxel = y <= h ? 1 : 0
          this.voxels[this.index(x, y, z)] = isSolid
        }
      }
    }
  }

  private isSolid(x: number, y: number, z: number): boolean {
    const { sizeX, sizeY, sizeZ } = this.config
    if (x < 0 || y < 0 || z < 0 || x >= sizeX || y >= sizeY || z >= sizeZ) return false
    return this.voxels[this.index(x, y, z)] === 1
  }

  private buildMesh() {
    const positions: number[] = []
    const normals: number[] = []
    const uvs: number[] = []
    const indices: number[] = []

    const pushFace = (
      x: number, y: number, z: number,
      nx: number, ny: number, nz: number,
      corners: [number, number, number][]
    ) => {
      const baseIndex = positions.length / 3
      for (const [cx, cy, cz] of corners) {
        positions.push(x + cx, y + cy, z + cz)
        normals.push(nx, ny, nz)
        uvs.push(cx, cz)
      }
      indices.push(baseIndex, baseIndex + 1, baseIndex + 2, baseIndex, baseIndex + 2, baseIndex + 3)
    }

    const { sizeX, sizeY, sizeZ } = this.config
    for (let z = 0; z < sizeZ; z++) {
      for (let y = 0; y < sizeY; y++) {
        for (let x = 0; x < sizeX; x++) {
          if (!this.isSolid(x, y, z)) continue
          // -X
          if (!this.isSolid(x - 1, y, z))
            pushFace(x, y, z, -1, 0, 0, [ [0,0,1], [0,1,1], [0,1,0], [0,0,0] ])
          // +X
          if (!this.isSolid(x + 1, y, z))
            pushFace(x + 1, y, z, 1, 0, 0, [ [0,0,0], [0,1,0], [0,1,1], [0,0,1] ])
          // -Y
          if (!this.isSolid(x, y - 1, z))
            pushFace(x, y, z, 0, -1, 0, [ [0,0,0], [1,0,0], [1,0,1], [0,0,1] ])
          // +Y
          if (!this.isSolid(x, y + 1, z))
            pushFace(x, y + 1, z, 0, 1, 0, [ [0,0,1], [1,0,1], [1,0,0], [0,0,0] ])
          // -Z
          if (!this.isSolid(x, y, z - 1))
            pushFace(x, y, z, 0, 0, -1, [ [1,0,0], [1,1,0], [0,1,0], [0,0,0] ])
          // +Z
          if (!this.isSolid(x, y, z + 1))
            pushFace(x, y, z + 1, 0, 0, 1, [ [0,0,0], [0,1,0], [1,1,0], [1,0,0] ])
        }
      }
    }

    const geometry = new THREE.BufferGeometry()
    geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3))
    geometry.setAttribute('normal', new THREE.Float32BufferAttribute(normals, 3))
    geometry.setAttribute('uv', new THREE.Float32BufferAttribute(uvs, 2))
    geometry.setIndex(indices)
    geometry.computeBoundingSphere()

    const material = new THREE.MeshStandardMaterial({ color: 0x55aa55, flatShading: true })
    this.mesh = new THREE.Mesh(geometry, material)
    this.mesh.receiveShadow = false
    this.mesh.castShadow = false
    ;(this.mesh as any).userData.chunk = this
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

  constructor(scene: THREE.Scene) {
    this.scene = scene
    this.noise2D = createNoise2D(Math.random)
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
    if (chunk.mesh) {
      chunk.mesh.position.set(worldX, 0, worldZ)
      this.scene.add(chunk.mesh)
    }
    this.chunks.set(k, chunk)
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
  }

  public raycastFrom(_camera: THREE.Camera, raycaster: THREE.Raycaster): THREE.Intersection | null {
    // Collect chunk meshes only
    const meshes: THREE.Object3D[] = []
    for (const ch of this.chunks.values()) if (ch.mesh) meshes.push(ch.mesh)
    const hits = raycaster.intersectObjects(meshes, false)
    return hits.length > 0 ? hits[0] : null
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
  yaw -= movementX * 0.0025
  pitch -= movementY * 0.0025
  const maxPitch = Math.PI / 2 - 0.01
  pitch = Math.max(-maxPitch, Math.min(maxPitch, pitch))
})

document.addEventListener('keydown', (e) => {
  switch (e.code) {
    case 'KeyW': moveForward.state = true; break
    case 'KeyS': moveBackward.state = true; break
    case 'KeyA': moveLeft.state = true; break
    case 'KeyD': moveRight.state = true; break
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

// Selection highlight
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
    world.setVoxelAtWorld(wx, wy, wz, 1)
  }
}

document.addEventListener('mousedown', (e) => {
  if (!isLocked) return
  if (e.button === 0 || e.button === 2) interact(e.button)
})

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

  const speed = 20
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
  world.updateStreaming(camera.position, 2)
  updateSelection()
  updateCamera(dt)
  renderer.render(scene, camera)
  requestAnimationFrame(animate)
}
animate()

// Resize
window.addEventListener('resize', () => {
  camera.aspect = window.innerWidth / window.innerHeight
  camera.updateProjectionMatrix()
  renderer.setSize(window.innerWidth, window.innerHeight)
})
