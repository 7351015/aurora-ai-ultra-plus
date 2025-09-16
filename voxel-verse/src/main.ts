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

// Terrain generation
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
  }
}

class World {
  private chunks: Chunk[] = []
  private noise2D: (x: number, y: number) => number
  private chunkSize = 16
  private height = 48

  constructor() {
    this.noise2D = createNoise2D(() => 0.5)
  }

  buildAround(originX: number, originZ: number) {
    const radius = 2 // builds (2*radius+1)^2 chunks
    const sizeX = this.chunkSize
    const sizeZ = this.chunkSize
    const sizeY = this.height
    for (let cz = -radius; cz <= radius; cz++) {
      for (let cx = -radius; cx <= radius; cx++) {
        const worldX = Math.floor(originX / sizeX) * sizeX + cx * sizeX
        const worldZ = Math.floor(originZ / sizeZ) * sizeZ + cz * sizeZ
        const chunk = new Chunk({ sizeX, sizeY, sizeZ, worldX, worldZ }, this.noise2D)
        if (chunk.mesh) {
          chunk.mesh.position.set(worldX, 0, worldZ)
          scene.add(chunk.mesh)
        }
        this.chunks.push(chunk)
      }
    }
  }
}

const world = new World()
world.buildAround(0, 0)

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
