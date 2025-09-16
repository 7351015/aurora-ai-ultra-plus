// Lightweight greedy mesher running off the main thread

export interface MesherRequest {
	 sizeX: number
	 sizeY: number
	 sizeZ: number
	 voxels: ArrayBuffer
	 blockColors: number[]
}

export interface MesherResponse {
	 positions: Float32Array
	 normals: Float32Array
	 colors: Float32Array
	 uvs: Float32Array
	 indices: Uint32Array
}

// const Voxel type removed as unused

const DIRS = [
	{ d: [ 1, 0, 0], u: [0, -1, 0], v: [0, 0, 1], normal: [ 1, 0, 0] }, // +X
	{ d: [-1, 0, 0], u: [0, -1, 0], v: [0, 0, 1], normal: [-1, 0, 0] }, // -X
	{ d: [ 0, 1, 0], u: [0, 0, 1], v: [1, 0, 0], normal: [ 0, 1, 0] }, // +Y
	{ d: [ 0,-1, 0], u: [0, 0, 1], v: [1, 0, 0], normal: [ 0,-1, 0] }, // -Y
	{ d: [ 0, 0, 1], u: [0, 1, 0], v: [1, 0, 0], normal: [ 0, 0, 1] }, // +Z
	{ d: [ 0, 0,-1], u: [0, 1, 0], v: [1, 0, 0], normal: [ 0, 0,-1] }, // -Z
]

self.onmessage = (e: MessageEvent<MesherRequest>) => {
	const { sizeX, sizeY, sizeZ, voxels: voxBuf, blockColors } = e.data
	const voxels = new Uint8Array(voxBuf)

	const positions: number[] = []
	const normals: number[] = []
	const colors: number[] = []
	const uvs: number[] = []
	const indices: number[] = []

	const dims = [sizeX, sizeY, sizeZ]

	const index = (x: number, y: number, z: number) => x + y * sizeX + z * sizeX * sizeY

	for (let dir = 0; dir < 6; dir++) {
		const { d, u, v, normal } = DIRS[dir]
		const x = [0, 0, 0]
		const q = d
		const du = [0, 0, 0]
		const dv = [0, 0, 0]
		du[0] = u[0]; du[1] = u[1]; du[2] = u[2]
		dv[0] = v[0]; dv[1] = v[1]; dv[2] = v[2]

		const mask = new Uint32Array(dims[(dir/2|0 + 1) % 3] * dims[(dir/2|0 + 2) % 3])

		for (x[0]=0; x[0] < dims[0]; x[0]++)
		for (x[1]=0; x[1] < dims[1]; x[1]++)
		for (x[2]=0; x[2] < dims[2]; x[2]++) {
			// Build mask for the current slice along q axis
			let n=0
			for (let j=0; j<dims[(dir/2|0 + 2) % 3]; j++) {
				for (let i=0; i<dims[(dir/2|0 + 1) % 3]; i++) {
					const aX = x[0], aY = x[1], aZ = x[2]
					const bX = aX + q[0], bY = aY + q[1], bZ = aZ + q[2]
					const a = inside(aX,aY,aZ) ? voxels[index(aX,aY,aZ)] : 0
					const b = inside(bX,bY,bZ) ? voxels[index(bX,bY,bZ)] : 0
					const faceVoxel = dir % 2 === 0 ? a : b
					mask[n++] = (a !== 0) !== (b !== 0) ? faceVoxel : 0
					x[0] += du[0]; x[1] += du[1]; x[2] += du[2]
				}
				x[0] -= du[0]*dims[(dir/2|0 + 1) % 3]; x[1] -= du[1]*dims[(dir/2|0 + 1) % 3]; x[2] -= du[2]*dims[(dir/2|0 + 1) % 3]
				x[0] += dv[0]; x[1] += dv[1]; x[2] += dv[2]
			}
			x[0] -= dv[0]*dims[(dir/2|0 + 2) % 3]; x[1] -= dv[1]*dims[(dir/2|0 + 2) % 3]; x[2] -= dv[2]*dims[(dir/2|0 + 2) % 3]

			// Greedy merge on mask
			let m=0
			for (let j=0; j<dims[(dir/2|0 + 2) % 3]; j++) {
				for (let i=0; i<dims[(dir/2|0 + 1) % 3]; ) {
					const voxelType = mask[m]
					if (voxelType === 0) { m++; i++; continue }
					// Compute quad width
					let w = 1
					while (i + w < dims[(dir/2|0 + 1) % 3] && mask[m + w] === voxelType) w++
					// Compute quad height
					let h = 1
					while (j + h < dims[(dir/2|0 + 2) % 3]) {
						let k=0
						for (; k<w; k++) if (mask[m + k + h * dims[(dir/2|0 + 1) % 3]] !== voxelType) break
						if (k < w) break
						h++
					}
					// Set mask to zero for consumed area
					for (let dj=0; dj<h; dj++) {
						for (let di=0; di<w; di++) {
							mask[m + di + dj * dims[(dir/2|0 + 1) % 3]] = 0
						}
					}
					// Compute quad corners in world space
					x[0] = 0; x[1] = 0; x[2] = 0
					const start = [0,0,0]
					start[(dir/2|0)] = (dir % 2 === 0) ? 0 : -1
					start[(dir/2|0 + 1) % 3] = i
					start[(dir/2|0 + 2) % 3] = j
					const duv = [0,0,0]
					const dvv = [0,0,0]
					duv[(dir/2|0 + 1) % 3] = w
					dvv[(dir/2|0 + 2) % 3] = h

					// Corner positions
					const corners = [
						[start[0], start[1], start[2]],
						[start[0] + duv[0], start[1] + duv[1], start[2] + duv[2]],
						[start[0] + duv[0] + dvv[0], start[1] + duv[1] + dvv[1], start[2] + duv[2] + dvv[2]],
						[start[0] + dvv[0], start[1] + dvv[1], start[2] + dvv[2]],
					]

					// Adjust along normal to correct side
					for (const c of corners) { c[0]+= (dir===0?1:0); c[1]+= (dir===2?1:0); c[2]+= (dir===4?1:0) }

					const base = positions.length / 3
					const colorHex = blockColors[voxelType] ?? 0xffffff
					const cr = ((colorHex >> 16) & 255) / 255
					const cg = ((colorHex >> 8) & 255) / 255
					const cb = (colorHex & 255) / 255

					for (const c of corners) {
						positions.push(c[0], c[1], c[2])
						normals.push(normal[0], normal[1], normal[2])
						uvs.push(0, 0)
						colors.push(cr, cg, cb)
					}
					indices.push(base, base+1, base+2, base, base+2, base+3)

					m += w
					i += w
				}
			}
		}
	}

	const positionsArr = new Float32Array(positions)
	const normalsArr = new Float32Array(normals)
	const colorsArr = new Float32Array(colors)
	const uvsArr = new Float32Array(uvs)
	const indicesArr = new Uint32Array(indices)
	const res: MesherResponse = {
		positions: positionsArr,
		normals: normalsArr,
		colors: colorsArr,
		uvs: uvsArr,
		indices: indicesArr,
	}
	;(self as any).postMessage(res, [positionsArr.buffer, normalsArr.buffer, colorsArr.buffer, uvsArr.buffer, indicesArr.buffer])

	function inside(x: number, y: number, z: number): boolean {
		return x >= 0 && y >= 0 && z >= 0 && x < sizeX && y < sizeY && z < sizeZ
	}
}
