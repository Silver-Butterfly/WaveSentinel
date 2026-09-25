import * as THREE from 'three';

// =========================================================
// SEABED TERRAIN & DEEP-SEA ENVIRONMENT
// =========================================================

export function buildSeabed(scene) {
	const seabedSize = 10000;
	const seabedGeom = new THREE.PlaneGeometry(seabedSize, seabedSize, 100, 100);
	seabedGeom.rotateX(-Math.PI / 2);

	// Undulating underwater hills and canyons
	const pos = seabedGeom.attributes.position;
	for (let i = 0; i < pos.count; i++) {
		const vx = pos.getX(i);
		const vz = pos.getZ(i);
		const height = Math.sin(vx * 0.008) * Math.cos(vz * 0.008) * 22.0 +
			Math.sin(vx * 0.02 + vz * 0.015) * 8.0;
		pos.setY(i, height - 250.0);
	}
	seabedGeom.computeVertexNormals();

	const seabedMat = new THREE.MeshStandardMaterial({
		color: 0x0a1c2a,
		roughness: 0.95,
		metalness: 0.1,
		flatShading: true
	});

	const seabedMesh = new THREE.Mesh(seabedGeom, seabedMat);
	scene.add(seabedMesh);

	return { seabedMesh };
}
