import * as THREE from 'three';

// =========================================================
// PARTICLE SYSTEMS (UNIFORM STRATIFIED SCATTER & DRIFT)
// =========================================================

export function createCircleParticleTexture() {
	const canvas = document.createElement('canvas');
	canvas.width = 64;
	canvas.height = 64;
	const ctx = canvas.getContext('2d');

	const gradient = ctx.createRadialGradient(32, 32, 0, 32, 32, 32);
	gradient.addColorStop(0.0, 'rgba(255, 255, 255, 1.0)');
	gradient.addColorStop(0.25, 'rgba(220, 248, 255, 0.9)');
	gradient.addColorStop(0.6, 'rgba(125, 211, 252, 0.4)');
	gradient.addColorStop(1.0, 'rgba(125, 211, 252, 0.0)');

	ctx.fillStyle = gradient;
	ctx.beginPath();
	ctx.arc(32, 32, 32, 0, Math.PI * 2);
	ctx.fill();

	const texture = new THREE.CanvasTexture(canvas);
	return texture;
}

// Bounding box dimensions for the volumetric particle field
const BOX_X = 280.0;
const BOX_Y = 160.0;
const BOX_Z = 280.0;
const HALF_X = BOX_X / 2;
const HALF_Y = BOX_Y / 2;
const HALF_Z = BOX_Z / 2;

export function buildMarineParticles(scene, circleTexture) {
	// Stratified 3D grid dimensions for mathematically uniform spatial scattering
	const gridX = 18;
	const gridY = 16;
	const gridZ = 20;
	const particleCount = gridX * gridY * gridZ; // 5760 uniformly scattered particles

	const positions = new Float32Array(particleCount * 3);
	const marineSpeedOffsets = new Float32Array(particleCount);

	let index = 0;
	for (let ix = 0; ix < gridX; ix++) {
		for (let iy = 0; iy < gridY; iy++) {
			for (let iz = 0; iz < gridZ; iz++) {
				// Stratified jittered position inside each grid cell ensures zero clumping
				const x = ((ix + Math.random()) / gridX - 0.5) * BOX_X;
				const y = -2.0 - ((iy + Math.random()) / gridY) * BOX_Y;
				const z = ((iz + Math.random()) / gridZ - 0.5) * BOX_Z;

				positions[index * 3] = x;
				positions[index * 3 + 1] = y;
				positions[index * 3 + 2] = z;
				marineSpeedOffsets[index] = 0.75 + Math.random() * 0.5;
				index++;
			}
		}
	}

	const marineGeo = new THREE.BufferGeometry();
	marineGeo.setAttribute('position', new THREE.BufferAttribute(positions, 3));

	const particleMat = new THREE.PointsMaterial({
		color: 0xbaefff,
		size: 0.42, // Small, clear circular micro-particles
		map: circleTexture,
		transparent: true,
		opacity: 0.7,
		blending: THREE.AdditiveBlending,
		depthWrite: false,
		sizeAttenuation: true
	});

	const marineParticles = new THREE.Points(marineGeo, particleMat);
	scene.add(marineParticles);

	return { marineParticles, marineGeo, marineSpeedOffsets };
}

export function updateMarineParticles(marineGeo, marineSpeedOffsets, delta, subPos, diveVelocity) {
	if (!marineGeo || !subPos) return;
	const pos = marineGeo.attributes.position.array;
	const count = pos.length / 3;

	// Constant forward oceanic flow into the screen (-Z)
	const baseStationarySpeed = 10.0;
	const scrollBoostSpeed = Math.abs(diveVelocity) * 45.0;
	const currentSpeed = (baseStationarySpeed + scrollBoostSpeed) * delta;
	const time = performance.now() * 0.001;

	// Center point of the active particle volume (staying strictly submerged)
	const centerX = subPos.x;
	const centerY = Math.min(subPos.y, -HALF_Y - 2.0);
	const centerZ = subPos.z;

	for (let i = 0; i < count; i++) {
		const idx = i * 3;
		const speedFactor = marineSpeedOffsets[i];

		// Forward current away from camera into the screen
		pos[idx + 2] -= currentSpeed * speedFactor;

		// Subtle organic undulating drift
		pos[idx] += Math.sin(i * 0.08 + time * 0.7) * 0.01;
		pos[idx + 1] += Math.cos(i * 0.12 + time * 0.5) * 0.006;

		// Toroidal wrap in Z (seamless depth streaming)
		if (pos[idx + 2] < centerZ - HALF_Z) {
			pos[idx + 2] += BOX_Z;
		} else if (pos[idx + 2] > centerZ + HALF_Z) {
			pos[idx + 2] -= BOX_Z;
		}

		// Toroidal wrap in X (seamless horizontal distribution)
		if (pos[idx] < centerX - HALF_X) {
			pos[idx] += BOX_X;
		} else if (pos[idx] > centerX + HALF_X) {
			pos[idx] -= BOX_X;
		}

		// Toroidal wrap in Y (seamless vertical distribution without bottom piling)
		if (pos[idx + 1] < centerY - HALF_Y) {
			pos[idx + 1] += BOX_Y;
		} else if (pos[idx + 1] > centerY + HALF_Y) {
			pos[idx + 1] -= BOX_Y;
		}

		// Keep below ocean surface
		if (pos[idx + 1] > -1.0) {
			pos[idx + 1] -= 5.0;
		}
	}

	marineGeo.attributes.position.needsUpdate = true;
}

export function buildCavitationParticles(scene, circleTexture) {
	const bubbleCount = 400;
	const positions = new Float32Array(bubbleCount * 3);

	for (let i = 0; i < bubbleCount; i++) {
		positions[i * 3] = 0;
		positions[i * 3 + 1] = 0;
		positions[i * 3 + 2] = 0;
	}

	const cavitationGeo = new THREE.BufferGeometry();
	cavitationGeo.setAttribute('position', new THREE.BufferAttribute(positions, 3));

	const bubbleMat = new THREE.PointsMaterial({
		color: 0xe0f2fe,
		size: 0.85,
		map: circleTexture,
		transparent: true,
		opacity: 0.75,
		blending: THREE.AdditiveBlending,
		depthWrite: false,
		sizeAttenuation: true
	});

	const cavitationParticles = new THREE.Points(cavitationGeo, bubbleMat);
	scene.add(cavitationParticles);

	return { cavitationParticles, cavitationGeo };
}

let bubbleIndex = 0;
export function updateCavitation(cavitationGeo, subPos, speed) {
	if (!cavitationGeo) return;
	const pos = cavitationGeo.attributes.position.array;
	const count = pos.length / 3;

	// Spawn new bubbles at propeller position
	for (let k = 0; k < 3; k++) {
		bubbleIndex = (bubbleIndex + 1) % count;
		pos[bubbleIndex * 3] = subPos.x - 18.0 + (Math.random() - 0.5) * 1.5;
		pos[bubbleIndex * 3 + 1] = subPos.y + (Math.random() - 0.5) * 1.5;
		pos[bubbleIndex * 3 + 2] = subPos.z + (Math.random() - 0.5) * 1.5;
	}

	// Drift bubbles backwards and upwards
	for (let i = 0; i < count; i++) {
		pos[i * 3] -= speed * 0.4;
		pos[i * 3 + 1] += 0.08 + Math.random() * 0.05; // Float to surface
		// Dissipate near surface
		if (pos[i * 3 + 1] > -0.5) {
			pos[i * 3 + 1] = -999;
		}
	}
	cavitationGeo.attributes.position.needsUpdate = true;
}
