import * as THREE from 'three';

// =========================================================
// ISOLATED CUBE SCENE
// A self-contained Three.js renderer on a dedicated canvas
// Outer Cube: Highly reflective light blue
// Inner Cube: Dim glowing cyan light source
// =========================================================

const CANVAS_SIZE = 480; // High resolution with ample frustum padding

function createThickHollowCube(size = 1.4, thickness = 0.09, material) {
	const group    = new THREE.Group();
	const strutGeo = new THREE.BoxGeometry(1, 1, 1);

	const halfSize = size / 2;
	const offset   = halfSize - thickness / 2;

	// 12 struts — 4 per axis
	const struts = [
		// X-axis edges
		{ pos: [ 0,  offset,  offset], scale: [size, thickness, thickness] },
		{ pos: [ 0,  offset, -offset], scale: [size, thickness, thickness] },
		{ pos: [ 0, -offset,  offset], scale: [size, thickness, thickness] },
		{ pos: [ 0, -offset, -offset], scale: [size, thickness, thickness] },
		// Y-axis edges
		{ pos: [ offset, 0,  offset], scale: [thickness, size, thickness] },
		{ pos: [ offset, 0, -offset], scale: [thickness, size, thickness] },
		{ pos: [-offset, 0,  offset], scale: [thickness, size, thickness] },
		{ pos: [-offset, 0, -offset], scale: [thickness, size, thickness] },
		// Z-axis edges
		{ pos: [ offset,  offset, 0], scale: [thickness, thickness, size] },
		{ pos: [ offset, -offset, 0], scale: [thickness, thickness, size] },
		{ pos: [-offset,  offset, 0], scale: [thickness, thickness, size] },
		{ pos: [-offset, -offset, 0], scale: [thickness, thickness, size] },
	];

	struts.forEach(({ pos, scale }) => {
		const mesh = new THREE.Mesh(strutGeo, material);
		mesh.position.set(...pos);
		mesh.scale.set(...scale);
		group.add(mesh);
	});

	return group;
}

export class CubeSceneController {
	constructor(canvasElement) {
		this.canvas = canvasElement;
		this.cubeRenderer = null;
		this.cubeScene = null;
		this.cubeCamera = null;
		this.hollowCube = null;
		this.innerCube = null;
		this.innerLight = null;
		this.rafId = null;
		this.running = false;
		this.observer = null;
		this.isExpanding = false;
		this.expandProgress = 0;
		this.expandProgress = 0;
		this.onExpandCallback = null;
		this.centerImage = null;
	}

	init() {
		if (!this.canvas) return;

		this.cubeScene = new THREE.Scene();
		this.cubeCamera = new THREE.PerspectiveCamera(46, 1, 0.1, 100);
		this.cubeCamera.position.z = 4.6;

		this.cubeRenderer = new THREE.WebGLRenderer({
			canvas: this.canvas,
			antialias: true,
			alpha: true
		});
		this.cubeRenderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
		this.cubeRenderer.setSize(CANVAS_SIZE, CANVAS_SIZE);
		this.cubeRenderer.setClearColor(0x000000, 0);
		this.cubeRenderer.shadowMap.enabled = false;

		// Lighting
		this.cubeScene.add(new THREE.AmbientLight(0x0f172a, 1.8));

		const keyLight = new THREE.DirectionalLight(0xe0f2fe, 3.2);
		keyLight.position.set(4, 5, 5);
		this.cubeScene.add(keyLight);

		const rimLight = new THREE.DirectionalLight(0x1e3a8a, 2.8);
		rimLight.position.set(-4, -2, -3);
		this.cubeScene.add(rimLight);

		const fillLight = new THREE.DirectionalLight(0x0284c7, 1.4);
		fillLight.position.set(0, -5, 2);
		this.cubeScene.add(fillLight);

		// 1. Outer Reflective Light Blue Cube
		const outerMat = new THREE.MeshStandardMaterial({
			color: 0x93c5fd, // Light blue / icy titanium
			roughness: 0.12,  // Highly reflective
			metalness: 0.88,
			envMapIntensity: 2.5
		});
		this.hollowCube = createThickHollowCube(1.4, 0.09, outerMat);
		this.cubeScene.add(this.hollowCube);

		// 2. Inner Dim Glowing Dark Blue Light Source Cube
		const innerMat = new THREE.MeshStandardMaterial({
			color: 0x1e3a8a,
			emissive: 0x1e3a8a,
			emissiveIntensity: 1.4,
			roughness: 0.2,
			metalness: 0.3
		});
		this.innerCube = createThickHollowCube(0.78, 0.055, innerMat);
		this.cubeScene.add(this.innerCube);

		// 3. Point light emanating from within the inner cube
		this.innerLight = new THREE.PointLight(0x1e3a8a, 3.8, 8);
		this.innerLight.position.set(0, 0, 0);
		this.cubeScene.add(this.innerLight);

		// 4. Center Image (Sprite)
		// It stays still in the middle and always faces the camera
		const textureLoader = new THREE.TextureLoader();
		const centerTexture = textureLoader.load('/cube-center.png');
		// Tint the image to a dark blue shade
		const centerMat = new THREE.SpriteMaterial({ map: centerTexture, color: 0x1e3a8a, transparent: true, depthWrite: false });
		this.centerImage = new THREE.Sprite(centerMat);
		this.centerImage.scale.set(0.85, 0.85, 1); // Increased size
		this.cubeScene.add(this.centerImage);

		// IntersectionObserver — only render when on-screen
		this.observer = new IntersectionObserver(entries => {
			entries.forEach(e => e.isIntersecting ? this.start() : this.stop());
		}, { threshold: 0.1 });
		this.observer.observe(this.canvas);
	}

	triggerExpand(onComplete) {
		if (this.isExpanding) return;
		this.isExpanding = true;
		this.expandProgress = 0;
		this.onExpandCallback = onComplete;
	}

	start() {
		if (this.running) return;
		this.running = true;
		this.tick();
	}

	stop() {
		this.running = false;
		if (this.rafId) cancelAnimationFrame(this.rafId);
		this.rafId = null;
	}

	tick() {
		if (!this.running) return;
		this.rafId = requestAnimationFrame(() => this.tick());

		const speedMultiplier = this.isExpanding ? 4.0 : 1.0;

		// Outer cube rotation
		if (this.hollowCube) {
			this.hollowCube.rotation.x += 0.008 * speedMultiplier;
			this.hollowCube.rotation.y += 0.012 * speedMultiplier;
			this.hollowCube.rotation.z += 0.003 * speedMultiplier;
		}

		// Inner cube counter-rotation
		if (this.innerCube) {
			this.innerCube.rotation.x -= 0.011 * speedMultiplier;
			this.innerCube.rotation.y -= 0.009 * speedMultiplier;
			this.innerCube.rotation.z -= 0.005 * speedMultiplier;
		}

		// Smooth Non-Clipping Acceleration & Flare Animation
		if (this.isExpanding) {
			this.expandProgress += 0.038;
			
			// Controlled 3D scale so it never hits the frustum border
			const meshScale = 1.0 + this.expandProgress * 0.35;
			if (this.hollowCube) this.hollowCube.scale.set(meshScale, meshScale, meshScale);
			if (this.innerCube) this.innerCube.scale.set(meshScale * 1.05, meshScale * 1.05, meshScale * 1.05);
			if (this.innerLight) this.innerLight.intensity = 3.8 + this.expandProgress * 25;
			if (this.centerImage) this.centerImage.scale.set(meshScale * 0.85, meshScale * 0.85, 1);

			if (this.expandProgress >= 1.0) {
				this.isExpanding = false;
				if (this.onExpandCallback) {
					const cb = this.onExpandCallback;
					this.onExpandCallback = null;
					cb();
				}
			}
		}

		if (this.cubeRenderer && this.cubeScene && this.cubeCamera) {
			this.cubeRenderer.render(this.cubeScene, this.cubeCamera);
		}
	}

	destroy() {
		this.stop();
		if (this.observer) {
			this.observer.disconnect();
			this.observer = null;
		}
		if (this.cubeRenderer) {
			this.cubeRenderer.dispose();
		}
	}
}
