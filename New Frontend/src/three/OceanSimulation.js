import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';

import { sunsetConfig, maxDepth } from './config.js';
import { createOceanWater, createSkyBox, updateSunPosition } from './ocean.js';
import { updateUnderwaterAtmosphere } from './atmosphere.js';
import { buildSubmarine, updateSubmarinePose } from './submarine.js';
import { buildSeabed } from './seabed.js';
import {
	createCircleParticleTexture,
	buildMarineParticles,
	updateMarineParticles
} from './particles.js';
import { createTitleSprite } from './titleSprite.js';

export class OceanSimulation {
	constructor(container, options = {}) {
		this.container = container;
		this.onTelemetryUpdate = options.onTelemetryUpdate || null;

		this.camera = null;
		this.scene = null;
		this.renderer = null;
		this.timer = null;
		this.controls = null;
		this.water = null;
		this.sun = null;
		this.sky = null;

		this.dirLight = null;
		this.hemiLight = null;
		this.ambientLight = null;
		this.fillLight = null;
		this.underwaterAmbient = null;

		this.submarine = null;
		this.propeller = null;
		this.subSearchlight = null;
		this.lightCones = [];

		this.marineParticles = null;
		this.marineGeo = null;
		this.marineSpeedOffsets = null;
		this.seabedMesh = null;

		this.state = {
			scrollProgress: 0.0,
			targetProgress: 0.0,
			lightsEnabled: true
		};

		this.rafId = null;
		this.isDestroyed = false;

		this.init();
	}

	init() {
		if (!this.container) return;

		// 1. WebGL Renderer
		this.renderer = new THREE.WebGLRenderer({
			antialias: true,
			powerPreference: 'high-performance'
		});
		this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
		this.renderer.setSize(window.innerWidth, window.innerHeight);
		this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
		this.renderer.toneMappingExposure = sunsetConfig.exposure;
		this.container.appendChild(this.renderer.domElement);

		// 2. Scene & Fog Setup
		this.scene = new THREE.Scene();
		this.scene.fog = new THREE.FogExp2(0x082c48, sunsetConfig.fogDensity);

		// 3. Camera Setup
		this.camera = new THREE.PerspectiveCamera(55, window.innerWidth / window.innerHeight, 0.5, 30000);
		this.camera.position.set(32, 14, 52);
		this.camera.layers.enable(1); // Enable layer 1 so it can see the title sprite (preventing it from reflecting in water)

		this.timer = new THREE.Timer ? new THREE.Timer() : null;
		this.lastTime = performance.now();
		this.sun = new THREE.Vector3();

		// 4. Lighting Setup
		this.dirLight = new THREE.DirectionalLight(0xe0f2fe, sunsetConfig.sunIntensity);
		this.scene.add(this.dirLight);

		this.hemiLight = new THREE.HemisphereLight(0x1a5885, 0x062035, sunsetConfig.hemiIntensity);
		this.scene.add(this.hemiLight);

		this.ambientLight = new THREE.AmbientLight(0x0a385c, sunsetConfig.ambientIntensity);
		this.scene.add(this.ambientLight);

		this.fillLight = new THREE.DirectionalLight(0x0284c7, sunsetConfig.fillIntensity);
		this.scene.add(this.fillLight);

		this.underwaterAmbient = new THREE.AmbientLight(0x051a2e, 0.0);
		this.scene.add(this.underwaterAmbient);

		// 5. Ocean & Sky Systems
		this.water = createOceanWater(this.scene);
		this.sky = createSkyBox(this.scene);

		const pmremGenerator = new THREE.PMREMGenerator(this.renderer);
		const sceneEnv = new THREE.Scene();
		updateSunPosition(this.sun, this.sky, this.water, this.dirLight, this.fillLight, pmremGenerator, this.scene, sceneEnv);

		// 5.5 Parallax Title Sprite (rendered in WebGL so it can sit behind the submarine)
		this.titleSprite = createTitleSprite();
		// Set x to -180 to perfectly align with camera's steady state line of sight (dx=32, dz=32)
		this.titleSprite.position.set(-180, 50, -180);
		this.titleSprite.layers.set(1); // Only visible to main camera, excluded from water mirror reflection
		this.scene.add(this.titleSprite);

		// 6. Submarine Model
		const subData = buildSubmarine();
		this.submarine = subData.submarine;
		this.propeller = subData.propeller;
		this.subSearchlight = subData.subSearchlight;
		this.lightCones = subData.lightCones;
		this.scene.add(this.submarine);

		// 7. Seabed Terrain
		const seabedData = buildSeabed(this.scene);
		this.seabedMesh = seabedData.seabedMesh;

		// 8. Particle Systems
		const circleTexture = createCircleParticleTexture();
		const marineData = buildMarineParticles(this.scene, circleTexture);
		this.marineParticles = marineData.marineParticles;
		this.marineGeo = marineData.marineGeo;
		this.marineSpeedOffsets = marineData.marineSpeedOffsets;

		// 9. OrbitControls for smooth camera dampening
		this.controls = new OrbitControls(this.camera, this.renderer.domElement);
		this.controls.maxPolarAngle = Math.PI * 0.85;
		this.controls.minDistance = 15.0;
		this.controls.maxDistance = 200.0;
		this.controls.enableDamping = true;
		this.controls.dampingFactor = 0.05;
		this.controls.enableZoom = false;
		this.controls.enablePan = false;

		// 10. Start Animation Loop
		this.animate = this.animate.bind(this);
		this.rafId = requestAnimationFrame(this.animate);
	}

	setTargetProgress(progress) {
		this.state.targetProgress = THREE.MathUtils.clamp(progress, 0.0, 1.0);
	}

	toggleLights() {
		this.state.lightsEnabled = !this.state.lightsEnabled;
		if (this.subSearchlight) {
			this.subSearchlight.intensity = this.state.lightsEnabled ? 8.0 : 0.0;
		}
		this.lightCones.forEach(c => (c.visible = this.state.lightsEnabled));
		return this.state.lightsEnabled;
	}

	setLights(enabled) {
		this.state.lightsEnabled = enabled;
		if (this.subSearchlight) {
			this.subSearchlight.intensity = enabled ? 8.0 : 0.0;
		}
		this.lightCones.forEach(c => (c.visible = enabled));
	}

	onResize() {
		if (!this.camera || !this.renderer) return;
		this.camera.aspect = window.innerWidth / window.innerHeight;
		this.camera.updateProjectionMatrix();
		this.renderer.setSize(window.innerWidth, window.innerHeight);
	}

	animate() {
		if (this.isDestroyed) return;

		const now = performance.now();
		let delta = 0.016;
		if (this.timer) {
			this.timer.update();
			delta = this.timer.getDelta();
		} else {
			delta = Math.min((now - this.lastTime) * 0.001, 0.1);
			this.lastTime = now;
		}
		const time = now * 0.001;

		// Smooth scroll interpolation
		const prevProgress = this.state.scrollProgress;
		this.state.scrollProgress += (this.state.targetProgress - this.state.scrollProgress) * 0.055;
		const diveVelocity = (this.state.scrollProgress - prevProgress) * 60.0;

		// Depth calculation
		const currentDepthMeters = this.state.scrollProgress * maxDepth;

		// Update Submarine Motion & Trajectory
		if (this.submarine && this.camera) {
			updateSubmarinePose(this.submarine, this.camera, time, this.state.scrollProgress, diveVelocity);
		}

		// Update Atmosphere, Fog & Lighting
		updateUnderwaterAtmosphere(
			this.state.scrollProgress,
			this.scene,
			this.renderer,
			this.dirLight,
			this.hemiLight,
			this.ambientLight,
			this.sky,
			this.water
		);

		// Update Parallax Title (Moves UP, Fades OUT, Zooms OUT as we dive)
		if (this.titleSprite && this.submarine && this.camera) {
			const titleFadeProgress = Math.min(this.state.scrollProgress / 0.3, 1.0);
			
			// Keep perfectly centered horizontally by aligning with camera-submarine line of sight
			const camZ = this.camera.position.z;
			const subZ = this.submarine.position.z;
			const camX = this.camera.position.x;
			const subX = this.submarine.position.x;
			const spriteZ = this.titleSprite.position.z;
			const dz = subZ - camZ;
			if (Math.abs(dz) > 0.001) {
				const m = (subX - camX) / dz;
				this.titleSprite.position.x = camX + (spriteZ - camZ) * m;
			}

			// No artificial jump: let it naturally go up as camera dives.
			this.titleSprite.position.y = 50; 
			
			// Zoom out smoothly over the entire scroll, not just the first 30%
			const scaleFactor = 1.0 - (this.state.scrollProgress * 0.5); 
			this.titleSprite.scale.set(400 * scaleFactor, 100 * scaleFactor, 1);
			
			// Stay visible as it passes overhead underwater
			this.titleSprite.visible = true; 
		}

		// Update Propeller
		if (this.propeller) {
			const propSpeed = (8.0 + Math.abs(diveVelocity) * 40.0) * delta;
			this.propeller.rotation.x += propSpeed;
		}

		// Update Marine Particles
		if (this.marineParticles && this.submarine) {
			updateMarineParticles(this.marineGeo, this.marineSpeedOffsets, delta, this.submarine.position, diveVelocity);
		}

		// Water wave animation - use absolute time to prevent glitching/stuttering
		if (this.water && this.water.material && this.water.material.uniforms && this.water.material.uniforms['time']) {
			this.water.material.uniforms['time'].value = time * sunsetConfig.waveSpeed;
		}

		// Notify HUD Telemetry listeners
		if (this.onTelemetryUpdate) {
			this.onTelemetryUpdate({
				depth: currentDepthMeters,
				velocity: diveVelocity,
				progress: this.state.scrollProgress
			});
		}

		// Camera Chase Target
		if (this.controls && this.submarine) {
			this.controls.target.set(
				this.submarine.position.x,
				this.submarine.position.y,
				this.submarine.position.z
			);
			this.controls.update();
		}

		if (this.renderer && this.scene && this.camera) {
			this.renderer.render(this.scene, this.camera);
		}

		this.rafId = requestAnimationFrame(this.animate);
	}

	destroy() {
		this.isDestroyed = true;
		if (this.rafId) {
			cancelAnimationFrame(this.rafId);
			this.rafId = null;
		}
		if (this.renderer) {
			this.renderer.dispose();
			if (this.renderer.domElement && this.renderer.domElement.parentNode) {
				this.renderer.domElement.parentNode.removeChild(this.renderer.domElement);
			}
		}
	}
}
