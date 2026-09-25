import * as THREE from 'three';
import { sunsetConfig } from './config.js';

// =========================================================
// UNDERWATER ATMOSPHERIC & LIGHTING TRANSITIONS
// Pure Oceanic Cyan-Blue to Pitch-Black Void
// Completely eliminates red/orange cast underwater
// =========================================================

// Shallow Underwater (0m - 10m): Crisp oceanic blue
const shallowOcean = {
	fogCol: new THREE.Color(0x0a4066),        // Brighter oceanic blue
	hemiSky: new THREE.Color(0x136a9f),       // Cyan-tinted downwelling skylight
	hemiGnd: new THREE.Color(0x052740),       // Deep water floor
	ambient: new THREE.Color(0x0c4b78),       // Ambient blue fill
	waterColor: new THREE.Color(0x0a395c),
	fogDensity: 0.0035,
	exposure: 0.85
};

// Mid-Depth (25m): Midnight Abyss Blue
const midDepth = {
	fogCol: new THREE.Color(0x062842),        // Deep navy
	hemiSky: new THREE.Color(0x0a4066),
	hemiGnd: new THREE.Color(0x03182b),
	ambient: new THREE.Color(0x073254),
	waterColor: new THREE.Color(0x05243b),
	fogDensity: 0.010,
	exposure: 0.65
};

// 50m Floor: Dark but visible blue
const abyssalEnd = {
	fogCol: new THREE.Color(0x021222),        // Dark but not black
	hemiSky: new THREE.Color(0x04223d),       // Ambient blue
	hemiGnd: new THREE.Color(0x010814),       // Ground dark blue
	ambient: new THREE.Color(0x03182b),       // Enough ambient so sub is visible
	waterColor: new THREE.Color(0x02101e),
	fogDensity: 0.025,                        // High fog density but not pure black
	exposure: 0.50
};

export function updateUnderwaterAtmosphere(progress, scene, renderer, dirLight, hemiLight, ambientLight, sky, water) {
	if (!scene || !renderer) return;

	let fogCol, fogDens, hemiSkyCol, hemiGndCol, ambientCol, exposureVal, waterCol;

	if (progress <= 0.015) {
		// Surface: Crisp Oceanic Horizon with subtle golden sunlight
		fogCol = new THREE.Color(0x0a3959); // Brighter surface fog
		fogDens = sunsetConfig.fogDensity;
		hemiSkyCol = new THREE.Color(0x1a5885);
		hemiGndCol = new THREE.Color(0x062035);
		ambientCol = new THREE.Color(0x0a385c);
		exposureVal = sunsetConfig.exposure;
		waterCol = new THREE.Color(0x0c3959); // Brighter surface water
		if (dirLight) {
			dirLight.color.setHex(0xe0f2fe);
			dirLight.intensity = sunsetConfig.sunIntensity * 1.15;
		}
		if (sky) sky.visible = true;
	} else if (progress < 0.10) {
		// 0m → 25m: Pure Deep Ocean Blue
		const t = (progress - 0.015) / 0.085;
		fogCol = shallowOcean.fogCol.clone().lerp(midDepth.fogCol, t);
		fogDens = THREE.MathUtils.lerp(shallowOcean.fogDensity, midDepth.fogDensity, t);
		hemiSkyCol = shallowOcean.hemiSky.clone().lerp(midDepth.hemiSky, t);
		hemiGndCol = shallowOcean.hemiGnd.clone().lerp(midDepth.hemiGnd, t);
		ambientCol = shallowOcean.ambient.clone().lerp(midDepth.ambient, t);
		exposureVal = THREE.MathUtils.lerp(shallowOcean.exposure, midDepth.exposure, t);
		waterCol = shallowOcean.waterColor.clone().lerp(midDepth.waterColor, t);
		if (dirLight) {
			dirLight.color.setHex(0x38bdf8);
			dirLight.intensity = sunsetConfig.sunIntensity * (1.15 - t * 0.7);
		}
		if (sky) sky.visible = true;
	} else if (progress < 0.20) {
		// 25m → 50m: Midnight navy smoothly fades into pure #000000 pitch black void
		const t = (progress - 0.10) / 0.10;
		fogCol = midDepth.fogCol.clone().lerp(abyssalEnd.fogCol, t);
		fogDens = THREE.MathUtils.lerp(midDepth.fogDensity, abyssalEnd.fogDensity, t);
		hemiSkyCol = midDepth.hemiSky.clone().lerp(abyssalEnd.hemiSky, t);
		hemiGndCol = midDepth.hemiGnd.clone().lerp(abyssalEnd.hemiGnd, t);
		ambientCol = midDepth.ambient.clone().lerp(abyssalEnd.ambient, t);
		exposureVal = THREE.MathUtils.lerp(midDepth.exposure, abyssalEnd.exposure, t);
		waterCol = midDepth.waterColor.clone().lerp(abyssalEnd.waterColor, t);
		if (dirLight) {
			dirLight.color.setHex(0x0284c7);
			dirLight.intensity = sunsetConfig.sunIntensity * 0.35 * (1.0 - t);
		}
		if (sky) sky.visible = false;
	} else {
		// 50m and beyond: Absolute seamless pure black
		fogCol = abyssalEnd.fogCol;
		fogDens = abyssalEnd.fogDensity;
		hemiSkyCol = abyssalEnd.hemiSky;
		hemiGndCol = abyssalEnd.hemiGnd;
		ambientCol = abyssalEnd.ambient;
		exposureVal = abyssalEnd.exposure;
		waterCol = abyssalEnd.waterColor;
		if (dirLight) dirLight.intensity = 0.0;
		if (sky) sky.visible = false;
	}

	if (scene.fog) {
		scene.fog.color.copy(fogCol);
		scene.fog.density = fogDens;
		renderer.setClearColor(fogCol);
	}

	if (hemiLight) {
		hemiLight.color.copy(hemiSkyCol);
		hemiLight.groundColor.copy(hemiGndCol);
	}
	if (ambientLight) {
		ambientLight.color.copy(ambientCol);
	}
	renderer.toneMappingExposure = exposureVal;

	if (water && water.material && water.material.uniforms && water.material.uniforms['waterColor']) {
		water.material.uniforms['waterColor'].value.copy(waterCol);
	}
}
