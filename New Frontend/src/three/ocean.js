import * as THREE from 'three';
import { Water } from 'three/examples/jsm/objects/Water.js';
import { Sky } from 'three/examples/jsm/objects/Sky.js';
import { sunsetConfig } from './config.js';

export function createOceanWater(scene) {
	const waterGeometry = new THREE.PlaneGeometry(18000, 18000);
	const textureLoader = new THREE.TextureLoader();
	const waterNormals = textureLoader.load('/textures/waternormals.jpg', function (texture) {
		texture.wrapS = texture.wrapT = THREE.RepeatWrapping;
	});

	const water = new Water(
		waterGeometry,
		{
			textureWidth: 512,
			textureHeight: 512,
			waterNormals: waterNormals,
			sunDirection: new THREE.Vector3(),
			sunColor: 0xffaa44,
			waterColor: 0x081e2e,
			distortionScale: sunsetConfig.distortionScale,
			fog: scene.fog !== undefined,
			side: THREE.DoubleSide
		}
	);

	water.material.uniforms['size'].value = sunsetConfig.size;
	water.rotation.x = -Math.PI / 2;
	scene.add(water);

	return water;
}

export function createSkyBox(scene) {
	const sky = new Sky();
	sky.scale.setScalar(18000);
	scene.add(sky);

	const skyUniforms = sky.material.uniforms;
	skyUniforms['turbidity'].value = sunsetConfig.turbidity;
	skyUniforms['rayleigh'].value = sunsetConfig.rayleigh;
	skyUniforms['mieCoefficient'].value = sunsetConfig.mieCoefficient;
	skyUniforms['mieDirectionalG'].value = sunsetConfig.mieDirectionalG;

	return sky;
}

export function updateSunPosition(sun, sky, water, dirLight, fillLight, pmremGenerator, scene, sceneEnv) {
	const phi = THREE.MathUtils.degToRad(90 - sunsetConfig.elevation);
	const theta = THREE.MathUtils.degToRad(sunsetConfig.azimuth);

	sun.setFromSphericalCoords(1, phi, theta);

	sky.material.uniforms['sunPosition'].value.copy(sun);
	water.material.uniforms['sunDirection'].value.copy(sun).normalize();

	dirLight.position.copy(sun).multiplyScalar(1000);
	fillLight.position.set(-sun.x, Math.max(sun.y * 0.5, 0.2), -sun.z).multiplyScalar(800);

	sceneEnv.add(sky);
	const renderTarget = pmremGenerator.fromScene(sceneEnv);
	scene.add(sky);

	scene.environment = renderTarget.texture;
}
