import * as THREE from 'three';

// =========================================================
// PROCEDURAL 3D AUV BUILDER & POSE CONTROLLER
// =========================================================

export function buildSubmarine() {
	const sub = new THREE.Group();

	// 1. Materials
	const hullMat = new THREE.MeshStandardMaterial({
		color: 0x777777, // Lighter grey
		roughness: 0.35, // More glossy
		metalness: 0.65  // More metallic
	});

	const darkMat = new THREE.MeshStandardMaterial({
		color: 0x0a0a0a,
		roughness: 0.9,
		metalness: 0.1
	});

	const hullRadius = 2.0;
	const midLength = 14.0;
	
	// 2. Hull Construction
	// Nose (Hemisphere)
	const noseGeom = new THREE.SphereGeometry(hullRadius, 32, 16, 0, Math.PI * 2, 0, Math.PI / 2);
	noseGeom.rotateZ(-Math.PI / 2); // Point forward (+X)
	const nose = new THREE.Mesh(noseGeom, hullMat);
	nose.position.set(midLength / 2, 0, 0);
	sub.add(nose);

	// Mid Body (Cylinder)
	const midGeom = new THREE.CylinderGeometry(hullRadius, hullRadius, midLength, 32);
	midGeom.rotateZ(Math.PI / 2);
	const midBody = new THREE.Mesh(midGeom, hullMat);
	sub.add(midBody);

	// Tapered Tail
	const tailLength = 4.0;
	const tailEndRadius = 0.8;
	const tailGeom = new THREE.CylinderGeometry(hullRadius, tailEndRadius, tailLength, 32);
	tailGeom.rotateZ(Math.PI / 2);
	const tail = new THREE.Mesh(tailGeom, hullMat);
	const tailPos = -(midLength / 2 + tailLength / 2);
	tail.position.set(tailPos, 0, 0);
	sub.add(tail);

	// 3. X-Tail Fins
	const finShape = new THREE.Shape();
	finShape.moveTo(1.2, 0);       // front base
	finShape.lineTo(0.2, 2.8);     // front tip (swept back)
	finShape.lineTo(-1.2, 2.8);    // rear tip
	finShape.lineTo(-1.8, 0);      // rear base
	finShape.lineTo(1.2, 0);
	
	const extrudeSettings = { depth: 0.12, bevelEnabled: true, bevelSegments: 2, steps: 1, bevelSize: 0.03, bevelThickness: 0.03 };
	const finGeom = new THREE.ExtrudeGeometry(finShape, extrudeSettings);
	finGeom.translate(0, 0, -0.06); // Center the depth

	const finGroup = new THREE.Group();
	for (let i = 0; i < 4; i++) {
		const fin = new THREE.Mesh(finGeom, hullMat);
		fin.rotation.x = (Math.PI / 2) * i + (Math.PI / 4); // X-configuration (45 deg)
		finGroup.add(fin);
	}
	finGroup.position.set(tailPos - 0.5, 0, 0);
	sub.add(finGroup);

	// 4. Propulsor & Cage
	const shroudLength = 2.0;
	const shroudRadius = 1.35;
	
	// Shroud inner and outer
	const shroudGeom = new THREE.CylinderGeometry(shroudRadius, shroudRadius, shroudLength, 32, 1, true);
	shroudGeom.rotateZ(Math.PI / 2);
	const shroud = new THREE.Mesh(shroudGeom, new THREE.MeshStandardMaterial({
		color: 0x1f1f1f, roughness: 0.75, metalness: 0.25, side: THREE.DoubleSide
	}));
	
	const shroudOuterGeom = new THREE.CylinderGeometry(shroudRadius + 0.05, shroudRadius + 0.05, shroudLength, 32, 1, true);
	shroudOuterGeom.rotateZ(Math.PI / 2);
	const shroudOuter = new THREE.Mesh(shroudOuterGeom, hullMat);
	shroud.add(shroudOuter);
	
	const shroudPos = -(midLength / 2 + tailLength + shroudLength / 2);
	shroud.position.set(shroudPos, 0, 0);
	sub.add(shroud);

	// Propeller
	const propGroup = new THREE.Group();
	const propHubGeom = new THREE.CylinderGeometry(0.3, 0.45, 0.8, 16);
	propHubGeom.rotateZ(Math.PI / 2);
	const propHub = new THREE.Mesh(propHubGeom, darkMat);
	propGroup.add(propHub);

	for (let i = 0; i < 7; i++) { // 7 bladed prop
		const angle = (i / 7) * Math.PI * 2;
		const bladeGeom = new THREE.BoxGeometry(0.35, 1.25, 0.05);
		const blade = new THREE.Mesh(bladeGeom, darkMat);
		blade.position.set(0, Math.sin(angle) * 0.65, Math.cos(angle) * 0.65);
		blade.rotation.x = angle;
		blade.rotation.y = 0.5; // pitch
		propGroup.add(blade);
	}
	propGroup.position.set(shroudPos, 0, 0);
	sub.add(propGroup);

	// Propulsor Cage
	const cageGroup = new THREE.Group();
	const ringGeom = new THREE.TorusGeometry(shroudRadius - 0.1, 0.03, 8, 32);
	ringGeom.rotateY(Math.PI / 2);
	
	const ring1 = new THREE.Mesh(ringGeom, hullMat);
	ring1.position.set(shroudLength / 2 + 0.2, 0, 0);
	cageGroup.add(ring1);
	const ring2 = new THREE.Mesh(ringGeom, hullMat);
	ring2.position.set(shroudLength / 2 + 0.6, 0, 0);
	cageGroup.add(ring2);

	const rodGeom = new THREE.CylinderGeometry(0.03, 0.03, 1.0, 8);
	rodGeom.rotateZ(Math.PI / 2);
	for(let i=0; i<8; i++) {
		const angle = (i / 8) * Math.PI * 2;
		const rod = new THREE.Mesh(rodGeom, hullMat);
		rod.position.set(shroudLength / 2 + 0.4, Math.sin(angle) * (shroudRadius - 0.1), Math.cos(angle) * (shroudRadius - 0.1));
		cageGroup.add(rod);
	}
	cageGroup.position.set(shroudPos, 0, 0);
	sub.add(cageGroup);

	// 5. Mast Assembly
	const mastGroup = new THREE.Group();
	
	const mastBaseGeom = new THREE.CylinderGeometry(0.5, 0.6, 1.5, 32);
	const mastBase = new THREE.Mesh(mastBaseGeom, hullMat);
	mastBase.position.set(0, hullRadius + 0.75, 0);
	mastGroup.add(mastBase);

	const plateGeom = new THREE.CylinderGeometry(0.8, 0.8, 0.1, 32);
	const plate = new THREE.Mesh(plateGeom, hullMat);
	plate.position.set(0, hullRadius + 1.5, 0);
	mastGroup.add(plate);

	// Antennas
	const ant1 = new THREE.Mesh(new THREE.CylinderGeometry(0.06, 0.06, 2.5, 16), hullMat);
	ant1.position.set(0, hullRadius + 1.5 + 1.25, 0);
	mastGroup.add(ant1);

	const ant2 = new THREE.Mesh(new THREE.CylinderGeometry(0.12, 0.12, 1.5, 16), hullMat);
	ant2.position.set(0.4, hullRadius + 1.5 + 0.75, 0);
	mastGroup.add(ant2);

	const ant3 = new THREE.Mesh(new THREE.CylinderGeometry(0.18, 0.18, 0.8, 16), hullMat);
	ant3.position.set(-0.4, hullRadius + 1.5 + 0.4, 0.3);
	for(let i=0; i<4; i++) {
		const ridge = new THREE.Mesh(new THREE.TorusGeometry(0.20, 0.03, 8, 16), hullMat);
		ridge.rotation.x = Math.PI / 2;
		ridge.position.set(-0.4, hullRadius + 1.5 + 0.3 + (i * 0.15), 0.3);
		mastGroup.add(ridge);
	}
	mastGroup.add(ant3);

	const ant4 = new THREE.Mesh(new THREE.CylinderGeometry(0.04, 0.04, 1.0, 8), hullMat);
	ant4.position.set(-0.3, hullRadius + 1.5 + 0.5, -0.4);
	mastGroup.add(ant4);

	mastGroup.position.set(midLength * 0.2, 0, 0);
	sub.add(mastGroup);

	// 6. Lifting Lugs and Side Bumps
	const lugGeom = new THREE.BoxGeometry(0.6, 0.2, 0.15);
	const lug1 = new THREE.Mesh(lugGeom, hullMat);
	lug1.position.set(midLength * 0.4, hullRadius, 0);
	sub.add(lug1);
	
	const lug2 = new THREE.Mesh(lugGeom, hullMat);
	lug2.position.set(0, hullRadius, 0);
	sub.add(lug2);

	const bumpGeom = new THREE.CylinderGeometry(0.1, 0.1, 0.2, 16);
	bumpGeom.rotateX(Math.PI / 2);
	for(let i=0; i<3; i++) {
		const bumpL = new THREE.Mesh(bumpGeom, hullMat);
		bumpL.position.set(midLength * 0.2 - (i*3), 0, hullRadius);
		sub.add(bumpL);

		const bumpR = new THREE.Mesh(bumpGeom, hullMat);
		bumpR.position.set(midLength * 0.2 - (i*3), 0, -hullRadius);
		sub.add(bumpR);
	}

	// 7. Subtle Illumination
	const topFill = new THREE.PointLight(0xffffff, 2.5, 40);
	topFill.position.set(0, hullRadius + 8.0, 0);
	sub.add(topFill);

	const sideFill = new THREE.PointLight(0x7dd3fc, 1.8, 50);
	sideFill.position.set(-2, 2, hullRadius + 12);
	sub.add(sideFill);

	const tailFill = new THREE.PointLight(0xffffff, 1.5, 30);
	tailFill.position.set(shroudPos, 2, 5);
	sub.add(tailFill);

	// Scale & position submarine to fit the existing scene perfectly
	sub.scale.set(1.5, 1.5, 1.5);
	sub.position.set(0, 0, 0);

	return {
		submarine: sub,
		propeller: propGroup,
		subSearchlight: null,
		lightCones: []
	};
}

// Submarine pitch, roll, and coordinate trajectories
export function updateSubmarinePose(submarine, camera, time, progress, diveVelocity) {
	const depthY = - (progress * 230.0);
	const forwardZ = - (progress * 140.0);

	// Surface buoyancy fades out quickly as we dive
	const surfaceFade = THREE.MathUtils.clamp(1.0 - (progress / 0.02), 0.0, 1.0);
	const waveBob = Math.sin(time * 1.6) * 0.45 * surfaceFade;
	const waveRoll = Math.cos(time * 1.2) * 0.035 * surfaceFade;

	// Base rotation from diving
	const staticPitch = THREE.MathUtils.clamp((progress - 0.005) / 0.015, 0.0, 1.0) * THREE.MathUtils.clamp((0.9 - progress) / 0.05, 0.0, 1.0) * 0.12;
	const diveAngle = THREE.MathUtils.clamp(-diveVelocity * 0.35 - staticPitch, -0.45, 0.45);
	const subRoll = Math.sin(time * 0.8) * 0.02 * (1.0 - surfaceFade);

	submarine.position.set(0, depthY + waveBob, forwardZ);
	submarine.rotation.x = THREE.MathUtils.lerp(submarine.rotation.x, diveAngle, 0.08);
	submarine.rotation.z = waveRoll + subRoll;

	// Smooth Camera Tracking
	const desiredCamY = submarine.position.y + 12.0;
	const desiredCamZ = submarine.position.z + 48.0;
	camera.position.y += (desiredCamY - camera.position.y) * 0.055;
	camera.position.z += (desiredCamZ - camera.position.z) * 0.055;
}
