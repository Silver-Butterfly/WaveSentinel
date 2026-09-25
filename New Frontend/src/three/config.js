// ==========================================
// Simulation Configuration & Lighting Settings
// ==========================================

export const sunsetConfig = {
	elevation: 3.5,
	azimuth: 180,
	exposure: 0.55,
	turbidity: 0,
	rayleigh: 4,
	mieCoefficient: 0.005,
	mieDirectionalG: 0.82,
	sunIntensity: 2.2,
	hemiIntensity: 1.1,
	ambientIntensity: 0.35,
	fillIntensity: 0.45,
	distortionScale: 1.2,
	size: 2.0,
	waveSpeed: 1.0,
	fogDensity: 0.00008
};

export const maxDepth = 250.0; // Max depth in meters (50m active floor)
export const PROGRESS_MAX = 0.20; // Underwater scene caps at 50m mark
