import React, { useEffect, useRef, useState, useCallback } from 'react';
import { useScroll } from 'framer-motion';
import { OceanSimulation } from '../../three/OceanSimulation.js';
import { PROGRESS_MAX } from '../../three/config.js';
import ScrollHint from '../hud/ScrollHint.jsx';
import './OceanScene.css';

export default function OceanScene() {
	const containerRef = useRef(null);
	const zoneOceanRef = useRef(null);
	const simRef = useRef(null);

	const [status, setStatus] = useState('SURFACE CRUISE');
	const [lightsEnabled, setLightsEnabled] = useState(true);
	const [progress, setProgress] = useState(0.0);

	const { scrollYProgress } = useScroll({
		target: zoneOceanRef,
		offset: ['start start', 'end start']
	});

	const handleTelemetryUpdate = useCallback(({ depth, velocity, progress: curProgress }) => {
		setProgress(curProgress);

		if (curProgress < 0.03) {
			setStatus('SURFACE CRUISE');
		} else if (Math.abs(velocity) > 0.05 && curProgress < 0.38) {
			setStatus(velocity > 0 ? 'DIVING...' : 'SURFACING...');
		} else if (curProgress >= 0.38) {
			setStatus('MESOPELAGIC ZONE');
		} else {
			setStatus('SUBMERGED PATROL');
		}
	}, []);

	// Initialize Three.js simulation
	useEffect(() => {
		if (!containerRef.current) return;

		const sim = new OceanSimulation(containerRef.current, {
			onTelemetryUpdate: handleTelemetryUpdate
		});
		simRef.current = sim;

		// Scroll listener for 3D dive progress
		const handleScroll = () => {
			if (!zoneOceanRef.current) return;
			const rect = zoneOceanRef.current.getBoundingClientRect();
			const zoneHeight = zoneOceanRef.current.offsetHeight;
			const scrolled = -rect.top;
			const raw = Math.max(0, scrolled) / zoneHeight;
			const clampedRaw = Math.max(0, Math.min(1.0, raw));
			const targetProgress = Math.max(0, Math.min(1.0, (clampedRaw * PROGRESS_MAX) / 1.0));
			sim.setTargetProgress(targetProgress);
		};

		const handleResize = () => {
			sim.onResize();
		};

		// Keyboard controls
		const handleKeyDown = (e) => {
			const scrollStep = window.innerHeight * 0.4;
			if (e.key === 'ArrowDown' || e.key === 'PageDown' || e.key === ' ') {
				window.scrollBy({ top: scrollStep, behavior: 'smooth' });
			} else if (e.key === 'ArrowUp' || e.key === 'PageUp') {
				window.scrollBy({ top: -scrollStep, behavior: 'smooth' });
			} else if (e.key === 'l' || e.key === 'L') {
				const active = sim.toggleLights();
				setLightsEnabled(active);
			}
		};

		window.addEventListener('scroll', handleScroll, { passive: true });
		window.addEventListener('resize', handleResize);
		window.addEventListener('keydown', handleKeyDown);

		return () => {
			window.removeEventListener('scroll', handleScroll);
			window.removeEventListener('resize', handleResize);
			window.removeEventListener('keydown', handleKeyDown);
			sim.destroy();
		};
	}, [handleTelemetryUpdate]);

	return (
		<>
			{/* Fixed WebGL Canvas Container */}
			<div className="ocean-canvas-container" id="container" ref={containerRef}></div>

			{/* Zone 1: Transparent Ocean Scrolling Runway */}
			<section className="zone-ocean" id="zone-ocean" ref={zoneOceanRef}>
				<ScrollHint progress={progress} />
			</section>
		</>
	);
}
