import React, { useEffect, useRef, useState } from 'react';
import { CubeSceneController } from '../../three/cubeScene.js';
import GridScan from '../gridscan/GridScan.jsx';
import './CommandInterface.css';

export default function CommandInterface({ onOpenDashboard }) {
	const canvasRef = useRef(null);
	const controllerRef = useRef(null);
	const [isExpanding, setIsExpanding] = useState(false);

	useEffect(() => {
		if (!canvasRef.current) return;
		const controller = new CubeSceneController(canvasRef.current);
		controller.init();
		controllerRef.current = controller;

		return () => {
			controller.destroy();
		};
	}, []);

	const handleCubeClick = () => {
		if (isExpanding) return;
		setIsExpanding(true);

		if (controllerRef.current) {
			controllerRef.current.triggerExpand(() => {
				if (onOpenDashboard) onOpenDashboard();
			});
		} else if (onOpenDashboard) {
			setTimeout(onOpenDashboard, 500);
		}
	};

	return (
		<section id="zone-cube" style={{ position: 'relative' }}>
			{/* GridScan Background */}
			<div style={{ width: '100%', height: '100vh', position: 'absolute', top: 0, left: 0, zIndex: 0 }}>
				<GridScan
					sensitivity={0.55}
					lineThickness={1.7}
					linesColor="#b7b7b7"
					gridScale={0.1}
					scanColor="#0021ff"
					scanOpacity={0.4}
					enablePost
					bloomIntensity={0.6}
					chromaticAberration={0.002}
					noiseIntensity={0}
					lineJitter={0.08}
					scanGlow={0.6}
					scanSoftness={1.3}
					enableWebcam
				/>
				{/* Top transition gradient to blend seamlessly with the section above */}
				<div style={{ position: 'absolute', top: 0, left: 0, right: 0, height: '250px', background: 'linear-gradient(to bottom, rgba(0,0,0,1) 0%, rgba(0,0,0,0) 100%)', pointerEvents: 'none' }} />
				{/* Bottom transition gradient to blend seamlessly with the section below */}
				<div style={{ position: 'absolute', bottom: 0, left: 0, right: 0, height: '250px', background: 'linear-gradient(to top, rgba(0,0,0,1) 0%, rgba(0,0,0,0) 100%)', pointerEvents: 'none' }} />
			</div>

			<div id="cube-layout" style={{ position: 'relative', zIndex: 1 }}>
				{/* Top Header */}
				<div className={`cube-header ${isExpanding ? 'fading-out' : ''}`}>

					<h2 className="cube-title">COMMAND CENTER</h2>
				</div>

				{/* Interactive Spinning Reflective Hollow Cube (Click to Expand & Launch Dashboard) */}
				<div
					id="cube-canvas-wrap"
					className={isExpanding ? 'is-expanding' : ''}
					onClick={handleCubeClick}
					title="Click to launch Mission Dashboard"
					role="button"
					tabIndex={0}
					onKeyDown={(e) => (e.key === 'Enter' || e.key === ' ') && handleCubeClick()}
				>
					<canvas id="cube-canvas" width="480" height="480" ref={canvasRef}></canvas>
					<div className="cube-ring"></div>
				</div>

				{/* Telemetry interaction hint */}
				<div className={`cube-hint-wrap ${isExpanding ? 'fading-out' : ''}`}>
					<span className="cube-click-hint">[ CLICK CUBE TO ACCESS DASHBOARD ]</span>
				</div>
			</div>
		</section>
	);
}
