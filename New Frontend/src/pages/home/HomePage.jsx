import React, { useLayoutEffect, useState, useRef, useEffect } from 'react';
import './HomePage.css';
import OceanScene from '../../components/ocean/OceanScene.jsx';
import PortalSection from '../../components/portal/PortalSection.jsx';
import CommandInterface from '../../components/command/CommandInterface.jsx';
import TeamSection from '../../components/team/TeamSection.jsx';
import storyImage from '../../assets/images/story.png';
import audioFile from '../../assets/new 88888-trimmed.mp3';

export default function HomePage({ onOpenDashboard, onOpenWebpage, onOpenNewModule, onOpenSim2D, targetSection }) {
	const [isPlaying, setIsPlaying] = useState(false);
	const audioRef = useRef(null);

	const toggleAudio = () => {
		if (audioRef.current) {
			if (isPlaying) {
				audioRef.current.pause();
			} else {
				audioRef.current.play();
			}
			setIsPlaying(!isPlaying);
		}
	};

	useLayoutEffect(() => {
		if (targetSection) {
			const el = document.getElementById(targetSection);
			if (el) el.scrollIntoView({ behavior: 'instant' });
			// Fallback in case it needs a tick
			setTimeout(() => {
				const el2 = document.getElementById(targetSection);
				if (el2) el2.scrollIntoView({ behavior: 'instant' });
			}, 10);
		} else {
			window.scrollTo(0, 0);
		}
	}, [targetSection]);

	const [isScrolled, setIsScrolled] = useState(false);

	useEffect(() => {
		const handleScroll = () => {
			const portal = document.getElementById('zone-portal');
			if (portal) {
				const rect = portal.getBoundingClientRect();
				// Turn navbar solid when Portal section reaches the top of the screen (or navbar area)
				if (rect.top <= 60) {
					setIsScrolled(true);
				} else {
					setIsScrolled(false);
				}
			} else {
				if (window.scrollY > 10) {
					setIsScrolled(true);
				} else {
					setIsScrolled(false);
				}
			}
		};
		window.addEventListener('scroll', handleScroll, { passive: true });
		return () => window.removeEventListener('scroll', handleScroll);
	}, []);

	return (
		<div className="home-page">
			{/* Top Navigation */}
			<nav className={`wave-nav ${isScrolled ? 'scrolled' : ''}`}>
				<button 
					className="audio-text-btn"
					onClick={toggleAudio}
					aria-label="Toggle Audio"
				>
					{isPlaying ? 'SOUND ON' : 'SOUND OFF'}
				</button>
				<div className="wave-links">
					<a href="#zone-ocean" onClick={(e) => { e.preventDefault(); window.scrollTo({top: 0, behavior: 'smooth'})}}>HOME</a>
					<a href="#zone-portal" onClick={(e) => { e.preventDefault(); document.getElementById('zone-portal')?.scrollIntoView({behavior: 'smooth'})}}>STORY</a>
					<a href="#zone-cube" onClick={(e) => { e.preventDefault(); document.getElementById('zone-cube')?.scrollIntoView({behavior: 'smooth'})}}>MODEL</a>
					<a href="#zone-team" onClick={(e) => { e.preventDefault(); document.getElementById('zone-team')?.scrollIntoView({behavior: 'smooth'})}}>TEAM</a>
				</div>
			</nav>

			{/* Audio Element */}
			<audio ref={audioRef} src={audioFile} loop />

			{/* Zone 1: Fixed 3D Ocean Simulation & HUD Navigation */}
			<OceanScene />

			{/* Scrollable Overlay Content */}
			<div className="scroll-root" id="scroll-root">
				{/* Zone 2: External Portal with Parallax Split Effect */}
				<PortalSection onOpenWebpage={onOpenWebpage} previewImage={storyImage} />


				{/* Zone 4: Autonomous Command Interface with Dedicated Cube Canvas */}
				<CommandInterface onOpenDashboard={onOpenDashboard} />

				{/* Zone 5: Team Showcase (3x2 Halftone Reveal Grid) */}
				<TeamSection />
			</div>
		</div>
	);
}

