import React from 'react';
import { motion } from 'framer-motion';
import './PortalSection.css';

// Avatars removed

export default function PortalSection({ onOpenWebpage }) {
	return (
		<section id="zone-portal" className="network-portal" style={{ position: 'relative' }}>
			{/* Transition gradient above the section to blend the Ocean into black seamlessly */}
			<div style={{ position: 'absolute', top: '-250px', left: 0, right: 0, height: '252px', background: 'linear-gradient(to bottom, rgba(0,0,0,0) 0%, rgba(0,0,0,1) 100%)', pointerEvents: 'none', zIndex: 10 }} />

			{/* Full-screen Video Background */}
			<div style={{ position: 'absolute', inset: 0, width: '100%', height: '100%', zIndex: 0, overflow: 'hidden' }}>
				<video style={{ width: '100%', height: '100%', objectFit: 'cover' }} src="/animo-card-tunnel-720p.mp4" autoPlay muted loop playsInline />
				
				{/* Black tint overlay so it looks like it's in the background and keeps text readable */}
				<div style={{ position: 'absolute', inset: 0, backgroundColor: 'rgba(0, 0, 0, 0.65)' }} />
				
				{/* Top gradient blend */}
				<div style={{ position: 'absolute', top: 0, left: 0, right: 0, height: '250px', background: 'linear-gradient(to bottom, rgba(0,0,0,1) 0%, rgba(0,0,0,0) 100%)', zIndex: 1 }} />
				
				{/* Bottom gradient blend */}
				<div style={{ position: 'absolute', bottom: 0, left: 0, right: 0, height: '250px', background: 'linear-gradient(to top, rgba(0,0,0,1) 0%, rgba(0,0,0,0) 100%)', zIndex: 1 }} />
			</div>



			<div style={{ position: 'relative', zIndex: 10, width: '100%', height: '100vh', display: 'flex', flexDirection: 'column', justifyContent: 'center', alignItems: 'center', gap: '40px', boxSizing: 'border-box', textAlign: 'center', overflow: 'hidden' }}>
				{/* Hero Content */}
				<div className="network-hero" style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', maxWidth: '900px' }}>
					<motion.p 
						className="hero-subtitle"
						initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.7, delay: 0.1, ease: 'easeOut' }}
					>
						UNDERSTAND THE GLOBAL UNSEEN.
					</motion.p>
					<motion.h1 
						className="hero-title"
						initial={{ opacity: 0, y: 24 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.85, delay: 0.22, ease: 'easeOut' }}
						style={{ textShadow: '0 2px 30px rgba(0,0,0,0.5)' }}
					>
						THE OCEAN'S <br />
						<em style={{ fontStyle: 'italic', fontWeight: 500 }}>HIDDEN TOLL.</em>
					</motion.h1>
					<motion.div 
						className="network-access-right"
						initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.8, delay: 0.7, ease: 'easeOut' }}
						style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', marginBottom: '40px' }}
					>
						<motion.a
							href="#"
							onClick={(e) => {
								e.preventDefault();
								if (onOpenWebpage) onOpenWebpage();
							}}
							whileHover={{ scale: 1.02 }}
							whileTap={{ scale: 0.98 }}
							style={{ 
								display: 'inline-flex', alignItems: 'center', gap: '32px', 
								padding: '16px 16px 16px 48px', borderRadius: '100px', 
								background: 'rgba(255, 255, 255, 0.05)', 
								border: '1px solid rgba(255, 255, 255, 0.15)', 
								backdropFilter: 'blur(24px)',
								WebkitBackdropFilter: 'blur(24px)',
								textDecoration: 'none', 
								cursor: 'pointer',
								transition: 'all 0.4s cubic-bezier(0.23, 1, 0.32, 1)',
								boxShadow: '0 4px 30px rgba(0, 0, 0, 0.1)'
							}}
							onMouseEnter={(e) => {
								e.currentTarget.style.background = 'rgba(255, 255, 255, 0.1)';
								e.currentTarget.style.border = '1px solid rgba(255, 255, 255, 0.3)';
								e.currentTarget.style.boxShadow = '0 0 30px rgba(255,255,255,0.1)';
							}}
							onMouseLeave={(e) => {
								e.currentTarget.style.background = 'rgba(255, 255, 255, 0.05)';
								e.currentTarget.style.border = '1px solid rgba(255, 255, 255, 0.15)';
								e.currentTarget.style.boxShadow = '0 4px 30px rgba(0, 0, 0, 0.1)';
							}}
						>
							<span style={{ fontSize: '15px', fontFamily: '"Inter", sans-serif', fontWeight: 600, letterSpacing: '0.2em', textTransform: 'uppercase', color: '#fff' }}>CLICK HERE TO ACCESS THE STORY</span>
							<span style={{ width: '64px', height: '64px', borderRadius: '50%', background: '#fff', display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0 }}>
								<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#000" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"><path d="M5 12h14" /><path d="m12 5 7 7-7 7" /></svg>
							</span>
						</motion.a>
					</motion.div>

					<motion.p 
						className="hero-paragraph"
						initial={{ opacity: 0, y: 18 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.8, delay: 0.4, ease: 'easeOut' }}
						style={{ textShadow: '0 1px 14px rgba(0,0,0,0.5)', margin: 0 }}
					>
						Lost nets, storm wreckage, and industrial debris are quietly reshaping India's coastline drowning wildlife, blocking ports, and burying reefs that took centuries to grow.
					</motion.p>
				</div>
				

			</div>
		</section>
	);
}
