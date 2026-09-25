import React from 'react';
import { teamMembers, mentorData } from '../../info/teamData.js';
import HalftoneReveal from './HalftoneReveal.jsx';
import './TeamSection.css';

export default function TeamSection() {
	return (
		<section id="zone-team">
			{/* Section Header */}
			<div className="team-header">
				<div className="team-badge-row">

					<span className="team-status-pill">DEPLOYED</span>
				</div>
				<h2 className="team-main-title">MEET THE TEAM</h2>
				<p className="team-sub-desc">
					Autonomous navigation engineers, underwater acoustic specialists, and neural telemetry architects powering WaveSentinel.
				</p>
				<div className="team-divider-line"></div>
			</div>

			{/* 3x2 Grid Container */}
			<div className="team-3x2-grid">
				{teamMembers.map((member) => (
					<div className="team-halftone-card" key={member.id}>
						<div className="card-corner top-left"></div>
						<div className="card-corner top-right"></div>
						<div className="card-corner bottom-left"></div>
						<div className="card-corner bottom-right"></div>

						<div className="team-image-wrapper">
							<HalftoneReveal
								src={member.src + "?rev=1"}
								inkColor="#141414"
								paperColor="#fff7e6"
								mode="mono"
								dotDensity={101}
								angle={56}
								revealRadius={0.16}
								dotSize={0.9}
								shape="circle"
								contrast={1.3}
								invert={false}
								edge={0.59}
								follow={0.37}
								idleReveal={0.1}
								trigger="hover"
								className="team-member-img"
							/>
						</div>

						<div className="team-card-info-bar">
							<div className="team-info-top">
								<h3 className="team-member-name">{member.name}</h3>
								<span className="team-role-tag">{member.title}</span>
							</div>
							<a
								href={member.linkedin}
								target="_blank"
								rel="noopener noreferrer"
								className="team-linkedin-btn"
								aria-label="LinkedIn"
							>
								<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor">
									<path d="M19 0h-14c-2.761 0-5 2.239-5 5v14c0 2.761 2.239 5 5 5h14c2.762 0 5-2.239 5-5v-14c0-2.761-2.238-5-5-5zm-11 19h-3v-11h3v11zm-1.5-12.268c-.966 0-1.75-.79-1.75-1.764s.784-1.764 1.75-1.764 1.75.79 1.75 1.764-.783 1.764-1.75 1.764zm13.5 12.268h-3v-5.604c0-3.368-4-3.113-4 0v5.604h-3v-11h3v1.765c1.396-2.586 7-2.777 7 2.476v6.759z" />
								</svg>
							</a>
						</div>
					</div>
				))}
			</div>

			{/* Mentor Section Header */}
			<div className="team-header" style={{ marginTop: '4rem' }}>
				<div className="team-badge-row">

				</div>
				<h2 className="team-main-title">OUR MENTOR</h2>
				<p className="team-sub-desc">
					Providing strategic direction and technical advisory for WaveSentinel.
				</p>
				<div className="team-divider-line"></div>
			</div>

			{/* Mentor Card Container */}
			<div className="team-3x2-grid" style={{ display: 'flex', justifyContent: 'center' }}>
				<div className="team-halftone-card" key={mentorData.id} style={{ maxWidth: '400px', width: '100%' }}>
					<div className="card-corner top-left"></div>
					<div className="card-corner top-right"></div>
					<div className="card-corner bottom-left"></div>
					<div className="card-corner bottom-right"></div>

					<div className="team-image-wrapper">
						<HalftoneReveal
							src={mentorData.src.includes('?') ? mentorData.src : mentorData.src + "?rev=1"}
							inkColor="#141414"
							paperColor="#fff7e6"
							mode="mono"
							dotDensity={101}
							angle={56}
							revealRadius={0.16}
							dotSize={0.9}
							shape="circle"
							contrast={1.3}
							invert={false}
							edge={0.59}
							follow={0.37}
							idleReveal={0.1}
							trigger="hover"
							className="team-member-img"
						/>
					</div>

					<div className="team-card-info-bar">
						<div className="team-info-top">
							<h3 className="team-member-name">{mentorData.name}</h3>
							<span className="team-role-tag">{mentorData.title}</span>
						</div>
						<a
							href={mentorData.linkedin}
							target="_blank"
							rel="noopener noreferrer"
							className="team-linkedin-btn"
							aria-label="LinkedIn"
						>
							<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor">
								<path d="M19 0h-14c-2.761 0-5 2.239-5 5v14c0 2.761 2.239 5 5 5h14c2.762 0 5-2.239 5-5v-14c0-2.761-2.238-5-5-5zm-11 19h-3v-11h3v11zm-1.5-12.268c-.966 0-1.75-.79-1.75-1.764s.784-1.764 1.75-1.764 1.75.79 1.75 1.764-.783 1.764-1.75 1.764zm13.5 12.268h-3v-5.604c0-3.368-4-3.113-4 0v5.604h-3v-11h3v1.765c1.396-2.586 7-2.777 7 2.476v6.759z" />
							</svg>
						</a>
					</div>
				</div>
			</div>
		</section>
	);
}
