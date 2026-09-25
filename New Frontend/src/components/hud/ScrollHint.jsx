import React from 'react';
import './ScrollHint.css';

export default function ScrollHint({ progress = 0.0 }) {
	const isHidden = progress > 0.05;

	return (
		<div id="scroll-hint" style={{ opacity: isHidden ? 0 : 1 }}>
			<span className="scroll-text">Scroll down to dive IN</span>
		</div>
	);
}
