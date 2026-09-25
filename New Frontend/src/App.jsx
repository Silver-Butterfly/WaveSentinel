import React, { useState } from 'react';
import HomePage from './pages/home/HomePage.jsx';
import DashboardApp from './pages/dashboard/DashboardApp.jsx';
import WebPage from './pages/webpage/WebPage.jsx';
import NewModulePage from './pages/newmodule/NewModulePage.jsx';
import Sim2DPage from './pages/sim2d/Sim2DPage.jsx';
import './App.css';

export default function App() {
	const [view, setView] = useState('home');
	const [homeTarget, setHomeTarget] = useState(null);

	if (view === 'dashboard') {
		return <DashboardApp onBack={() => { setHomeTarget(null); setView('home'); }} />;
	}

	if (view === 'webpage') {
		return <WebPage 
					onBack={() => { setHomeTarget(null); setView('home'); }} 
					onContinue={() => { setHomeTarget('zone-cube'); setView('home'); }}
				/>;
	}

	if (view === 'newmodule') {
		return <NewModulePage onBack={() => { setHomeTarget(null); setView('home'); }} />;
	}

	if (view === 'sim2d') {
		return <Sim2DPage onBack={() => { setHomeTarget('zone-cube'); setView('home'); }} />;
	}

	return (
		<main className="simulation-app">
			<HomePage 
				onOpenDashboard={() => setView('dashboard')} 
				onOpenWebpage={() => setView('webpage')}
				onOpenNewModule={() => setView('newmodule')}
				onOpenSim2D={() => setView('sim2d')}
				targetSection={homeTarget}
			/>
		</main>
	);
}

