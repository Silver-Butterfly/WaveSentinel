import React, { useState, useEffect } from 'react';
import { SimulationProvider, useSimulation } from '../../simulation/SimulationContext.jsx';
import DashboardSidebar from './DashboardSidebar.jsx';
import OverviewView from './OverviewView.jsx';
import MissionControlView from './MissionControlView.jsx';
import LiveScanView from './LiveScanView.jsx';
import UploadView from './UploadView.jsx';
import BatchAnalysisView from './BatchAnalysisView.jsx';
import DetectionArchiveView from './DetectionArchiveView.jsx';
import ReviewTriageView from './ReviewTriageView.jsx';
import SurveyMapView from './SurveyMapView.jsx';
import EvidenceReportsView from './EvidenceReportsView.jsx';
import SensorTelemetryView from './SensorTelemetryView.jsx';
import ModelSystemView from './ModelSystemView.jsx';
import SettingsView from './SettingsView.jsx';
import './DashboardApp.css';

function DashboardContent({ onBack }) {
    const [activeTab, setActiveTab] = useState('dashboard');
    const { backendOnline, runtimeBenchmark, telemetry } = useSimulation();

    useEffect(() => {
        window.scrollTo(0, 0);
    }, []);

    const handleSelectDetection = (detection) => {
        setActiveTab('archive');
    };

    const getTabLabel = (tab) => {
        switch (tab) {
            case 'dashboard': return 'Dashboard Overview';
            case 'missioncontrol': return 'Mission Control';
            case 'livescan': return 'Live Side-Scan Sonar';
            case 'uploadscan': return 'Upload Scan';
            case 'batchanalysis': return 'Batch Analysis';
            case 'archive': return 'Detection Archive';
            case 'reviewtriage': return 'Review & Triage';
            case 'map': return 'Survey Map';
            case 'evidencereports': return 'Evidence & Reports';
            case 'sensortelemetry': return 'Sensor & Telemetry';
            case 'modelsystem': return 'Model & System';
            case 'settings': return 'Settings';
            default: return tab.toUpperCase();
        }
    };

    return (
        <div className="dashboard-root min-h-screen flex bg-[#020617] text-white">
            {/* Sidebar Navigation */}
            <DashboardSidebar 
                activeTab={activeTab} 
                onSelectTab={setActiveTab} 
                onBack={onBack}
            />

            {/* Main Workstation View Area */}
            <div className="flex-1 flex flex-col min-h-screen overflow-x-hidden relative">
                {/* Ambient Subsea Backlight Effects */}
                <div className="absolute top-0 left-1/4 w-96 h-96 bg-cyan-500/10 rounded-full blur-[140px] pointer-events-none"></div>
                <div className="absolute top-1/3 right-10 w-96 h-96 bg-emerald-500/10 rounded-full blur-[160px] pointer-events-none"></div>

                {/* Top Header Bar with Global Persistent Simulation Status */}
                <header className="sticky top-0 z-20 backdrop-blur-xl bg-black/60 border-b border-white/10 px-8 py-3.5 flex flex-wrap items-center justify-between gap-4">
                    <div className="flex items-center gap-3 font-mono text-xs">
                        <span className="text-white font-bold tracking-widest uppercase">
                            WAVESENTINEL WORKSTATION
                        </span>
                        <span className="text-slate-500">|</span>
                        <span className="text-cyan-400 font-semibold uppercase">
                            {getTabLabel(activeTab)}
                        </span>
                    </div>

                    {/* Global Simulation State Header Indicator */}
                    <div className="flex items-center gap-4 font-mono text-xs">
                        <div className={`flex items-center gap-2 bg-black/50 px-3 py-1.5 rounded-lg border font-bold ${backendOnline ? 'border-emerald-500/40 text-emerald-400' : 'border-amber-500/40 text-amber-300'}`}>
                            <span className={`w-2 h-2 rounded-full ${backendOnline ? 'bg-emerald-400 animate-pulse' : 'bg-amber-400'}`}></span>
                            {backendOnline ? '● BACKEND ONLINE' : '● BACKEND OFFLINE'}
                        </div>

                        <div className="hidden sm:flex items-center gap-3 bg-black/50 px-3.5 py-1.5 rounded-lg border border-cyan-500/40 text-cyan-300 font-bold">
                            <span>MODE: <strong className="text-cyan-400">SIMULATION</strong></span>
                            <span className="text-slate-600">|</span>
                            <span>SSS: <strong className="text-cyan-400">SIMULATED</strong></span>
                            <span className="text-slate-600">|</span>
                            <span>TELEMETRY: <strong className="text-cyan-400">{telemetry?.source?.toUpperCase() || 'SIMULATED'}</strong></span>
                            <span className="text-slate-600">|</span>
                            <span>TRT P50: <strong className="text-cyan-400">{runtimeBenchmark ? `${runtimeBenchmark.p50_ms.toFixed(2)} ms` : '—'}</strong></span>
                        </div>
                    </div>
                </header>

                {/* Dynamic Page Content */}
                <main className="flex-1 p-6 md:p-8 max-w-7xl w-full mx-auto">
                    {activeTab === 'dashboard' && (
                        <OverviewView 
                            onNavigateTab={setActiveTab} 
                            onSelectDetection={handleSelectDetection}
                        />
                    )}
                    {activeTab === 'missioncontrol' && <MissionControlView />}
                    {activeTab === 'livescan' && <LiveScanView />}
                    {activeTab === 'uploadscan' && <UploadView />}
                    {activeTab === 'batchanalysis' && (
                        <BatchAnalysisView onSelectDetection={handleSelectDetection} />
                    )}
                    {activeTab === 'archive' && <DetectionArchiveView />}
                    {activeTab === 'reviewtriage' && <ReviewTriageView />}
                    {activeTab === 'map' && (
                        <SurveyMapView onNavigateTab={setActiveTab} />
                    )}
                    {activeTab === 'evidencereports' && <EvidenceReportsView />}
                    {activeTab === 'sensortelemetry' && <SensorTelemetryView />}
                    {activeTab === 'modelsystem' && <ModelSystemView />}
                    {activeTab === 'settings' && <SettingsView />}
                </main>
            </div>
        </div>
    );
}

export default function DashboardApp({ onBack }) {
    return (
        <SimulationProvider>
            <DashboardContent onBack={onBack} />
        </SimulationProvider>
    );
}
