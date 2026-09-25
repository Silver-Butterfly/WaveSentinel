import React, { useState, useEffect, useRef } from 'react';
import { 
    Play, 
    Pause, 
    RefreshCw, 
    ShieldCheck, 
    Zap, 
    Maximize2, 
    Minimize2,
    Scan,
    Waves,
    Compass,
    Activity
} from 'lucide-react';
import { useSimulation } from '../../simulation/SimulationContext.jsx';
import fallbackSonarImg from '../../assets/sim2d/ocean.jpg';
import { getLiveFrame } from '../../api/client.js';

export default function LiveScanView() {
    const { telemetry, missionState, backendOnline } = useSimulation();
    const canvasRef = useRef(null);
    const containerRef = useRef(null);

    const [isScanning, setIsScanning] = useState(true);
    const [scanProgress, setScanProgress] = useState(0);
    const [scanSeconds, setScanSeconds] = useState(0);
    const [isFullscreen, setIsFullscreen] = useState(false);
    const [activeDetection, setActiveDetection] = useState('SHIPWRECK');
    const [liveFrame, setLiveFrame] = useState(null);


    useEffect(() => {
        if (!backendOnline) return undefined;
        let cancelled = false;
        const poll = async () => {
            try {
                const response = await getLiveFrame();
                if (!cancelled && response) {
                    setLiveFrame(response);
                    if (response.detection?.class) setActiveDetection(String(response.detection.class).toUpperCase());
                }
            } catch {}
        };
        poll();
        const interval = setInterval(poll, 1500);
        return () => { cancelled = true; clearInterval(interval); };
    }, [backendOnline]);

    // Synthetic Side-Scan Sonar Waterfall Canvas Effect
    useEffect(() => {
        const canvas = canvasRef.current;
        if (!canvas) return;
        const ctx = canvas.getContext('2d');

        let offset = 0;
        let animationFrameId;

        const renderWaterfall = () => {
            const width = canvas.width;
            const height = canvas.height;

            // Clear background with dark navy subsea color
            ctx.fillStyle = '#020d1a';
            ctx.fillRect(0, 0, width, height);

            // Draw central nadir line (gap in side scan swath)
            ctx.strokeStyle = '#00f0ff33';
            ctx.lineWidth = 4;
            ctx.beginPath();
            ctx.moveTo(width / 2, 0);
            ctx.lineTo(width / 2, height);
            ctx.stroke();

            // Draw horizontal scrolling scanlines (synthetic waterfall)
            offset = (offset + 1.5) % 24;
            ctx.lineWidth = 1;

            for (let y = 0; y < height; y += 4) {
                const intensity = Math.sin((y + offset) * 0.1) * 0.15 + 0.15;
                ctx.strokeStyle = `rgba(0, 240, 255, ${intensity})`;
                ctx.beginPath();
                ctx.moveTo(0, y);
                ctx.lineTo(width, y);
                ctx.stroke();
            }

            // Draw acoustic backscatter intensity noise particles
            ctx.fillStyle = 'rgba(0, 240, 255, 0.25)';
            for (let i = 0; i < 60; i++) {
                const px = (Math.sin(i * 99 + offset * 0.05) * 0.5 + 0.5) * width;
                const py = (Math.cos(i * 37 + offset * 0.05) * 0.5 + 0.5) * height;
                ctx.fillRect(px, py, 2, 2);
            }

            // Draw target anomaly shadow on sonar waterfall
            ctx.fillStyle = 'rgba(239, 68, 68, 0.2)';
            ctx.strokeStyle = 'rgba(239, 68, 68, 0.8)';
            ctx.lineWidth = 2;
            const targetX = width * 0.35;
            const targetY = height * 0.35;
            ctx.fillRect(targetX, targetY, 140, 75);
            ctx.strokeRect(targetX, targetY, 140, 75);

            // Acoustic shadow behind target
            ctx.fillStyle = 'rgba(2, 6, 23, 0.7)';
            ctx.beginPath();
            ctx.moveTo(targetX + 140, targetY);
            ctx.lineTo(width * 0.8, targetY - 20);
            ctx.lineTo(width * 0.8, targetY + 95);
            ctx.lineTo(targetX + 140, targetY + 75);
            ctx.closePath();
            ctx.fill();

            animationFrameId = requestAnimationFrame(renderWaterfall);
        };

        renderWaterfall();

        return () => {
            cancelAnimationFrame(animationFrameId);
        };
    }, []);

    // 4.5 Second Sonar Scan Sequence
    const triggerScanSequence = () => {
        setIsScanning(true);
        setScanProgress(0);
        setScanSeconds(0);

        const startTime = Date.now();
        const duration = 4500;

        const interval = setInterval(() => {
            const elapsed = Date.now() - startTime;
            const currentProgress = Math.min(100, Math.floor((elapsed / duration) * 100));
            const currentSecs = (elapsed / 1000).toFixed(1);
            
            setScanProgress(currentProgress);
            setScanSeconds(currentSecs);

            if (elapsed >= duration) {
                clearInterval(interval);
                setIsScanning(false);
                setScanProgress(100);
            }
        }, 100);
    };

    useEffect(() => {
        triggerScanSequence();
    }, []);

    const toggleFullscreen = () => {
        if (!containerRef.current) return;
        if (!document.fullscreenElement) {
            containerRef.current.requestFullscreen().catch(err => console.error(err));
            setIsFullscreen(true);
        } else {
            document.exitFullscreen().catch(err => console.error(err));
            setIsFullscreen(false);
        }
    };

    return (
        <div className="space-y-6 animate-fade-in pb-12 font-mono" ref={containerRef}>
            {/* Live Scan Header Banner */}
            <div className="glass-card p-6 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
                <div className="space-y-1">
                    <div className="flex items-center gap-3">
                        <Waves className="text-cyan-400 animate-pulse" size={24} />
                        <h2 className="text-2xl font-bold text-white tracking-wide uppercase">
                            LIVE SCAN
                        </h2>
                        <span className="flex items-center gap-2 px-3 py-1 rounded-full bg-cyan-500/20 border border-cyan-400/50 text-cyan-300 text-xs font-bold uppercase tracking-wider">
                            ● SIMULATED SIDE-SCAN SONAR
                        </span>
                    </div>
                    <p className="text-slate-300 text-xs">
                        Ingesting synthetic acoustic backscatter telemetry stream for real-time subsea debris extraction.
                    </p>
                </div>

                <div className="flex items-center gap-3 text-xs">
                    <button 
                        onClick={triggerScanSequence}
                        disabled={isScanning}
                        className="px-4 py-2.5 rounded-xl bg-cyan-500 hover:bg-cyan-400 text-black font-bold uppercase tracking-wider transition-all flex items-center gap-2 cursor-pointer shadow-lg shadow-cyan-500/20 disabled:opacity-50"
                    >
                        <RefreshCw size={14} className={isScanning ? 'animate-spin' : ''} />
                        {isScanning ? 'SWEEPING...' : 'RESTART SONAR SWEEP'}
                    </button>
                </div>
            </div>

            {/* Central Viewport Stage */}
            <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
                {/* Sonar Waterfall Viewport (3 Columns) */}
                <div className="lg:col-span-3 glass-card p-4 relative flex flex-col justify-between min-h-[480px] bg-black/90 rounded-2xl overflow-hidden border border-cyan-500/30">
                    
                    {/* Canvas Sonar Viewport */}
                    <div className="relative w-full h-[440px] rounded-xl overflow-hidden bg-black flex items-center justify-center">
                        <canvas 
                            ref={canvasRef}
                            width={800}
                            height={440}
                            className="w-full h-full object-cover"
                        />

                        {/* Top HUD Telemetry Overlay */}
                        <div className="absolute top-4 left-4 right-4 flex items-center justify-between pointer-events-none text-xs">
                            <div className="flex items-center gap-2 bg-black/70 backdrop-blur-md px-3 py-1.5 rounded-lg border border-cyan-500/40 text-cyan-300 font-bold">
                                <span className="w-2.5 h-2.5 rounded-full bg-cyan-400 animate-ping"></span>
                                ● SIMULATED SSS STREAM
                            </div>

                            <div className="bg-black/70 backdrop-blur-md px-3.5 py-1.5 rounded-lg border border-white/10 text-slate-300 font-bold">
                                RANGE: {missionState.sssRange} m | {missionState.frameRate} FPS
                            </div>
                        </div>

                        {/* Sonar Frame Metadata Bounding Box Overlay */}
                        <div className="absolute bottom-16 left-8 bg-black/80 backdrop-blur-md p-3 rounded-xl border border-red-500/50 shadow-[0_0_20px_rgba(239,68,68,0.3)] pointer-events-none text-xs space-y-1">
                            <div className="flex items-center gap-2 text-red-400 font-extrabold uppercase tracking-wider">
                                <span>{liveFrame?.detection ? `${liveFrame.detection.class} ${(Number(liveFrame.detection.confidence) * (Number(liveFrame.detection.confidence) <= 1 ? 100 : 1)).toFixed(1)}%` : 'SIMULATED CONTACT'}</span>
                                <span className="bg-cyan-500 text-black px-1.5 py-0.2 rounded text-[9px]">SIMULATION</span>
                            </div>
                            <div className="text-slate-400 text-[10px]">
                                RANGE: {missionState.sssRange} m | TIME: {telemetry.timestamp} | {telemetry.position}
                            </div>
                        </div>

                        {/* Fullscreen Button */}
                        <button 
                            onClick={toggleFullscreen}
                            className="absolute top-16 right-4 p-2 rounded-lg bg-black/70 hover:bg-black/90 border border-white/20 text-white transition-all cursor-pointer z-30"
                        >
                            {isFullscreen ? <Minimize2 size={16} /> : <Maximize2 size={16} />}
                        </button>

                        {/* Scanning Sweep State Overlay */}
                        {isScanning && (
                            <div className="absolute inset-0 bg-black/50 backdrop-blur-[2px] flex flex-col items-center justify-center space-y-4 p-6 pointer-events-none">
                                <div className="w-16 h-16 rounded-full border-2 border-cyan-400/40 flex items-center justify-center relative">
                                    <div className="absolute inset-0 border-2 border-cyan-400 border-t-transparent rounded-full animate-spin"></div>
                                    <Scan size={24} className="text-cyan-400" />
                                </div>
                                <div className="text-center space-y-1">
                                    <div className="text-cyan-300 font-bold text-base tracking-widest uppercase">
                                        SWEEPING SIMULATED SSS WATERFALL...
                                    </div>
                                    <div className="text-slate-300 text-xs">
                                        Acoustic Backscatter Ingestion ({scanSeconds}s / 4.5s)
                                    </div>
                                </div>
                                <div className="w-64 bg-black/70 rounded-full h-2 overflow-hidden border border-cyan-500/40">
                                    <div className="bg-cyan-400 h-full transition-all duration-100" style={{ width: `${scanProgress}%` }}></div>
                                </div>
                            </div>
                        )}
                    </div>
                </div>

                {/* Right Panel: Live AI Diagnostics & Motion Panel */}
                <div className="lg:col-span-1 space-y-6">
                    {/* Live AI Diagnostics Card */}
                    <div className="glass-card p-6 space-y-4">
                        <div className="flex items-center justify-between border-b border-white/10 pb-3">
                            <h3 className="font-bold text-white uppercase text-sm">
                                LIVE AI DIAGNOSTICS
                            </h3>
                            <Zap size={16} className="text-cyan-400" />
                        </div>

                        <div className="space-y-2.5 text-xs">
                            <div className="flex justify-between items-center bg-black/40 p-2.5 rounded-lg border border-white/5">
                                <span className="text-slate-400">STATUS</span>
                                <span className={isScanning ? 'text-cyan-300 font-bold animate-pulse' : 'text-emerald-400 font-bold'}>
                                    {isScanning ? 'STREAMING / ANALYZING' : 'PAUSED'}
                                </span>
                            </div>

                            <div className="flex justify-between items-center bg-black/40 p-2.5 rounded-lg border border-white/5">
                                <span className="text-slate-400">INPUT</span>
                                <span className="text-cyan-400 font-bold">SIMULATED SSS</span>
                            </div>

                            <div className="flex justify-between items-center bg-black/40 p-2.5 rounded-lg border border-white/5">
                                <span className="text-slate-400">TARGET CLASS</span>
                                <span className="text-cyan-300 font-bold">{activeDetection || 'SIM CONTACT'}</span>
                            </div>

                            <div className="flex justify-between items-center bg-black/40 p-2.5 rounded-lg border border-white/5">
                                <span className="text-slate-400">CONFIDENCE</span>
                                <span className="text-cyan-300 font-bold">87%</span>
                            </div>

                            <div className="flex justify-between items-center bg-black/40 p-2.5 rounded-lg border border-white/5">
                                <span className="text-slate-400">RISK</span>
                                <span className="text-red-400 font-bold">HIGH</span>
                            </div>

                            <div className="flex justify-between items-center bg-black/40 p-2.5 rounded-lg border border-white/5">
                                <span className="text-slate-400">INPUT QUALITY</span>
                                <span className="text-emerald-400 font-bold">GOOD</span>
                            </div>

                            <div className="flex justify-between items-center bg-black/40 p-2.5 rounded-lg border border-white/5">
                                <span className="text-slate-400">DETECTIONS</span>
                                <span className="text-white font-bold">1</span>
                            </div>

                            <div className="flex justify-between items-center bg-black/40 p-2.5 rounded-lg border border-white/5">
                                <span className="text-slate-400">LATENCY</span>
                                <span className="text-cyan-300 font-bold">{telemetry.latency}</span>
                            </div>
                        </div>
                    </div>

                    {/* Vessel Motion & Motion Correction Panel */}
                    <div className="glass-card p-6 space-y-4">
                        <div className="flex items-center justify-between border-b border-white/10 pb-3">
                            <h3 className="font-bold text-white uppercase text-sm">
                                VESSEL MOTION
                            </h3>
                            <Activity size={16} className="text-cyan-400" />
                        </div>

                        <div className="grid grid-cols-2 gap-2 text-xs">
                            <div className="bg-black/40 p-2.5 rounded-lg border border-white/5">
                                <span className="text-slate-400 block text-[10px]">HEAVE</span>
                                <span className="text-cyan-300 font-bold">{telemetry.heave}</span>
                            </div>

                            <div className="bg-black/40 p-2.5 rounded-lg border border-white/5">
                                <span className="text-slate-400 block text-[10px]">PITCH</span>
                                <span className="text-white font-bold">{telemetry.pitch}</span>
                            </div>

                            <div className="bg-black/40 p-2.5 rounded-lg border border-white/5">
                                <span className="text-slate-400 block text-[10px]">ROLL</span>
                                <span className="text-white font-bold">{telemetry.roll}</span>
                            </div>

                            <div className="bg-black/40 p-2.5 rounded-lg border border-white/5">
                                <span className="text-slate-400 block text-[10px]">HEADING</span>
                                <span className="text-white font-bold">{telemetry.heading}</span>
                            </div>
                        </div>

                        {/* Motion Correction State */}
                        <div className="pt-2 space-y-2 text-xs">
                            <div className="text-slate-400 font-bold uppercase text-[11px] border-b border-white/10 pb-2">
                                MOTION CORRECTION
                            </div>

                            <div className="space-y-1.5">
                                <div className="flex justify-between text-[11px]">
                                    <span className="text-slate-400">Telemetry</span>
                                    <span className="text-emerald-400 font-bold">CONNECTED</span>
                                </div>
                                <div className="flex justify-between text-[11px]">
                                    <span className="text-slate-400">Alignment</span>
                                    <span className="text-white font-bold">8 ms</span>
                                </div>
                                <div className="flex justify-between text-[11px]">
                                    <span className="text-slate-400">Heave / Pitch / Roll</span>
                                    <span className="text-emerald-400 font-bold">READY</span>
                                </div>
                                <div className="flex justify-between text-[11px]">
                                    <span className="text-slate-400">Geometry</span>
                                    <span className="text-amber-400 font-bold">NOT VALIDATED</span>
                                </div>
                                <div className="flex justify-between text-[11px] pt-1 bg-cyan-500/10 p-2 rounded-lg border border-cyan-500/30">
                                    <span className="text-slate-300 font-bold">Correction</span>
                                    <span className="text-cyan-300 font-extrabold uppercase">METADATA ONLY</span>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            {/* Bottom Frame Telemetry Strip */}
            <div className="glass-card p-4 grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-8 gap-4 text-xs">
                <div className="space-y-0.5">
                    <div className="text-slate-400 text-[10px] uppercase">Frame ID</div>
                    <div className="text-white font-bold">{telemetry.frameId}</div>
                </div>
                <div className="space-y-0.5">
                    <div className="text-slate-400 text-[10px] uppercase">Timestamp</div>
                    <div className="text-white font-bold">{telemetry.timestamp}</div>
                </div>
                <div className="space-y-0.5">
                    <div className="text-slate-400 text-[10px] uppercase">Position</div>
                    <div className="text-cyan-300 font-bold text-[11px] truncate">
                        {telemetry.position} <span className="text-[9px] bg-cyan-500/20 px-1 rounded border border-cyan-400/40">SIMULATED</span>
                    </div>
                </div>
                <div className="space-y-0.5">
                    <div className="text-slate-400 text-[10px] uppercase">Sonar Range</div>
                    <div className="text-white font-bold">{missionState.sssRange} m</div>
                </div>
                <div className="space-y-0.5">
                    <div className="text-slate-400 text-[10px] uppercase">Depth</div>
                    <div className="text-white font-bold">{telemetry.depth}</div>
                </div>
                <div className="space-y-0.5">
                    <div className="text-slate-400 text-[10px] uppercase">Vessel Speed</div>
                    <div className="text-white font-bold">{telemetry.vesselSpeed}</div>
                </div>
                <div className="space-y-0.5">
                    <div className="text-slate-400 text-[10px] uppercase">Heading</div>
                    <div className="text-white font-bold">{telemetry.heading}</div>
                </div>
                <div className="space-y-0.5">
                    <div className="text-slate-400 text-[10px] uppercase">Latency</div>
                    <div className="text-emerald-400 font-bold">{telemetry.latency}</div>
                </div>
            </div>
        </div>
    );
}
