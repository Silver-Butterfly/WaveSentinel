import React from 'react';
import { 
    Play, 
    Pause, 
    Square, 
    RotateCcw, 
    Radio, 
    Navigation, 
    Compass, 
    Layers, 
    Activity, 
    AlertTriangle, 
    ShieldCheck, 
    Clock,
    Sliders,
    Cpu
} from 'lucide-react';
import { useSimulation } from '../../simulation/SimulationContext.jsx';

export default function MissionControlView() {
    const { 
        missionState,
        detections,
        startMission,
        pauseMission,
        resumeMission,
        endMission
    } = useSimulation();

    const simulatedEvents = detections.slice(0, 5).map((item) => ({
        time: item.timestamp,
        title: `${item.class} contact logged`,
        line: missionState.surveyLine,
        risk: item.risk || 'UNASSESSED',
        color: item.class === 'Mine' ? 'text-red-500 border-red-600 font-extrabold' :
            item.class === 'Shipwreck' ? 'text-red-400 border-red-500' :
            item.class === 'Ghost Net' ? 'text-purple-400 border-purple-400' :
            'text-amber-400 border-amber-400'
    }));

    const progressPercentage = Math.min(100, Math.round((missionState.framesProcessed / missionState.framesTotal) * 100));
    const remainingFrames = missionState.framesTotal - missionState.framesProcessed;

    return (
        <div className="space-y-8 animate-fade-in pb-12 font-mono">
            {/* Mission Control Header Banner */}
            <div className="glass-card p-6 flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
                <div className="space-y-1">
                    <div className="flex items-center gap-3">
                        <Radio className="text-cyan-400 animate-pulse" size={24} />
                        <h2 className="text-2xl font-bold text-white tracking-wide uppercase">
                            MISSION CONTROL
                        </h2>
                        <span className="px-3 py-1 rounded-full bg-cyan-500/20 border border-cyan-400/50 text-cyan-300 text-xs font-bold uppercase tracking-wider">
                            ● SIMULATION MODE
                        </span>
                    </div>
                    <p className="text-slate-300 text-xs">
                        Central Command & Mission Execution Control Desk • Designation: {missionState.missionId}
                    </p>
                </div>

                <div className="flex items-center gap-3 text-xs">
                    <div className="bg-black/50 px-3.5 py-2 rounded-xl border border-white/10 flex items-center gap-2">
                        <span className="text-slate-400">SURVEY AREA:</span>
                        <span className="text-cyan-300 font-bold">{missionState.surveyArea}</span>
                    </div>
                    <div className="bg-black/50 px-3.5 py-2 rounded-xl border border-white/10 flex items-center gap-2">
                        <span className="text-slate-400">LINE:</span>
                        <span className="text-white font-bold">{missionState.surveyLine}</span>
                    </div>
                </div>
            </div>

            {/* Mission Status & Control Actions Bar */}
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                {/* Status Indicator Card */}
                <div className="glass-card p-6 flex flex-col justify-between space-y-4">
                    <div className="flex items-center justify-between border-b border-white/10 pb-3">
                        <span className="text-slate-400 text-xs uppercase tracking-wider">MISSION STATUS</span>
                        <span className={`px-2.5 py-1 rounded text-xs font-bold uppercase tracking-wider ${
                            missionState.status === 'ACTIVE' ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/40 animate-pulse' :
                            missionState.status === 'PAUSED' ? 'bg-amber-500/20 text-amber-300 border border-amber-500/40' :
                            'bg-slate-500/20 text-slate-300 border border-slate-500/40'
                        }`}>
                            ● {missionState.status}
                        </span>
                    </div>

                    <div className="space-y-2 text-xs">
                        <div className="flex justify-between items-center text-slate-300">
                            <span>OPERATIONAL MODE</span>
                            <span className="text-cyan-400 font-bold bg-cyan-500/10 px-2 py-0.5 rounded border border-cyan-500/30">
                                {missionState.operationalMode}
                            </span>
                        </div>
                        <div className="flex justify-between items-center text-slate-300">
                            <span>SURVEY SECTOR</span>
                            <span className="text-white font-bold">{missionState.surveyArea}</span>
                        </div>
                        <div className="flex justify-between items-center text-slate-300">
                            <span>ACTIVE LINE</span>
                            <span className="text-white font-bold">{missionState.surveyLine}</span>
                        </div>
                    </div>
                </div>

                {/* Control Action Buttons Card */}
                <div className="lg:col-span-2 glass-card p-6 space-y-4">
                    <div className="text-slate-400 text-xs uppercase tracking-wider border-b border-white/10 pb-3">
                        MISSION COMMAND CONTROLS
                    </div>

                    <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 pt-1">
                        <button
                            onClick={startMission}
                            disabled={missionState.status === 'ACTIVE'}
                            className="py-3 px-4 rounded-xl bg-emerald-500/20 hover:bg-emerald-500/30 border border-emerald-400/50 text-emerald-300 font-bold text-xs uppercase transition-all flex items-center justify-center gap-2 cursor-pointer disabled:opacity-40 disabled:cursor-not-allowed"
                        >
                            <Play size={16} /> START
                        </button>

                        <button
                            onClick={pauseMission}
                            disabled={missionState.status !== 'ACTIVE'}
                            className="py-3 px-4 rounded-xl bg-amber-500/20 hover:bg-amber-500/30 border border-amber-400/50 text-amber-300 font-bold text-xs uppercase transition-all flex items-center justify-center gap-2 cursor-pointer disabled:opacity-40 disabled:cursor-not-allowed"
                        >
                            <Pause size={16} /> PAUSE
                        </button>

                        <button
                            onClick={resumeMission}
                            disabled={missionState.status === 'ACTIVE'}
                            className="py-3 px-4 rounded-xl bg-cyan-500/20 hover:bg-cyan-500/30 border border-cyan-400/50 text-cyan-300 font-bold text-xs uppercase transition-all flex items-center justify-center gap-2 cursor-pointer disabled:opacity-40 disabled:cursor-not-allowed"
                        >
                            <RotateCcw size={16} /> RESUME
                        </button>

                        <button
                            onClick={endMission}
                            disabled={missionState.status === 'ENDED'}
                            className="py-3 px-4 rounded-xl bg-red-500/20 hover:bg-red-500/30 border border-red-400/50 text-red-300 font-bold text-xs uppercase transition-all flex items-center justify-center gap-2 cursor-pointer disabled:opacity-40 disabled:cursor-not-allowed"
                        >
                            <Square size={16} /> END
                        </button>
                    </div>

                    <div className="text-[11px] text-slate-400 text-center italic pt-1">
                        Note: Mission controls trigger simulated telemetry sequence updates in DEMO MODE.
                    </div>
                </div>
            </div>

            {/* Mission Configuration & Parameters */}
            <div className="glass-card p-6 space-y-4">
                <div className="flex items-center justify-between border-b border-white/10 pb-3">
                    <div className="flex items-center gap-2">
                        <Sliders className="text-cyan-400" size={18} />
                        <h3 className="font-bold text-white uppercase text-sm">
                            SIMULATED MISSION CONFIGURATION
                        </h3>
                    </div>
                    <span className="text-xs text-slate-400">All values simulated</span>
                </div>

                <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-4 text-xs">
                    <div className="p-3.5 rounded-xl bg-black/40 border border-white/5 space-y-1">
                        <div className="text-slate-400 text-[10px] uppercase">SSS RANGE</div>
                        <div className="text-white font-bold text-base">{missionState.sssRange} m</div>
                        <div className="text-[9px] text-cyan-400 font-semibold">SIMULATED</div>
                    </div>

                    <div className="p-3.5 rounded-xl bg-black/40 border border-white/5 space-y-1">
                        <div className="text-slate-400 text-[10px] uppercase">SIMULATED SWATH</div>
                        <div className="text-white font-bold text-base">{missionState.simulatedSwath} m</div>
                        <div className="text-[9px] text-cyan-400 font-semibold">SIMULATED</div>
                    </div>

                    <div className="p-3.5 rounded-xl bg-black/40 border border-white/5 space-y-1">
                        <div className="text-slate-400 text-[10px] uppercase">FRAME RATE</div>
                        <div className="text-cyan-300 font-bold text-base">{missionState.frameRate} FPS</div>
                        <div className="text-[9px] text-cyan-400 font-semibold">SIMULATED</div>
                    </div>

                    <div className="p-3.5 rounded-xl bg-black/40 border border-white/5 space-y-1">
                        <div className="text-slate-400 text-[10px] uppercase">VESSEL SPEED</div>
                        <div className="text-white font-bold text-base">{missionState.vesselSpeed} kts</div>
                        <div className="text-[9px] text-cyan-400 font-semibold">SIMULATED</div>
                    </div>

                    <div className="p-3.5 rounded-xl bg-black/40 border border-white/5 space-y-1">
                        <div className="text-slate-400 text-[10px] uppercase">HEADING</div>
                        <div className="text-white font-bold text-base">{missionState.heading}°</div>
                        <div className="text-[9px] text-cyan-400 font-semibold">SIMULATED</div>
                    </div>

                    <div className="p-3.5 rounded-xl bg-black/40 border border-white/5 space-y-1">
                        <div className="text-slate-400 text-[10px] uppercase">DEPTH</div>
                        <div className="text-white font-bold text-base">{missionState.depth} m</div>
                        <div className="text-[9px] text-cyan-400 font-semibold">SIMULATED</div>
                    </div>
                </div>
            </div>

            {/* Mission Progress & Metrics */}
            <div className="glass-card p-6 space-y-5">
                <div className="flex items-center justify-between border-b border-white/10 pb-3">
                    <h3 className="font-bold text-white uppercase text-sm">
                        MISSION PROGRESS & METRICS
                    </h3>
                    <span className="text-cyan-400 font-bold text-sm">{progressPercentage}% COMPLETE</span>
                </div>

                <div className="space-y-2">
                    <div className="w-full bg-black/60 rounded-full h-3 overflow-hidden border border-white/10">
                        <div 
                            className="bg-cyan-400 h-full transition-all duration-300 shadow-[0_0_12px_#00F0FF]" 
                            style={{ width: `${progressPercentage}%` }}
                        ></div>
                    </div>
                </div>

                <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 pt-2 text-xs">
                    <div className="p-4 rounded-xl bg-black/40 border border-white/5">
                        <div className="text-slate-400 text-[10px] uppercase">Frames Processed</div>
                        <div className="text-2xl font-bold text-white mt-1">{missionState.framesProcessed.toLocaleString()}</div>
                    </div>

                    <div className="p-4 rounded-xl bg-black/40 border border-white/5">
                        <div className="text-slate-400 text-[10px] uppercase">Remaining Frames</div>
                        <div className="text-2xl font-bold text-slate-300 mt-1">{remainingFrames.toLocaleString()}</div>
                    </div>

                    <div className="p-4 rounded-xl bg-black/40 border border-white/5">
                        <div className="text-slate-400 text-[10px] uppercase">Detections Logged</div>
                        <div className="text-2xl font-bold text-cyan-300 mt-1">{missionState.detectionsCount}</div>
                    </div>

                    <div className="p-4 rounded-xl bg-black/40 border border-white/5">
                        <div className="text-slate-400 text-[10px] uppercase">High Risk Objects</div>
                        <div className="text-2xl font-bold text-red-400 mt-1">{missionState.highRiskCount}</div>
                    </div>
                </div>
            </div>

            {/* Compact Mission Event Feed */}
            <div className="glass-card p-6 space-y-4">
                <div className="flex items-center justify-between border-b border-white/10 pb-3">
                    <h3 className="font-bold text-white uppercase text-sm">
                        COMPACT MISSION EVENT FEED
                    </h3>
                    <span className="text-xs text-slate-400">Live Simulated Activity Stream</span>
                </div>

                <div className="space-y-3 text-xs">
                    {simulatedEvents.map((evt, idx) => (
                        <div key={idx} className="flex items-center justify-between p-3.5 rounded-xl bg-black/40 border border-white/5 hover:border-white/15 transition-all">
                            <div className="flex items-center gap-4">
                                <span className="text-slate-400 font-bold">{evt.time}</span>
                                <span className="text-white font-semibold">{evt.title}</span>
                            </div>

                            <div className="flex items-center gap-3">
                                <span className="text-slate-400 text-[11px]">{evt.line}</span>
                                <span className={`px-2.5 py-0.5 rounded text-[10px] font-bold ${evt.color} bg-white/5 border`}>
                                    {evt.risk}
                                </span>
                            </div>
                        </div>
                    ))}
                </div>
            </div>
        </div>
    );
}
