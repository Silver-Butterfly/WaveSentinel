import React from 'react';
import { 
    Activity, 
    ShieldCheck, 
    AlertTriangle, 
    Cpu, 
    ArrowRight, 
    Radio,
    Clock,
    FileText,
    Navigation,
    Compass
} from 'lucide-react';
import { useSimulation } from '../../simulation/SimulationContext.jsx';

export default function OverviewView({ onNavigateTab, onSelectDetection }) {
    const { missionState, detections, classCounts } = useSimulation();

    const distribution = [
        { label: 'SHIPWRECK', count: classCounts.SHIPWRECK || 0, color: 'bg-red-500' },
        { label: 'PIPE', count: classCounts.PIPE || 0, color: 'bg-amber-500' },
        { label: 'MINE', count: classCounts.MINE || 0, color: 'bg-cyan-500' },
        { label: 'GHOST NET', count: classCounts.GHOST_NET || 0, color: 'bg-purple-500' }
    ];

    const distributionTotal = distribution.reduce((sum, item) => sum + item.count, 0) || 1;
    const recentLogs = detections.slice(0, 4).map((item) => ({
        time: String(item.timestamp || '').slice(0, 5),
        title: `${item.class} detected`,
        frame: item.frame_id,
        risk: item.risk || 'UNASSESSED',
        color: item.class === 'Mine' ? 'text-red-500 border-red-600 font-extrabold' :
            item.class === 'Shipwreck' ? 'text-red-400 border-red-500' :
            item.class === 'Ghost Net' ? 'text-purple-400 border-purple-400' :
            'text-amber-400 border-amber-400'
    }));

    const progressPercent = Math.round((missionState.framesProcessed / missionState.framesTotal) * 100);

    return (
        <div className="space-y-8 animate-fade-in pb-12 font-mono">
            {/* Header Banner */}
            <div className="glass-card p-6 flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
                <div className="space-y-1">
                    <div className="flex items-center gap-3">
                        <h2 className="text-2xl font-bold text-white tracking-wide">
                            WaveSentinel Command Centre
                        </h2>
                        <span className="px-2.5 py-0.5 rounded-full bg-cyan-500/10 border border-cyan-400/40 text-cyan-300 text-xs font-semibold">
                            Mission {missionState.missionId}
                        </span>
                    </div>
                    <p className="text-slate-300 text-xs">
                        Underwater Sonar Intelligence & Marine Debris Detection Operational Workstation
                    </p>
                </div>

                <div className="flex items-center gap-4 text-xs">
                    <div className="flex items-center gap-2 bg-black/40 px-3 py-1.5 rounded-lg border border-emerald-500/30 text-emerald-400 font-bold">
                        <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span>
                        System ● Online
                    </div>
                    <div className="flex items-center gap-2 bg-black/40 px-3 py-1.5 rounded-lg border border-cyan-500/30 text-cyan-300 font-bold">
                        <Cpu size={14} />
                        Mode ● Simulation
                    </div>
                </div>
            </div>

            {/* Compact Operational Status Strip */}
            <div className="glass-card p-4 grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-4 text-xs border border-cyan-500/20">
                <div className="flex items-center justify-between bg-black/40 p-2.5 rounded-xl border border-white/5">
                    <span className="text-slate-400 text-[11px]">SSS STREAM</span>
                    <span className="text-cyan-400 font-bold">SIMULATED</span>
                </div>
                <div className="flex items-center justify-between bg-black/40 p-2.5 rounded-xl border border-white/5">
                    <span className="text-slate-400 text-[11px]">TELEMETRY</span>
                    <span className="text-cyan-400 font-bold">SIMULATED</span>
                </div>
                <div className="flex items-center justify-between bg-black/40 p-2.5 rounded-xl border border-white/5">
                    <span className="text-slate-400 text-[11px]">MOTION STATE</span>
                    <span className="text-emerald-400 font-bold">AVAILABLE</span>
                </div>
                <div className="flex items-center justify-between bg-black/40 p-2.5 rounded-xl border border-white/5">
                    <span className="text-slate-400 text-[11px]">MODEL</span>
                    <span className="text-emerald-400 font-bold">READY</span>
                </div>
                <div className="flex items-center justify-between bg-black/40 p-2.5 rounded-xl border border-white/5">
                    <span className="text-slate-400 text-[11px]">POSITION</span>
                    <span className="text-cyan-400 font-bold">SIMULATED</span>
                </div>
            </div>

            {/* Four KPI Cards */}
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
                <div className="glass-card p-6 flex flex-col justify-between space-y-4">
                    <div className="flex items-center justify-between">
                        <span className="text-slate-400 text-xs uppercase tracking-wider">ANALYZED FRAMES</span>
                        <div className="p-2 rounded-lg bg-cyan-500/10 border border-cyan-500/30 text-cyan-400">
                            <Activity size={18} />
                        </div>
                    </div>
                    <div>
                        <div className="text-3xl font-bold text-white">{missionState.framesProcessed.toLocaleString()}</div>
                        <div className="text-xs text-slate-300 mt-1">Processed Sonar Frames</div>
                    </div>
                </div>

                <div className="glass-card p-6 flex flex-col justify-between space-y-4">
                    <div className="flex items-center justify-between">
                        <span className="text-slate-400 text-xs uppercase tracking-wider">DETECTIONS</span>
                        <div className="p-2 rounded-lg bg-emerald-500/10 border border-emerald-500/30 text-emerald-400">
                            <ShieldCheck size={18} />
                        </div>
                    </div>
                    <div>
                        <div className="text-3xl font-bold text-cyan-300">{missionState.detectionsCount}</div>
                        <div className="text-xs text-emerald-400 mt-1">Submerged Contacts</div>
                    </div>
                </div>

                <div className="glass-card p-6 flex flex-col justify-between space-y-4">
                    <div className="flex items-center justify-between">
                        <span className="text-slate-400 text-xs uppercase tracking-wider">SIMULATED PRIORITY FLAGS</span>
                        <div className="p-2 rounded-lg bg-red-500/10 border border-red-500/30 text-red-400">
                            <AlertTriangle size={18} />
                        </div>
                    </div>
                    <div>
                        <div className="text-3xl font-bold text-red-400">{missionState.highRiskCount}</div>
                        <div className="text-xs text-slate-300 mt-1">Demo priority labels only</div>
                    </div>
                </div>

                <div className="glass-card p-6 flex flex-col justify-between space-y-4">
                    <div className="flex items-center justify-between">
                        <span className="text-slate-400 text-xs uppercase tracking-wider">AVG FRAME LATENCY</span>
                        <div className="p-2 rounded-lg bg-indigo-500/10 border border-indigo-500/30 text-indigo-400">
                            <Cpu size={18} />
                        </div>
                    </div>
                    <div>
                        <div className="text-3xl font-bold text-white">{missionState.avgLatencyMs} ms</div>
                        <div className="text-xs text-indigo-300 mt-1">YOLOv8 Runtime Latency</div>
                    </div>
                </div>
            </div>

            {/* Current Mission & Recent Activity Grid */}
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
                {/* Left Column: Current Mission & Distribution */}
                <div className="space-y-8 lg:col-span-1">
                    {/* Current Mission Card */}
                    <div className="glass-card p-6 space-y-5">
                        <div className="flex items-center justify-between border-b border-white/10 pb-3">
                            <h3 className="font-bold text-white uppercase text-sm">
                                CURRENT MISSION
                            </h3>
                            <span className="px-2 py-0.5 rounded bg-emerald-500/20 text-emerald-300 text-[10px] font-bold uppercase">
                                {missionState.status}
                            </span>
                        </div>

                        <div className="space-y-3.5 text-xs">
                            <div className="flex justify-between items-center bg-black/40 p-2.5 rounded-lg border border-white/5">
                                <span className="text-slate-400">MISSION</span>
                                <span className="text-white font-bold">{missionState.missionId}</span>
                            </div>

                            <div className="flex justify-between items-center bg-black/40 p-2.5 rounded-lg border border-white/5">
                                <span className="text-slate-400">MODE</span>
                                <span className="text-cyan-400 font-bold">SIMULATION</span>
                            </div>

                            <div className="flex justify-between items-center bg-black/40 p-2.5 rounded-lg border border-white/5">
                                <span className="text-slate-400">SURVEY AREA</span>
                                <span className="text-cyan-300 font-bold">{missionState.surveyArea}</span>
                            </div>

                            <div className="space-y-2 pt-1">
                                <div className="flex justify-between text-slate-300">
                                    <span>FRAMES</span>
                                    <span className="font-bold text-white">{missionState.framesProcessed.toLocaleString()} / {missionState.framesTotal.toLocaleString()}</span>
                                </div>
                                <div className="w-full bg-black/60 rounded-full h-2.5 overflow-hidden border border-white/10">
                                    <div 
                                        className="bg-cyan-400 h-full shadow-[0_0_10px_#00F0FF]" 
                                        style={{ width: `${progressPercent}%` }}
                                    ></div>
                                </div>
                                <div className="text-right text-[11px] text-cyan-400 font-bold">PROGRESS: {progressPercent}%</div>
                            </div>

                            <div className="grid grid-cols-2 gap-3 pt-2 border-t border-white/10 text-[11px]">
                                <div className="p-2 rounded-lg bg-black/40 border border-white/5">
                                    <span className="text-slate-400 block text-[10px]">DETECTIONS</span>
                                    <span className="text-cyan-300 font-bold text-sm">{missionState.detectionsCount}</span>
                                </div>
                                <div className="p-2 rounded-lg bg-black/40 border border-white/5">
                                    <span className="text-slate-400 block text-[10px]">HIGH RISK</span>
                                    <span className="text-red-400 font-bold text-sm">{missionState.highRiskCount}</span>
                                </div>
                            </div>
                        </div>
                    </div>

                    {/* Detection Distribution */}
                    <div className="glass-card p-6 space-y-4">
                        <h3 className="font-bold text-white uppercase text-sm border-b border-white/10 pb-3">
                            Detection Distribution
                        </h3>

                        <div className="space-y-3 text-xs">
                            {distribution.map((item, idx) => (
                                <div key={idx} className="space-y-1">
                                    <div className="flex justify-between text-slate-300 text-[11px]">
                                        <span>{item.label}</span>
                                        <span className="font-bold text-white">{item.count}</span>
                                    </div>
                                    <div className="w-full bg-black/50 rounded-full h-2 overflow-hidden border border-white/5">
                                        <div className={`${item.color} h-full`} style={{ width: `${Math.round((item.count / distributionTotal) * 100)}%` }}></div>
                                    </div>
                                </div>
                            ))}
                        </div>
                    </div>
                </div>

                {/* Right Column: Recent Activity Feed */}
                <div className="space-y-8 lg:col-span-2">
                    <div className="glass-card p-6 space-y-4">
                        <div className="flex items-center justify-between border-b border-white/10 pb-3">
                            <h3 className="font-bold text-white uppercase text-sm">
                                Recent Mission Activity
                            </h3>
                            <button 
                                onClick={() => onNavigateTab('archive')}
                                className="text-xs text-cyan-400 hover:underline flex items-center gap-1 cursor-pointer"
                            >
                                VIEW ALL IN ARCHIVE <ArrowRight size={12} />
                            </button>
                        </div>

                        <div className="space-y-3 text-xs">
                            {recentLogs.map((log, i) => (
                                <div key={i} className="flex items-center justify-between p-4 rounded-xl bg-black/40 border border-white/5 hover:border-white/15 transition-all">
                                    <div className="flex items-center gap-4">
                                        <span className="text-slate-400 text-[11px] font-bold">{log.time}</span>
                                        <span className="text-white font-semibold text-sm">{log.title}</span>
                                    </div>
                                    <div className="flex items-center gap-3">
                                        <span className="text-slate-400 text-[11px]">{log.frame}</span>
                                        <span className={`px-2.5 py-1 rounded text-[10px] font-bold ${log.color} bg-white/5 border`}>
                                            {log.risk}
                                        </span>
                                    </div>
                                </div>
                            ))}
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
}
