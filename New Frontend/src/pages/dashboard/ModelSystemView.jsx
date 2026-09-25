import React from 'react';
import { useSimulation } from '../../simulation/SimulationContext.jsx';
import { Cpu, CheckCircle2, Server, Activity, Layers, Zap, Shield, Sliders, AlertCircle } from 'lucide-react';

export default function ModelSystemView() {
    const { runtimeBenchmark, backendOnline } = useSimulation();

    const pipelineSteps = [
        'SIDE-SCAN SONAR',
        'PREPROCESSING',
        'TENSORRT FP32',
        'OBJECT + CONFIDENCE',
        'HUMAN REVIEW',
        'WAVESENTINEL'
    ];

    return (
        <div className="space-y-8 animate-fade-in pb-12 font-mono">
            {/* Header */}
            <div className="glass-card p-6 flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
                <div className="space-y-1">
                    <div className="flex items-center gap-3">
                        <Cpu className="text-cyan-400" size={24} />
                        <h2 className="text-2xl font-bold text-white tracking-wide uppercase">
                            MODEL & SYSTEM TRANSPARENCY
                        </h2>
                        <span className="px-3 py-1 rounded-full bg-cyan-500/20 border border-cyan-400/50 text-cyan-300 text-xs font-bold uppercase">
                            AI INFERENCE SPECIFICATIONS
                        </span>
                    </div>
                    <p className="text-slate-300 text-xs">
                        Subsea Debris Detection Neural Net Specifications, Frozen Calibrations & Hardware Diagnostics
                    </p>
                </div>
            </div>

            {/* Model Spec & Frozen Thresholds Grid */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                {/* Active Model Specification Card */}
                <div className="glass-card p-6 space-y-5">
                    <div className="flex items-center justify-between border-b border-white/10 pb-3">
                        <h3 className="font-bold text-white uppercase text-sm">
                            MODEL
                        </h3>
                        <span className="flex items-center gap-1.5 px-2.5 py-0.5 rounded bg-emerald-500/20 text-emerald-400 text-xs font-bold">
                            <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span>
                            Phase-D Robust-V2-HN
                        </span>
                    </div>

                    <div className="space-y-3 text-xs">
                        <div className="flex justify-between items-center bg-black/40 p-3 rounded-xl border border-white/5">
                            <span className="text-slate-400">DEPLOYMENT BACKEND</span>
                            <span className="text-white font-bold text-sm">TensorRT FP32</span>
                        </div>
                        <div className="flex justify-between items-center bg-black/40 p-3 rounded-xl border border-white/5">
                            <span className="text-slate-400">CLASSES</span>
                            <span className="text-cyan-300 font-bold">4 CLASSES</span>
                        </div>
                        <div className="flex justify-between items-center bg-black/40 p-3 rounded-xl border border-white/5">
                            <span className="text-slate-400">INPUT RESOLUTION</span>
                            <span className="text-white font-bold">640 × 640</span>
                        </div>

                        <div className="pt-2">
                            <div className="text-slate-400 text-[11px] uppercase mb-2">Class Taxonomy:</div>
                            <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 text-center text-[11px]">
                                <span className="bg-black/60 py-1.5 px-2 rounded border border-white/10 text-white font-bold">PIPE</span>
                                <span className="bg-black/60 py-1.5 px-2 rounded border border-white/10 text-white font-bold">SHIPWRECK</span>
                                <span className="bg-black/60 py-1.5 px-2 rounded border border-white/10 text-white font-bold">MINE</span>
                                <span className="bg-black/60 py-1.5 px-2 rounded border border-white/10 text-white font-bold">GHOST NET</span>
                            </div>
                        </div>
                    </div>
                </div>

                {/* Frozen Cutoff Thresholds Card */}
                <div className="glass-card p-6 space-y-5">
                    <div className="flex items-center justify-between border-b border-white/10 pb-3">
                        <h3 className="font-bold text-white uppercase text-sm">
                            FROZEN THRESHOLDS
                        </h3>
                        <span className="text-xs text-cyan-400 font-bold">SOURCE: VALIDATION CALIBRATION</span>
                    </div>

                    <div className="grid grid-cols-2 gap-3 text-xs">
                        <div className="p-3.5 rounded-xl bg-black/40 border border-white/5 flex justify-between items-center">
                            <span className="text-slate-400 font-bold">PIPE</span>
                            <span className="text-cyan-300 font-bold text-sm">0.56</span>
                        </div>

                        <div className="p-3.5 rounded-xl bg-black/40 border border-white/5 flex justify-between items-center">
                            <span className="text-slate-400 font-bold">SHIPWRECK</span>
                            <span className="text-cyan-300 font-bold text-sm">0.45</span>
                        </div>

                        <div className="p-3.5 rounded-xl bg-black/40 border border-white/5 flex justify-between items-center">
                            <span className="text-slate-400 font-bold">MINE</span>
                            <span className="text-cyan-300 font-bold text-sm">0.89</span>
                        </div>

                        <div className="p-3.5 rounded-xl bg-black/40 border border-white/5 flex justify-between items-center">
                            <span className="text-slate-400 font-bold">GHOST NET</span>
                            <span className="text-cyan-300 font-bold text-sm">0.36</span>
                        </div>
                    </div>
                </div>
            </div>

            {/* Runtime & System Mode Grid */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                {/* Runtime Compute Card */}
                <div className="glass-card p-6 space-y-5">
                    <div className="flex items-center justify-between border-b border-white/10 pb-3">
                        <h3 className="font-bold text-white uppercase text-sm">
                            RUNTIME METRICS
                        </h3>
                        <span className="text-xs text-cyan-400 font-bold">GPU ACCELERATION</span>
                    </div>

                    <div className="space-y-3 text-xs">
                        <div className="flex justify-between items-center bg-black/40 p-3 rounded-xl border border-white/5">
                            <span className="text-slate-400">GPU</span>
                            <span className="text-white font-bold">RTX 4050</span>
                        </div>

                        <div className="flex justify-between items-center bg-black/40 p-3 rounded-xl border border-white/5">
                            <span className="text-slate-400">BACKEND</span>
                            <span className="text-white font-bold">TensorRT FP32</span>
                        </div>

                        <div className="flex justify-between items-center bg-black/40 p-3 rounded-xl border border-white/5">
                            <span className="text-slate-400">BACKEND STATUS</span>
                            <span className={`font-bold ${backendOnline ? 'text-emerald-400' : 'text-amber-300'}`}>
                                {backendOnline ? 'ONLINE' : 'OFFLINE'}
                            </span>
                        </div>

                        <div className="flex justify-between items-center bg-black/40 p-3 rounded-xl border border-white/5">
                            <span className="text-slate-400">G3 P50</span>
                            <span className="text-cyan-300 font-bold">
                                {runtimeBenchmark ? `${runtimeBenchmark.p50_ms.toFixed(2)} ms` : '—'}
                            </span>
                        </div>

                        <div className="flex justify-between items-center bg-black/40 p-3 rounded-xl border border-white/5">
                            <span className="text-slate-400">G3 THROUGHPUT</span>
                            <span className="text-cyan-300 font-bold">
                                {runtimeBenchmark ? `${runtimeBenchmark.throughput_fps.toFixed(1)} FPS` : '—'}
                            </span>
                        </div>
                    </div>
                </div>

                {/* System Mode Card */}
                <div className="glass-card p-6 space-y-5">
                    <div className="flex items-center justify-between border-b border-white/10 pb-3">
                        <h3 className="font-bold text-white uppercase text-sm">
                            SYSTEM MODE
                        </h3>
                        <span className="text-xs text-amber-400 font-bold">SIMULATION DEMO</span>
                    </div>

                    <div className="space-y-3 text-xs">
                        <div className="flex justify-between items-center bg-black/40 p-3 rounded-xl border border-white/5">
                            <span className="text-slate-400">OPERATING MODE</span>
                            <span className="text-cyan-400 font-bold">SIMULATION</span>
                        </div>

                        <div className="flex justify-between items-center bg-black/40 p-3 rounded-xl border border-amber-500/30">
                            <span className="text-slate-400">PHYSICAL HARDWARE</span>
                            <span className="text-amber-400 font-bold">NOT CONNECTED</span>
                        </div>
                    </div>
                </div>
            </div>

            {/* Inference Pipeline Visual */}
            <div className="glass-card p-6 space-y-4">
                <h3 className="font-bold text-white uppercase text-sm border-b border-white/10 pb-3">
                    INFERENCE PIPELINE ARCHITECTURE
                </h3>

                <div className="flex flex-col md:flex-row items-center justify-between gap-3 text-xs pt-2">
                    {pipelineSteps.map((step, idx) => (
                        <React.Fragment key={idx}>
                            <div className="w-full md:w-auto p-3.5 rounded-xl bg-black/60 border border-cyan-500/30 text-center font-bold text-cyan-300 shadow-[0_0_12px_rgba(0,240,255,0.1)]">
                                {step}
                            </div>
                            {idx < pipelineSteps.length - 1 && (
                                <div className="text-slate-500 font-bold rotate-90 md:rotate-0">➔</div>
                            )}
                        </React.Fragment>
                    ))}
                </div>
            </div>
        </div>
    );
}
