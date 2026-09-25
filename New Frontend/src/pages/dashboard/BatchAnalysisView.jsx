import React, { useEffect, useState } from 'react';
import { Layers, RefreshCw, ChevronRight, Cpu } from 'lucide-react';
import { useSimulation } from '../../simulation/SimulationContext.jsx';
import { getBatchStatus, startBatch } from '../../api/client.js';

export default function BatchAnalysisView({ onSelectDetection }) {
    const { missionState, detections, runtimeBenchmark, backendOnline } = useSimulation();
    const [batch, setBatch] = useState({
        batchId: 'BATCH-018', status: 'IDLE', totalFrames: 248, processedFrames: 0, detections: 0, reviewRequired: 0
    });

    const refreshBatch = async () => {
        try {
            const response = await getBatchStatus();
            if (response?.batch) setBatch(response.batch);
        } catch {}
    };

    useEffect(() => {
        refreshBatch();
        const interval = setInterval(refreshBatch, 400);
        return () => clearInterval(interval);
    }, []);

    const startBatchProcessing = async () => {
        try {
            const response = await startBatch();
            if (response?.batch) setBatch(response.batch);
        } catch {
            setBatch(prev => ({ ...prev, status: 'PROCESSING' }));
        }
    };

    const progress = Math.round((batch.processedFrames / Math.max(1, batch.totalFrames)) * 100);
    const batchRows = detections.slice(0, 6).map((d) => ({
        frame: d.frame_id,
        detection: d.class,
        confidence: `${Number(d.confidence).toFixed(1)}%`,
        risk: d.risk || 'UNASSESSED',
        source: d.source || 'simulation'
    }));

    return (
        <div className="space-y-8 animate-fade-in pb-12 font-mono">
            <div className="glass-card p-6 flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
                <div className="space-y-1">
                    <div className="flex items-center gap-3">
                        <Layers className="text-cyan-400" size={24} />
                        <h2 className="text-2xl font-bold text-white tracking-wide uppercase">BATCH ANALYSIS</h2>
                        <span className="px-3 py-1 rounded-full bg-cyan-500/20 border border-cyan-400/50 text-cyan-300 text-xs font-bold uppercase">
                            {batch.batchId} • {batch.totalFrames} FRAMES
                        </span>
                    </div>
                    <p className="text-slate-300 text-xs">
                        Batch control is backend-backed; current batch metrics are simulation runtime state.
                    </p>
                </div>

                {batch.status !== 'PROCESSING' && (
                    <button
                        onClick={startBatchProcessing}
                        className="px-5 py-2.5 rounded-xl bg-cyan-500 hover:bg-cyan-400 text-black font-bold text-xs uppercase tracking-wider transition-all shadow-lg shadow-cyan-500/20 cursor-pointer"
                    >
                        RUN {batch.batchId} INFERENCE
                    </button>
                )}
            </div>

            <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-4 text-xs">
                {[
                    ['BATCH ID', batch.batchId],
                    ['TOTAL FRAMES', batch.totalFrames],
                    ['PROCESSED', batch.processedFrames],
                    ['REMAINING', Math.max(0, batch.totalFrames - batch.processedFrames)],
                    ['DETECTIONS', batch.detections],
                    ['REVIEW REQUIRED', batch.reviewRequired]
                ].map(([label, value]) => (
                    <div key={label} className="glass-card p-4 space-y-1">
                        <div className="text-slate-400 text-[10px] uppercase">{label}</div>
                        <div className="text-cyan-300 font-bold text-base">{value}</div>
                    </div>
                ))}
            </div>

            <div className="glass-card p-6 space-y-4">
                <div className="flex items-center justify-between border-b border-white/10 pb-3">
                    <div className="flex items-center gap-2">
                        <Cpu className="text-cyan-400" size={18} />
                        <h3 className="font-bold text-white uppercase text-sm">RUNTIME METRICS</h3>
                    </div>
                    <span className={`text-xs font-bold ${backendOnline ? 'text-emerald-400' : 'text-amber-300'}`}>
                        {backendOnline ? 'BACKEND CONNECTED' : 'SIMULATION FALLBACK'}
                    </span>
                </div>
                <div className="grid grid-cols-2 sm:grid-cols-5 gap-4 text-xs">
                    <div className="p-3.5 rounded-xl bg-black/40 border border-white/5"><div className="text-slate-400 text-[10px]">MODEL</div><div className="text-white font-bold">Phase-D Robust-V2-HN</div></div>
                    <div className="p-3.5 rounded-xl bg-black/40 border border-white/5"><div className="text-slate-400 text-[10px]">G3 P50</div><div className="text-cyan-300 font-bold">{runtimeBenchmark ? `${runtimeBenchmark.p50_ms.toFixed(2)} ms` : '—'}</div></div>
                    <div className="p-3.5 rounded-xl bg-black/40 border border-white/5"><div className="text-slate-400 text-[10px]">THROUGHPUT</div><div className="text-emerald-400 font-bold">{runtimeBenchmark ? `${runtimeBenchmark.throughput_fps.toFixed(1)} FPS` : '—'}</div></div>
                    <div className="p-3.5 rounded-xl bg-black/40 border border-white/5"><div className="text-slate-400 text-[10px]">GPU</div><div className="text-emerald-400 font-bold">RTX 4050</div></div>
                    <div className="p-3.5 rounded-xl bg-black/40 border border-white/5"><div className="text-slate-400 text-[10px]">MISSION</div><div className="text-white font-bold">{missionState.missionId}</div></div>
                </div>
            </div>

            <div className="glass-card p-6 space-y-4 text-xs">
                <div className="flex items-center justify-between border-b border-white/10 pb-3">
                    <h3 className="font-bold text-white uppercase text-sm">FRAME INGESTION PROGRESS FEED</h3>
                    <span className="text-cyan-400 font-bold">{progress}% PROCESSED</span>
                </div>
                <div className="w-full bg-black/60 rounded-full h-2.5 overflow-hidden border border-white/10">
                    <div className="bg-cyan-400 h-full transition-all duration-300" style={{ width: `${progress}%` }} />
                </div>
                <div className="text-slate-400 text-[11px]">Status: <span className="text-cyan-300 font-bold">{batch.status}</span></div>
            </div>

            <div className="glass-card p-6 space-y-4 text-xs">
                <div className="flex items-center justify-between border-b border-white/10 pb-3">
                    <h3 className="font-bold text-white uppercase text-sm">BATCH DETECTION RECORDS</h3>
                    <span className="text-slate-400">Persisted backend records</span>
                </div>
                <div className="overflow-x-auto">
                    <table className="w-full text-left border-collapse">
                        <thead><tr className="border-b border-white/10 text-slate-400 uppercase"><th className="py-3 px-4">FRAME</th><th className="py-3 px-4">DETECTION</th><th className="py-3 px-4 text-right">CONFIDENCE</th><th className="py-3 px-4">RISK</th><th className="py-3 px-4">SOURCE</th></tr></thead>
                        <tbody>
                            {batchRows.map((row) => (
                                <tr key={row.frame} onClick={() => onSelectDetection && onSelectDetection(row)} className="border-b border-white/5 hover:bg-white/5 cursor-pointer transition-colors">
                                    <td className="py-3 px-4 font-bold text-white">{row.frame}</td>
                                    <td className="py-3 px-4 text-slate-200">{row.detection}</td>
                                    <td className="py-3 px-4 text-right font-bold text-cyan-300">{row.confidence}</td>
                                    <td className="py-3 px-4 text-slate-300">{row.risk}</td>
                                    <td className="py-3 px-4 text-cyan-300 uppercase">{row.source}</td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
    );
}
