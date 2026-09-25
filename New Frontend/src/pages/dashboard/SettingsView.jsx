import React from 'react';
import { Settings, Sliders, Cpu, RotateCcw } from 'lucide-react';
import { useSimulation } from '../../simulation/SimulationContext.jsx';

export default function SettingsView() {
    const { settings, resetDemo } = useSimulation();
    const thresholds = settings.frozenThresholds || {
        pipe: 0.56,
        shipwreck: 0.45,
        mine: 0.89,
        ghost_net: 0.36
    };

    const handleDemoReset = async () => {
        if (window.confirm('Reset simulated mission state and detection records to defaults?')) {
            await resetDemo();
            alert('Simulation state reset to initial mission parameters.');
        }
    };

    return (
        <div className="space-y-8 animate-fade-in pb-12 font-mono text-xs">
            <div className="glass-card p-6 flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
                <div className="space-y-1">
                    <div className="flex items-center gap-3">
                        <Settings className="text-cyan-400" size={24} />
                        <h2 className="text-2xl font-bold text-white tracking-wide uppercase">
                            WORKSTATION SETTINGS
                        </h2>
                        <span className="px-3 py-1 rounded-full bg-cyan-500/20 border border-cyan-400/50 text-cyan-300 text-xs font-bold uppercase">
                            SIMULATION MODE
                        </span>
                    </div>
                    <p className="text-slate-300 text-xs">
                        Read-only deployment configuration for the frozen detector and current simulation workstation.
                    </p>
                </div>

                <button
                    onClick={handleDemoReset}
                    className="px-4 py-2.5 rounded-xl bg-red-500/20 hover:bg-red-500/30 border border-red-500/40 text-red-300 font-bold uppercase transition-all flex items-center gap-2 cursor-pointer"
                >
                    <RotateCcw size={14} /> DEMO RESET
                </button>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                <div className="glass-card p-6 space-y-5">
                    <div className="flex items-center justify-between border-b border-white/10 pb-3">
                        <div className="flex items-center gap-2">
                            <Cpu size={16} className="text-cyan-400" />
                            <h3 className="font-bold text-white uppercase text-sm">RUNTIME</h3>
                        </div>
                        <span className="text-xs text-cyan-400 font-bold">FROZEN DEPLOYMENT</span>
                    </div>

                    <div className="space-y-3">
                        <div className="flex justify-between items-center bg-black/40 p-3 rounded-xl border border-white/5">
                            <span className="text-slate-400">BACKEND MODE</span>
                            <span className="text-cyan-300 font-bold">LOCAL FASTAPI</span>
                        </div>
                        <div className="flex justify-between items-center bg-black/40 p-3 rounded-xl border border-white/5">
                            <span className="text-slate-400">INFERENCE</span>
                            <span className="text-emerald-400 font-bold">TENSORRT FP32</span>
                        </div>
                        <div className="flex justify-between items-center bg-black/40 p-3 rounded-xl border border-white/5">
                            <span className="text-slate-400">INPUT</span>
                            <span className="text-white font-bold">640 × 640 FLOAT</span>
                        </div>
                        <div className="flex justify-between items-center bg-black/40 p-3 rounded-xl border border-white/5">
                            <span className="text-slate-400">OUTPUT</span>
                            <span className="text-white font-bold">8 × 8400 FLOAT</span>
                        </div>
                    </div>
                </div>

                <div className="glass-card p-6 space-y-5">
                    <div className="flex items-center gap-2 border-b border-white/10 pb-3">
                        <Sliders size={16} className="text-cyan-400" />
                        <h3 className="font-bold text-white uppercase text-sm">FROZEN CLASS THRESHOLDS</h3>
                    </div>

                    <div className="grid grid-cols-2 gap-3">
                        {Object.entries({
                            PIPE: thresholds.pipe,
                            SHIPWRECK: thresholds.shipwreck,
                            MINE: thresholds.mine,
                            'GHOST NET': thresholds.ghost_net
                        }).map(([label, value]) => (
                            <div key={label} className="p-3.5 rounded-xl bg-black/40 border border-white/5 flex justify-between items-center">
                                <span className="text-slate-400 font-bold">{label}</span>
                                <span className="text-cyan-300 font-bold text-sm">{Number(value).toFixed(2)}</span>
                            </div>
                        ))}
                    </div>

                    <div className="p-3.5 rounded-xl bg-black/40 border border-amber-500/30 text-slate-300 space-y-1 text-[11px]">
                        <span className="text-amber-400 font-bold uppercase">LOCKED FOR DEPLOYMENT</span>
                        <p className="text-slate-400">
                            Thresholds were calibrated on validation data and are not editable from the workstation UI.
                            NMS IoU: {(settings.nmsIou ?? 0.50).toFixed(2)}.
                        </p>
                    </div>
                </div>
            </div>
        </div>
    );
}
