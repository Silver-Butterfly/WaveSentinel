import React, { useState } from 'react';
import { 
    Activity, 
    Radio, 
    Compass, 
    Gauge, 
    Waves, 
    Server, 
    AlertCircle, 
    Clock, 
    CheckCircle2,
    Sliders
} from 'lucide-react';
import { useSimulation } from '../../simulation/SimulationContext.jsx';

export default function SensorTelemetryView() {
    const { telemetry, telemetryLog } = useSimulation();
    const [telemetrySource, setTelemetrySource] = useState('SIMULATION');

    const sensors = [
        { name: 'SSS SONAR', status: 'SIMULATED', color: 'text-cyan-300 border-cyan-400/50 bg-cyan-500/10' },
        { name: 'IMU', status: 'SIMULATED', color: 'text-cyan-300 border-cyan-400/50 bg-cyan-500/10' },
        { name: 'GNSS', status: 'SIMULATED', color: 'text-cyan-300 border-cyan-400/50 bg-cyan-500/10' },
        { name: 'DEPTH', status: 'SIMULATED', color: 'text-cyan-300 border-cyan-400/50 bg-cyan-500/10' },
        { name: 'PRESSURE', status: 'SIMULATED', color: 'text-cyan-300 border-cyan-400/50 bg-cyan-500/10' }
    ];

    return (
        <div className="space-y-8 animate-fade-in pb-12 font-mono">
            {/* Header Banner */}
            <div className="glass-card p-6 flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
                <div className="space-y-1">
                    <div className="flex items-center gap-3">
                        <Activity className="text-cyan-400" size={24} />
                        <h2 className="text-2xl font-bold text-white tracking-wide uppercase">
                            SENSOR & TELEMETRY
                        </h2>
                        <span className="px-3 py-1 rounded-full bg-cyan-500/20 border border-cyan-400/50 text-cyan-300 text-xs font-bold uppercase tracking-wider">
                            SIMULATION ARCHITECTURE
                        </span>
                    </div>
                    <p className="text-slate-300 text-xs">
                        Physical Sensor Health Diagnostic Bus & Subsea Telemetry Simulation Layer
                    </p>
                </div>
            </div>

            {/* Sensor Health Grid */}
            <div className="glass-card p-6 space-y-4">
                <div className="flex items-center justify-between border-b border-white/10 pb-3">
                    <h3 className="font-bold text-white uppercase text-sm">
                        SENSOR HEALTH BUS
                    </h3>
                    <span className="text-xs text-slate-400">Physical hardware offline • Synthetic state active</span>
                </div>

                <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-4 text-xs">
                    {sensors.map((s, idx) => (
                        <div key={idx} className="p-4 rounded-xl bg-black/40 border border-white/10 space-y-2 text-center">
                            <div className="text-slate-400 text-[11px] font-bold">{s.name}</div>
                            <div className={`py-1 px-2 rounded-lg text-xs font-bold border ${s.color} inline-block`}>
                                ● {s.status}
                            </div>
                        </div>
                    ))}
                </div>
            </div>

            {/* Telemetry Stream & Telemetry Source Selection Grid */}
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
                {/* Left 2 Columns: Live Telemetry Stream Table */}
                <div className="lg:col-span-2 glass-card p-6 space-y-5">
                    <div className="flex items-center justify-between border-b border-white/10 pb-3">
                        <div className="flex items-center gap-2">
                            <Waves className="text-cyan-400" size={18} />
                            <h3 className="font-bold text-white uppercase text-sm">
                                LIVE TELEMETRY STREAM
                            </h3>
                        </div>
                        <span className="flex items-center gap-1.5 text-xs text-cyan-300 font-bold">
                            <span className="w-2 h-2 rounded-full bg-cyan-400 animate-ping"></span>
                            STREAMING (SIMULATION)
                        </span>
                    </div>

                    <div className="overflow-x-auto">
                        <table className="w-full text-left text-xs border-collapse">
                            <thead>
                                <tr className="border-b border-white/10 text-slate-400 uppercase">
                                    <th className="py-3 px-4">TIME</th>
                                    <th className="py-3 px-4 text-right">HEAVE</th>
                                    <th className="py-3 px-4 text-right">PITCH</th>
                                    <th className="py-3 px-4 text-right">ROLL</th>
                                    <th className="py-3 px-4 text-right">DEPTH</th>
                                </tr>
                            </thead>
                            <tbody>
                                {telemetryLog.map((row, idx) => (
                                    <tr key={idx} className="border-b border-white/5 hover:bg-white/5 font-mono">
                                        <td className="py-3 px-4 text-slate-400 font-bold">{row.time}</td>
                                        <td className="py-3 px-4 text-right text-cyan-300 font-bold">{row.heave}</td>
                                        <td className="py-3 px-4 text-right text-white">{row.pitch}</td>
                                        <td className="py-3 px-4 text-right text-white">{row.roll}</td>
                                        <td className="py-3 px-4 text-right text-slate-300">{row.depth}</td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                </div>

                {/* Right 1 Column: Telemetry Source Selector & Status */}
                <div className="lg:col-span-1 glass-card p-6 space-y-6 text-xs">
                    <div className="flex items-center justify-between border-b border-white/10 pb-3 font-bold text-white uppercase">
                        <span>TELEMETRY SOURCE</span>
                        <Radio size={16} className="text-cyan-400" />
                    </div>

                    <div className="space-y-3">
                        <label className="flex items-center justify-between p-3.5 rounded-xl bg-black/40 border border-cyan-500/40 text-cyan-300 cursor-pointer font-bold">
                            <div className="flex items-center gap-2">
                                <input 
                                    type="radio" 
                                    name="tel_source" 
                                    checked={telemetrySource === 'SIMULATION'}
                                    onChange={() => setTelemetrySource('SIMULATION')}
                                    className="accent-cyan-400"
                                />
                                <span>SIMULATION</span>
                            </div>
                            <span className="text-[10px] bg-cyan-500/20 px-2 py-0.5 rounded">ACTIVE</span>
                        </label>

                        <label className="flex items-center justify-between p-3.5 rounded-xl bg-black/20 border border-white/5 text-slate-500 cursor-not-allowed">
                            <div className="flex items-center gap-2">
                                <input 
                                    type="radio" 
                                    name="tel_source" 
                                    disabled
                                    className="accent-slate-500"
                                />
                                <span>REAL IMU</span>
                            </div>
                            <span className="text-[9px] bg-white/5 px-2 py-0.5 rounded text-slate-500">DISABLED</span>
                        </label>

                        <div className="p-3.5 rounded-xl bg-black/40 border border-amber-500/30 text-slate-300 space-y-1.5 text-[11px]">
                            <div className="text-amber-400 font-bold uppercase flex items-center gap-1.5">
                                <AlertCircle size={14} /> Hardware Note
                            </div>
                            <p className="leading-relaxed text-slate-400">
                                Real IMU telemetry integration requires physical hardware connection. System is operating strictly in DEMO MODE.
                            </p>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
}
