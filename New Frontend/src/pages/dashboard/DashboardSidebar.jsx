import React from 'react';
import { 
    LayoutDashboard, 
    Radio,
    Camera, 
    Upload, 
    Layers, 
    Database, 
    ShieldAlert,
    MapPin, 
    FileText,
    Activity,
    Cpu, 
    Settings, 
    ArrowLeft,
    Shield
} from 'lucide-react';

export default function DashboardSidebar({ activeTab, onSelectTab, onBack }) {
    const mainNav = [
        { id: 'dashboard', label: 'Dashboard', icon: LayoutDashboard }
    ];

    const operationsNav = [
        { id: 'missioncontrol', label: 'Mission Control', icon: Radio },
        { id: 'livescan', label: 'Live Scan', icon: Camera },
        { id: 'uploadscan', label: 'Upload Scan', icon: Upload },
        { id: 'batchanalysis', label: 'Batch Analysis', icon: Layers }
    ];

    const dataNav = [
        { id: 'archive', label: 'Detection Archive', icon: Database },
        { id: 'reviewtriage', label: 'Review & Triage', icon: ShieldAlert },
        { id: 'map', label: 'Survey Map', icon: MapPin },
        { id: 'evidencereports', label: 'Evidence & Reports', icon: FileText }
    ];

    const systemNav = [
        { id: 'sensortelemetry', label: 'Sensor & Telemetry', icon: Activity },
        { id: 'modelsystem', label: 'Model & System', icon: Cpu },
        { id: 'settings', label: 'Settings', icon: Settings }
    ];

    return (
        <aside className="w-64 bg-black/60 backdrop-blur-xl border-r border-white/10 flex flex-col justify-between p-4 min-h-screen text-slate-200 select-none z-30">
            {/* Header / Brand */}
            <div className="space-y-6">
                <div className="space-y-3">
                    <button 
                        onClick={onBack}
                        className="flex items-center gap-2 text-xs font-mono text-slate-400 hover:text-cyan-400 transition-colors cursor-pointer py-1 px-2 rounded hover:bg-white/5"
                    >
                        <ArrowLeft size={14} />
                        FLEET COMMAND
                    </button>

                    <div className="flex items-center gap-3 px-2 pt-1">
                        <div className="w-9 h-9 rounded-xl bg-cyan-500/20 border border-cyan-400/50 flex items-center justify-center text-cyan-400 shadow-[0_0_15px_rgba(0,240,255,0.2)]">
                            <Shield size={20} />
                        </div>
                        <div>
                            <h1 className="font-mono font-bold text-white tracking-wider text-base leading-tight">
                                WAVESENTINEL
                            </h1>
                            <div className="font-mono text-[10px] text-cyan-400 tracking-widest uppercase font-bold">
                                SONAR INTELLIGENCE
                            </div>
                        </div>
                    </div>
                </div>

                <div className="h-[1px] bg-white/10 mx-2"></div>

                {/* Nav Links */}
                <nav className="space-y-6 font-mono text-xs">
                    {/* Main Nav */}
                    <div className="space-y-1">
                        {mainNav.map((item) => {
                            const Icon = item.icon;
                            const isActive = activeTab === item.id;
                            return (
                                <button
                                    key={item.id}
                                    onClick={() => onSelectTab(item.id)}
                                    className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-xl font-bold transition-all cursor-pointer ${
                                        isActive 
                                            ? 'bg-cyan-500 text-black shadow-[0_0_18px_rgba(0,240,255,0.4)]' 
                                            : 'text-slate-300 hover:bg-white/5 hover:text-white'
                                    }`}
                                >
                                    <Icon size={16} />
                                    <span>{item.label}</span>
                                </button>
                            );
                        })}
                    </div>

                    {/* Operations */}
                    <div className="space-y-1">
                        <div className="px-3 text-[10px] font-bold text-slate-500 uppercase tracking-widest mb-1">
                            OPERATIONS
                        </div>
                        {operationsNav.map((item) => {
                            const Icon = item.icon;
                            const isActive = activeTab === item.id;
                            return (
                                <button
                                    key={item.id}
                                    onClick={() => onSelectTab(item.id)}
                                    className={`w-full flex items-center gap-3 px-3 py-2 rounded-xl transition-all cursor-pointer ${
                                        isActive 
                                            ? 'bg-cyan-500/20 border border-cyan-400/50 text-cyan-300 shadow-[0_0_12px_rgba(0,240,255,0.2)] font-bold' 
                                            : 'text-slate-300 hover:bg-white/5 hover:text-white'
                                    }`}
                                >
                                    <Icon size={15} />
                                    <span>{item.label}</span>
                                </button>
                            );
                        })}
                    </div>

                    {/* Data */}
                    <div className="space-y-1">
                        <div className="px-3 text-[10px] font-bold text-slate-500 uppercase tracking-widest mb-1">
                            DATA
                        </div>
                        {dataNav.map((item) => {
                            const Icon = item.icon;
                            const isActive = activeTab === item.id;
                            return (
                                <button
                                    key={item.id}
                                    onClick={() => onSelectTab(item.id)}
                                    className={`w-full flex items-center gap-3 px-3 py-2 rounded-xl transition-all cursor-pointer ${
                                        isActive 
                                            ? 'bg-cyan-500/20 border border-cyan-400/50 text-cyan-300 shadow-[0_0_12px_rgba(0,240,255,0.2)] font-bold' 
                                            : 'text-slate-300 hover:bg-white/5 hover:text-white'
                                    }`}
                                >
                                    <Icon size={15} />
                                    <span>{item.label}</span>
                                </button>
                            );
                        })}
                    </div>

                    {/* System */}
                    <div className="space-y-1">
                        <div className="px-3 text-[10px] font-bold text-slate-500 uppercase tracking-widest mb-1">
                            SYSTEM
                        </div>
                        {systemNav.map((item) => {
                            const Icon = item.icon;
                            const isActive = activeTab === item.id;
                            return (
                                <button
                                    key={item.id}
                                    onClick={() => onSelectTab(item.id)}
                                    className={`w-full flex items-center gap-3 px-3 py-2 rounded-xl transition-all cursor-pointer ${
                                        isActive 
                                            ? 'bg-cyan-500/20 border border-cyan-400/50 text-cyan-300 shadow-[0_0_12px_rgba(0,240,255,0.2)] font-bold' 
                                            : 'text-slate-300 hover:bg-white/5 hover:text-white'
                                    }`}
                                >
                                    <Icon size={15} />
                                    <span>{item.label}</span>
                                </button>
                            );
                        })}
                    </div>
                </nav>
            </div>

            {/* Footer System Status */}
            <div className="pt-4 border-t border-white/10 space-y-2 font-mono text-[11px]">
                <div className="flex items-center justify-between px-2">
                    <span className="flex items-center gap-2 text-cyan-400 font-bold">
                        <span className="w-2 h-2 rounded-full bg-cyan-400 animate-pulse"></span>
                        DEMO MODE
                    </span>
                    <span className="text-slate-400 text-[10px]">v1.4.0</span>
                </div>
                <div className="flex items-center justify-between px-2 text-slate-300">
                    <span>SSS STREAM</span>
                    <span className="text-cyan-400 font-bold">SIMULATED</span>
                </div>
            </div>
        </aside>
    );
}
