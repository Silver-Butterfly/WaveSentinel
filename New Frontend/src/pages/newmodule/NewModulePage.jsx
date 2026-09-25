import React, { useEffect } from 'react';
import { ArrowLeft, Cpu, Fan, Radio, Layers, Globe, Radar, ChevronRight } from 'lucide-react';
// Fallback images since the original assets are missing
const terminalBg = "https://images.unsplash.com/photo-1550751827-4bd374c3f58b?auto=format&fit=crop&w=1920&q=80";
const submarineImg = "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='400' height='150'%3E%3Cpath d='M50,75 L100,50 L300,50 C350,50 350,100 300,100 L100,100 Z' fill='%23555' stroke='%23333'/%3E%3Cpath d='M200,50 L200,25 L225,25 L225,50 Z' fill='%23555'/%3E%3Ctext x='180' y='80' font-size='24' fill='%23fff' font-family='sans-serif'%3ESUBMARINE%3C/text%3E%3C/svg%3E";

import Grainient from '../../components/Grainient';
import HardwareArchitecture from '../../components/newmodule/HardwareArchitecture';

export default function NewModulePage({ onBack }) {
    useEffect(() => {
        window.scrollTo(0, 0);
    }, []);

    return (
        <div className="min-h-screen bg-void text-white font-body relative overflow-x-hidden">
            <div className="relative z-10 max-w-7xl mx-auto px-6 py-12 md:py-24">
                {/* Navigation */}
                <nav className="mb-16">
                    <button 
                        onClick={onBack}
                        className="flex items-center gap-2 text-stardust hover:text-electric-blue transition-all duration-300 font-mono text-sm tracking-wider uppercase"
                    >
                        <ArrowLeft size={16} />
                        RETURN TO FLEET COMMAND
                    </button>
                </nav>

                {/* Hero Section */}
                <section className="mb-24 flex flex-col lg:flex-row items-center gap-16">
                    <div className="flex-1 space-y-6">
                        <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full border border-electric-blue/30 bg-electric-blue/10 text-electric-blue text-xs font-mono tracking-widest uppercase">
                            <span className="w-2 h-2 rounded-full bg-electric-blue animate-ping"></span>
                            AUSV SPECIFICATION : TACTICAL NODE
                        </div>
                        <h1 className="font-heading text-4xl sm:text-5xl md:text-7xl font-bold leading-tight">
                            Autonomous Subsea <br />
                            <span className="text-neon-teal">
                                Loyal Wingman
                            </span>
                        </h1>
                        <p className="text-stardust text-lg max-w-xl leading-relaxed">
                            Edge-compute AI intelligence platform paired with acoustic stealth propulsion. Deployable as a distributed, attritable subsea acoustic shield for maritime defense, port security, and EEZ surveillance.
                        </p>
                        
                        <div className="pt-4 flex gap-4">
                            <button onClick={() => document.getElementById('specs-section').scrollIntoView({ behavior: 'smooth' })} className="h-11 px-8 rounded-full bg-[#00D2FF] text-[#000000] font-bold font-mono tracking-wider uppercase text-sm shadow-[0_0_20px_-5px_rgba(0,210,255,0.5)] hover:scale-105 hover:shadow-[0_0_30px_-5px_rgba(0,210,255,0.6)] transition-all duration-300">
                                DEPLOYMENT SPECS
                            </button>
                            <button onClick={() => document.getElementById('telemetry-section').scrollIntoView({ behavior: 'smooth' })} className="h-11 px-8 rounded-full border-2 border-electric-blue bg-transparent text-electric-blue font-bold font-mono tracking-wider uppercase text-sm hover:bg-electric-blue/10 transition-all duration-300">
                                SYSTEM ARCHITECTURE
                            </button>
                        </div>
                    </div>

                    {/* Submarine Hero Graphic */}
                    <div 
                        className="flex-1 relative h-[300px] md:h-[450px] w-full flex items-center justify-center bg-transparent"
                        style={{ 
                            WebkitMaskImage: 'radial-gradient(circle at center, rgba(0,0,0,1) 40%, rgba(0,0,0,0) 80%)', 
                            maskImage: 'radial-gradient(circle at center, rgba(0,0,0,1) 40%, rgba(0,0,0,0) 80%)' 
                        }}
                    >
                        {/* Subtle Backlight Glow */}
                        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-3/4 h-1/2 bg-[#00D2FF] opacity-20 blur-[50px] rounded-full pointer-events-none"></div>
                        
                        {/* Submarine Image */}
                        <img 
                            src={submarineImg} 
                            alt="AUSV Submarine" 
                            className="relative z-10 w-[95%] max-w-[650px] object-contain drop-shadow-2xl -rotate-2 scale-105"
                        />
                        
                        {/* Visual Micro-Pins */}
                        <div className="absolute top-[35%] left-[5%] flex items-center gap-2 z-20 pointer-events-none">
                            <div className="w-1.5 h-1.5 rounded-full bg-[#00D2FF] shadow-[0_0_8px_#00D2FF]"></div>
                            <div className="h-[1px] w-4 sm:w-8 bg-slate-500/50"></div>
                            <span className="font-mono text-[11px] text-slate-400 whitespace-nowrap">NOSE: 16-CH PASSIVE HYDROPHONE ARRAY</span>
                        </div>
                        
                        <div className="absolute top-[65%] left-[45%] flex items-center gap-2 z-20 pointer-events-none">
                            <span className="font-mono text-[11px] text-slate-400 whitespace-nowrap text-right">MID: OIL-BASED PISTON VBS (300M RATED)</span>
                            <div className="h-[1px] w-4 sm:w-8 bg-slate-500/50"></div>
                            <div className="w-1.5 h-1.5 rounded-full bg-[#00D2FF] shadow-[0_0_8px_#00D2FF]"></div>
                        </div>

                        <div className="absolute top-[45%] right-[5%] flex items-center gap-2 z-20 pointer-events-none">
                            <span className="font-mono text-[11px] text-slate-400 whitespace-nowrap text-right">STERN: SHAFTLESS RIM-DRIVEN THRUSTER</span>
                            <div className="h-[1px] w-4 sm:w-8 bg-slate-500/50"></div>
                            <div className="w-1.5 h-1.5 rounded-full bg-[#00D2FF] shadow-[0_0_8px_#00D2FF]"></div>
                        </div>
                    </div>
                </section>

                {/* Data Cards Grid */}
                <section className="grid grid-cols-1 md:grid-cols-3 gap-8">
                    <div className="group bg-navy-blue/20 backdrop-blur-md p-8 rounded-2xl border border-slate-700/50 hover:-translate-y-1 hover:border-[#00D2FF]/15 hover:shadow-[0_0_15px_rgba(0,210,255,0.15)] transition-all duration-300 relative overflow-hidden flex flex-col h-full">
                        <Cpu className="absolute -bottom-4 -right-4 w-32 h-32 text-electric-blue opacity-5 group-hover:opacity-10 transition-all duration-300 transform -rotate-12" />
                        <div className="bg-electric-blue/10 border border-electric-blue/30 rounded-lg w-12 h-12 flex items-center justify-center mb-6 shadow-[0_0_20px_rgba(0,240,255,0.1)] group-hover:shadow-[0_0_20px_rgba(0,240,255,0.3)] transition-all">
                            <Cpu className="text-electric-blue" />
                        </div>
                        <h3 className="font-heading font-semibold text-2xl mb-2 text-white">Edge AI & Autonomy</h3>
                        <p className="text-stardust text-sm leading-relaxed mb-6 relative z-10">
                            NVIDIA Jetson Orin Nano (20–40 TOPS) running containerized EfficientNet-B0 for real-time acoustic classification and Grad-CAM explainability, fully isolated from the flight controller.
                        </p>
                        <a href="#" className="text-electric-blue font-mono text-sm tracking-wider uppercase hover:underline relative z-10 mt-auto">VIEW COMPUTE STACK <span className="text-neon-teal">&rarr;</span></a>
                    </div>

                    <div className="group bg-navy-blue/20 backdrop-blur-md p-8 rounded-2xl border border-slate-700/50 hover:-translate-y-1 hover:border-[#00D2FF]/15 hover:shadow-[0_0_15px_rgba(0,210,255,0.15)] transition-all duration-300 relative overflow-hidden flex flex-col h-full">
                        <Fan className="absolute -bottom-4 -right-4 w-32 h-32 text-neon-teal opacity-5 group-hover:opacity-10 transition-all duration-300 transform -rotate-12" />
                        <div className="bg-neon-teal/10 border border-neon-teal/30 rounded-lg w-12 h-12 flex items-center justify-center mb-6 shadow-[0_0_20px_rgba(0,255,204,0.1)] group-hover:shadow-[0_0_20px_rgba(0,255,204,0.3)] transition-all">
                            <Fan className="text-neon-teal" />
                        </div>
                        <h3 className="font-heading font-semibold text-2xl mb-2 text-white">Silent Propulsion & VBS</h3>
                        <p className="text-stardust text-sm leading-relaxed mb-6 relative z-10">
                            Single rear Rim-Driven Thruster (RDT) eliminates shaft-line cavitation noise. Piston-based Variable Buoyancy System (VBS) enables zero-power neutral loiter and near-silent depth control.
                        </p>
                        <a href="#" className="text-electric-blue font-mono text-sm tracking-wider uppercase hover:underline relative z-10 mt-auto">VIEW PROPULSION <span className="text-neon-teal">&rarr;</span></a>
                    </div>

                    <div className="group bg-navy-blue/20 backdrop-blur-md p-8 rounded-2xl border border-slate-700/50 hover:-translate-y-1 hover:border-[#00D2FF]/15 hover:shadow-[0_0_15px_rgba(0,210,255,0.15)] transition-all duration-300 relative overflow-hidden flex flex-col h-full">
                        <Radio className="absolute -bottom-4 -right-4 w-32 h-32 text-electric-blue opacity-5 group-hover:opacity-10 transition-all duration-300 transform -rotate-12" />
                        <div className="bg-electric-blue/10 border border-electric-blue/30 rounded-lg w-12 h-12 flex items-center justify-center mb-6 shadow-[0_0_20px_rgba(0,240,255,0.15)] group-hover:shadow-[0_0_20px_rgba(0,240,255,0.4)] transition-all">
                            <Radio className="text-electric-blue" />
                        </div>
                        <h3 className="font-heading font-semibold text-2xl mb-2 text-white">Sensing & Micro-Telemetry</h3>
                        <p className="text-stardust text-sm leading-relaxed mb-6 relative z-10">
                            Flush-mounted nose hydrophone array captures raw cavitation signatures. Onboard inference compresses terabytes of audio into kilobyte tactical metadata for burst SATCOM relay.
                        </p>
                        <a href="#" className="text-electric-blue font-mono text-sm tracking-wider uppercase hover:underline relative z-10 mt-auto">VIEW SENSOR ARRAY <span className="text-neon-teal">&rarr;</span></a>
                    </div>
                </section>
            </div>

            <HardwareArchitecture />

            {/* Data Input Terminal - Full Width Background */}
            <section id="telemetry-section" className="py-24 border-y border-slate-800 bg-cover bg-center bg-no-repeat relative" style={{ backgroundImage: `url(${terminalBg})` }}>
                {/* Optional subtle overlay to ensure terminal remains legible if image is bright */}
                <div className="absolute inset-0 bg-[#020617]/40 z-0"></div>
                <div className="relative z-10 max-w-7xl mx-auto px-6">
                    <div className="max-w-3xl mx-auto">
                        <div className="flex items-center gap-3 mb-6">
                            <h2 className="font-mono font-bold text-xl text-electric-blue tracking-wide">⛯ Subsea Node Telemetry & Subsystem Diagnostics</h2>
                        </div>
                        <div className="glass-card bg-navy-blue/20 backdrop-blur-md p-8 rounded-xl border border-slate-700/50 hover:border-[#00D2FF]/40 hover:shadow-[0_0_30px_-10px_rgba(0,210,255,0.15)] transition-all duration-300 relative flex flex-col h-full">
                            {/* Decorative Corner Accents */}
                            <div className="absolute top-0 left-0 w-4 h-4 border-t-2 border-l-2 border-neon-teal rounded-tl-xl"></div>
                            <div className="absolute top-0 right-0 w-4 h-4 border-t-2 border-r-2 border-neon-teal rounded-tr-xl"></div>
                            <div className="absolute bottom-0 left-0 w-4 h-4 border-b-2 border-l-2 border-neon-teal rounded-bl-xl"></div>
                            <div className="absolute bottom-0 right-0 w-4 h-4 border-b-2 border-r-2 border-neon-teal rounded-br-xl"></div>
                            
                            <div className="space-y-8">
                                <div className="space-y-2">
                                    <label className="font-mono text-xs text-stardust tracking-widest uppercase block">TACTICAL NODE ID / DESIGNATION</label>
                                    <input 
                                        type="text" 
                                        value="AUSV-WS-01 - SUBSEA_WINGMAN_ALPHA" 
                                        readOnly
                                        className="w-full bg-transparent border-b border-electric-blue/30 h-10 px-2 text-neon-teal font-mono text-sm outline-none cursor-default"
                                    />
                                </div>
                                <div className="space-y-2">
                                    <div className="flex justify-between items-center mb-2">
                                        <label className="font-mono text-xs text-stardust tracking-widest uppercase block">SUBSYSTEM STATUS & LIVE DIAGNOSTICS</label>
                                        <div className="flex items-center gap-2 text-[#00D2FF] font-mono text-xs animate-pulse">
                                            <span className="w-2 h-2 bg-[#00D2FF] rounded-full shadow-[0_0_8px_#00D2FF]"></span>
                                            LIVE STREAM
                                        </div>
                                    </div>
                                    <div className="w-full bg-[#020617] border border-electric-blue/20 rounded-md p-5 shadow-[inset_0_0_20px_rgba(0,240,255,0.05)] overflow-x-auto text-left">
                                        <pre className="text-white font-mono text-sm leading-loose text-left">
                                            {`POWER       : 48V Li-ion Pack (2.5 kWh) | BMS Nominal | 94% Capacity
PROPULSION  : Rim-Driven Thruster: Standby | RPM: 0 | Low-Noise Idle
BUOYANCY    : Piston VBS: Neutral Trim (Depth: 45.2m / Rating: 300m)
COMPUTE     : Jetson Orin Nano: 34°C | Inference: ACTIVE
CLASSIFIER  : Target: Commercial Trawler | Conf: 94.2%
TELEMETRY   : Payload Buffer: 1.4 KB Compressed Metadata | SATCOM: Queued`}
                                        </pre>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </section>

            <div className="relative z-10 max-w-7xl mx-auto px-6 pb-24">
                {/* Hardware Comparison Table */}
                <section id="specs-section" className="pt-16 border-slate-800">
                    <div className="max-w-5xl mx-auto">
                        <div className="mb-8">
                            <h2 className="font-mono font-bold text-xl text-white tracking-wide uppercase">Asymmetric Attritability - Competitor Hardware Benchmark</h2>
                        </div>
                        <div className="glass-card bg-navy-blue/20 backdrop-blur-md p-8 rounded-xl border border-slate-700/50 hover:border-[#00D2FF]/40 hover:shadow-[0_0_30px_-10px_rgba(0,210,255,0.15)] transition-all duration-300 relative overflow-x-auto">
                            <table className="w-full text-left border-collapse font-mono text-sm">
                                <thead>
                                    <tr>
                                        <th className="p-4 border-b border-slate-700/50 text-stardust font-medium w-1/4">Subsystem Spec</th>
                                        <th className="p-4 border-b border-electric-blue/50 bg-navy-blue/40 text-electric-blue font-bold w-2/5 shadow-[inset_0_-1px_0_rgba(0,240,255,0.5)]">WaveSentinel AUSV</th>
                                        <th className="p-4 border-b border-slate-700/50 text-slate-500 font-medium w-auto">Legacy Defense UUVs (HII / Teledyne)</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    <tr className="border-b border-slate-800/50 hover:bg-white/[0.02] transition-colors">
                                        <td className="p-4 text-stardust">Unit Cost</td>
                                        <td className="p-4 bg-navy-blue/20 text-neon-teal font-semibold">~$20,100 (Production)</td>
                                        <td className="p-4 text-slate-500">$500,000 – $1,500,000+</td>
                                    </tr>
                                    <tr className="border-b border-slate-800/50 hover:bg-white/[0.02] transition-colors">
                                        <td className="p-4 text-stardust">Propulsion Type</td>
                                        <td className="p-4 bg-navy-blue/20 text-white">Rim-Driven Thruster (Shaftless, Low-Noise)</td>
                                        <td className="p-4 text-slate-500">Conventional Bladed Propellers</td>
                                    </tr>
                                    <tr className="border-b border-slate-800/50 hover:bg-white/[0.02] transition-colors">
                                        <td className="p-4 text-stardust">Depth Control</td>
                                        <td className="p-4 bg-navy-blue/20 text-white">Piston-Based VBS (Silent Neutral Loiter)</td>
                                        <td className="p-4 text-slate-500">Thruster Dynamic Diving / Ballast Pumps</td>
                                    </tr>
                                    <tr className="border-b border-slate-800/50 hover:bg-white/[0.02] transition-colors">
                                        <td className="p-4 text-stardust">Edge Intelligence</td>
                                        <td className="p-4 bg-navy-blue/20 text-white">NVIDIA Jetson Orin Nano + EfficientNet</td>
                                        <td className="p-4 text-slate-500">Dedicated Mission Computers / Post-Mission Processing</td>
                                    </tr>
                                    <tr className="hover:bg-white/[0.02] transition-colors">
                                        <td className="p-4 text-stardust">Data Exfiltration</td>
                                        <td className="p-4 bg-navy-blue/20 text-white">Edge-Compressed Micro-Payloads (&lt;2 KB)</td>
                                        <td className="p-4 text-slate-500">High-Bandwidth Raw Acoustic Exfil / Fiber-Tethers</td>
                                    </tr>
                                </tbody>
                            </table>
                        </div>
                    </div>
                </section>
            </div>

            {/* Hardware Scalability Section (Part 1) */}
            <section className="pt-24 pb-32 relative overflow-hidden w-full">
                <div className="absolute inset-0 z-0">
                    <Grainient
                        color1="#020813"
                        color2="#002244"
                        color3="#00D2FF"
                        timeSpeed={0.5}
                        colorBalance={-0.2}
                        warpStrength={1.5}
                        warpFrequency={3.0}
                        warpSpeed={1.5}
                        warpAmplitude={60.0}
                        blendAngle={45.0}
                        blendSoftness={0.6}
                        rotationAmount={300.0}
                        noiseScale={1.5}
                        grainAmount={0.12}
                        grainScale={1.5}
                        grainAnimated={true}
                        contrast={1.2}
                        gamma={1.0}
                        saturation={0.7}
                        centerX={0.0}
                        centerY={0.0}
                        zoom={1.2}
                    />
                </div>
                <div className="relative z-10 max-w-7xl mx-auto px-6">
                    <div className="max-w-5xl mx-auto">
                        <div className="mb-12">
                            <h2 className="font-mono font-bold text-xl text-electric-blue tracking-wide uppercase mb-3">
                                STRATEGIC HARDWARE SCALABILITY : FLEET EVOLUTION
                            </h2>
                            <p className="text-stardust text-sm leading-relaxed max-w-3xl">
                                Transitioning from hardware initialization of a single autonomous picket to a distributed subsea acoustic shield.
                            </p>
                        </div>

                        <div className="grid grid-cols-1 md:grid-cols-3 gap-8 relative">
                            {/* Connectors */}
                            <div className="hidden md:flex absolute top-1/2 left-[33.33%] -translate-x-1/2 -translate-y-1/2 text-slate-700 font-mono text-lg tracking-[0.2em] pointer-events-none z-10">&gt;&gt;&gt;</div>
                            <div className="hidden md:flex absolute top-1/2 left-[66.66%] -translate-x-1/2 -translate-y-1/2 text-slate-700 font-mono text-lg tracking-[0.2em] pointer-events-none z-10">&gt;&gt;&gt;</div>

                            {/* Column 1 */}
                            <div className="glass-card bg-navy-blue/20 backdrop-blur-md p-8 rounded-xl border border-slate-700/50 hover:border-[#00D2FF]/40 hover:shadow-[0_0_30px_-10px_rgba(0,210,255,0.15)] transition-all duration-300 relative flex flex-col h-full">
                                <div className="flex justify-between items-start mb-4 gap-2">
                                    <div className="text-stardust font-mono text-[10px] tracking-widest uppercase">Hardware Initialization<br/>(Current Baseline)</div>
                                    <span className="px-2 py-1 rounded-full border border-[#00D2FF] bg-[#00D2FF]/10 text-[#00D2FF] text-[10px] font-mono tracking-wider font-bold whitespace-nowrap">CURRENT : TRL 4</span>
                                </div>
                                <h3 className="font-heading font-semibold text-2xl text-white mb-4 leading-normal">Block I: Autonomous Picket</h3>
                                <p className="text-stardust text-sm leading-relaxed mb-6">
                                    Current hardware realization prioritizing low-cost attritability and acoustic stealth.
                                </p>
                                <ul className="space-y-3 mt-auto">
                                    {[
                                        "Jetson Orin Nano (24V)",
                                        "Rim-Driven Thruster",
                                        "Piston-Based VBS",
                                        "SATCOM Telemetry Mast"
                                    ].map((spec, i) => (
                                        <li key={i} className="flex items-center gap-3 text-sm text-white font-mono">
                                            <ChevronRight className="w-4 h-4 text-[#00D2FF]" />
                                            {spec}
                                        </li>
                                    ))}
                                </ul>
                            </div>

                            {/* Column 2 */}
                            <div className="glass-card bg-navy-blue/20 backdrop-blur-md p-8 rounded-xl border border-slate-700/50 hover:border-[#00D2FF]/40 hover:shadow-[0_0_30px_-10px_rgba(0,210,255,0.15)] transition-all duration-300 relative flex flex-col h-full">
                                <div className="flex justify-between items-start mb-4 gap-2">
                                    <div className="text-stardust font-mono text-[10px] tracking-widest uppercase">Tactical Covert Upgrades<br/>(Near-Term)</div>
                                    <span className="px-2 py-1 rounded-full border border-slate-500 bg-slate-500/10 text-slate-300 text-[10px] font-mono tracking-wider font-bold whitespace-nowrap">NEAR-TERM : TRL 5-6</span>
                                </div>
                                <h3 className="font-heading font-semibold text-2xl text-white mb-4 leading-normal">Block II: Zero-Exposure Operations</h3>
                                <p className="text-stardust text-sm leading-relaxed mb-6">
                                    Eliminating surfacing vulnerabilities and acoustic blind spots.
                                </p>
                                <ul className="space-y-3 mt-auto">
                                    {[
                                        "Submerged Acoustic Modems",
                                        "Conformal Flank Sensor Arrays",
                                        "Phase-Change Material (PCM) Thermal Sinks"
                                    ].map((spec, i) => (
                                        <li key={i} className="flex items-center gap-3 text-sm text-white font-mono">
                                            <ChevronRight className="w-4 h-4 text-[#00D2FF]" />
                                            {spec}
                                        </li>
                                    ))}
                                </ul>
                            </div>

                            {/* Column 3 */}
                            <div className="glass-card bg-navy-blue/20 backdrop-blur-md p-8 rounded-xl border border-slate-700/50 hover:border-[#00D2FF]/40 hover:shadow-[0_0_30px_-10px_rgba(0,210,255,0.15)] transition-all duration-300 relative flex flex-col h-full">
                                <div className="flex justify-between items-start mb-4 gap-2">
                                    <div className="text-stardust font-mono text-[10px] tracking-widest uppercase">The Ultimate Vision<br/>(Future)</div>
                                    <span className="px-2 py-1 rounded-full border border-neon-teal bg-neon-teal/10 text-neon-teal text-[10px] font-mono tracking-wider font-bold whitespace-nowrap">FLEET SWARM : TRL 7+</span>
                                </div>
                                <h3 className="font-heading font-semibold text-2xl text-white mb-4 leading-normal">Block III: Subsea Loyal Wingman Swarm</h3>
                                <p className="text-stardust text-sm leading-relaxed mb-6">
                                    Multi-node distributed architecture for comprehensive maritime defense.
                                </p>
                                <ul className="space-y-3 mt-auto">
                                    {[
                                        "Mothership AI Fusion Center",
                                        "Multi-Vehicle Edge-Mesh Network",
                                        "Staggered 24-Hour Fleet Rotation"
                                    ].map((spec, i) => (
                                        <li key={i} className="flex items-center gap-3 text-sm text-white font-mono">
                                            <ChevronRight className="w-4 h-4 text-[#00D2FF]" />
                                            {spec}
                                        </li>
                                    ))}
                                </ul>
                            </div>
                        </div>
                    </div>
                </div>
            </section>
        </div>
    );
}
