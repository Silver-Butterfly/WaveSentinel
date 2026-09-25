import React from 'react';
// Fallback image since the original asset is missing
const cutawayImg = "https://images.unsplash.com/photo-1544827055-321a4fdb7bfd?auto=format&fit=crop&w=1920&q=80";

export default function HardwareArchitecture() {
    const glossary = [
        {
            title: "I. SENSORY & COMPUTE",
            items: [
                { name: "HYDROPHONE ARRAY", desc: "16-channel passive sensor capturing raw acoustic anomalies." },
                { name: "LOW NOISE PREAMP", desc: "Amplifies faint signals without electronic static." },
                { name: "PRESSURE SENSOR", desc: "Real-time depth data for flight and buoyancy systems." },
                { name: "POWER DISTRIBUTION", desc: "Routes 48V into stable 24V/12V/5V rails." },
                { name: "IMU", desc: "Tracks pitch, roll, yaw, and acceleration." },
                { name: "STM32 CONTROLLER", desc: "Manages physical vehicle stability and dive operations." },
                { name: "GPU EDGE COMPUTE", desc: "Jetson Orin Nano running local AI acoustic classification." },
                { name: "SSD STORAGE", desc: "Buffers massive raw audio files prior to compression." }
            ]
        },
        {
            title: "II. ENERGY & BUOYANCY",
            items: [
                { name: "48V LI-ION PACK", desc: "High-density storage for 12–24 hours of endurance." },
                { name: "BATTERY MANAGEMENT", desc: "Actively balances voltage and monitors thermal limits." },
                { name: "VARIABLE BUOYANCY", desc: "Mechanical system for silent vertical movement/hovering." },
                { name: "OIL RESERVOIR", desc: "Stores hydraulic fluid for the displacement system." },
                { name: "LINEAR ACTUATOR", desc: "Electric motor/lead screw driving the displacement piston." },
                { name: "FLEXIBLE BLADDER", desc: "Expands/contracts to physically alter water displacement." }
            ]
        },
        {
            title: "III. TELEMETRY & PROPULSION",
            items: [
                { name: "RETRACTABLE MAST", desc: "Telescoping structure for surface telemetry updates." },
                { name: "SATCOM ANTENNA", desc: "Transmits compressed tactical metadata to fleet command." },
                { name: "MOTOR DRIVER", desc: "Translates digital commands into precise electrical currents." },
                { name: "ESC", desc: "Regulates exact RPM and torque of the propulsion unit." },
                { name: "WATERPROOF CONNECTORS", desc: "Pressure-rated bulkheads for dry-to-wet wire routing." },
                { name: "RIM-DRIVEN THRUSTER", desc: "Hubless, shaftless motor eliminating cavitation noise." }
            ]
        }
    ];

    return (
        <section className="relative w-full bg-[#000000] pt-16 pb-[100px]">
            <div className="relative z-10 w-full max-w-[1200px] mx-auto px-6">
                
                <div className="mb-12 text-center">
                    <h2 className="font-mono font-bold text-2xl text-white tracking-widest uppercase">
                        HARDWARE ARCHITECTURE
                    </h2>
                </div>

                <div className="relative w-full mx-auto flex justify-center items-center mb-16">
                    <img 
                        src={cutawayImg} 
                        alt="WaveSentinel AUSV Hardware Cutaway" 
                        className="w-full h-auto object-contain" 
                    />
                </div>

                {/* Glossary Grid */}
                <div className="grid grid-cols-1 md:grid-cols-3 gap-x-8 gap-y-12 text-[12px] leading-snug">
                    {glossary.map((col, colIndex) => (
                        <div key={`col-${colIndex}`} className="glass-card bg-navy-blue/20 backdrop-blur-md p-6 rounded-xl border border-slate-700/50 hover:border-[#00D2FF]/40 hover:shadow-[0_0_30px_-10px_rgba(0,210,255,0.15)] transition-all duration-300 relative flex flex-col gap-4">
                            <h3 className="font-mono font-bold text-[#00D2FF] mb-2 tracking-wider">
                                {col.title}
                            </h3>
                            {col.items.map((item, itemIndex) => (
                                <div key={`item-${colIndex}-${itemIndex}`} className="flex flex-col mb-1.5">
                                    <span className="font-mono font-semibold text-[#00D2FF] uppercase mb-0.5">
                                        {item.name}
                                    </span>
                                    <span className="font-sans text-[#8892B0]">
                                        {item.desc}
                                    </span>
                                </div>
                            ))}
                        </div>
                    ))}
                </div>

            </div>
        </section>
    );
}
