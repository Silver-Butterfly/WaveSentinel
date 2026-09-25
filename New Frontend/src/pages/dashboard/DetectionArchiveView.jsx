import React, { useState } from 'react';
import { Search, Filter, Database, X, Download, Save, Eye, MapPin, Calendar, Clock } from 'lucide-react';
import { useSimulation } from '../../simulation/SimulationContext.jsx';
import sonarImg from '../../assets/sim2d/ocean.jpg';

export default function DetectionArchiveView() {
    const { detections } = useSimulation();

    const [searchQuery, setSearchQuery] = useState('');
    const [selectedClass, setSelectedClass] = useState('ALL');
    const [selectedRisk, setSelectedRisk] = useState('ALL');
    const [selectedItem, setSelectedItem] = useState(null);
    const [operatorNotes, setOperatorNotes] = useState('');

    const filteredData = detections.filter(item => {
        const matchesQuery = item.frame_id.toLowerCase().includes(searchQuery.toLowerCase()) || 
                             item.class.toLowerCase().includes(searchQuery.toLowerCase()) ||
                             item.mission_id.toLowerCase().includes(searchQuery.toLowerCase());
        const matchesClass = selectedClass === 'ALL' || item.class.toUpperCase() === selectedClass.toUpperCase();
        const matchesRisk = selectedRisk === 'ALL' || item.risk.toUpperCase() === selectedRisk.toUpperCase();
        return matchesQuery && matchesClass && matchesRisk;
    });

    const openDetail = (item) => {
        setSelectedItem(item);
        setOperatorNotes(item.operator_notes || '');
    };

    return (
        <div className="space-y-8 animate-fade-in pb-12 relative font-mono">
            {/* Header */}
            <div className="glass-card p-6 flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
                <div className="space-y-1">
                    <div className="flex items-center gap-3">
                        <Database className="text-cyan-400" size={24} />
                        <h2 className="text-2xl font-bold text-white tracking-wide uppercase">
                            DETECTION ARCHIVE
                        </h2>
                        <span className="px-3 py-1 rounded-full bg-cyan-500/20 border border-cyan-400/40 text-cyan-300 text-xs font-bold uppercase">
                            SUBSEA EVIDENCE PROVENANCE LIBRARY
                        </span>
                    </div>
                    <p className="text-slate-300 text-xs">
                        Historical Mission Detection Records & Provenance Audit Manifest
                    </p>
                </div>
            </div>

            {/* Search and Filters Bar */}
            <div className="glass-card p-4 flex flex-col md:flex-row items-center justify-between gap-4 text-xs">
                {/* Search Bar */}
                <div className="relative w-full md:w-96">
                    <Search size={16} className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-400" />
                    <input 
                        type="text" 
                        placeholder="Search frames, objects, missions..." 
                        value={searchQuery}
                        onChange={(e) => setSearchQuery(e.target.value)}
                        className="w-full bg-black/40 border border-white/10 rounded-xl py-2 pl-9 pr-4 text-white placeholder-slate-500 focus:outline-none focus:border-cyan-400/50"
                    />
                </div>

                {/* Filter Pills */}
                <div className="flex items-center gap-3 w-full md:w-auto">
                    <select 
                        value={selectedClass}
                        onChange={(e) => setSelectedClass(e.target.value)}
                        className="bg-black/40 border border-white/10 rounded-xl py-2 px-3 text-slate-300 focus:outline-none focus:border-cyan-400/50 cursor-pointer"
                    >
                        <option value="ALL">All Classes</option>
                        <option value="SHIPWRECK">Shipwreck</option>
                        <option value="PIPE">Pipe</option>
                        <option value="MINE">Mine</option>
                        <option value="GHOST NET">Ghost Net</option>
                    </select>

                    <select 
                        value={selectedRisk}
                        onChange={(e) => setSelectedRisk(e.target.value)}
                        className="bg-black/40 border border-white/10 rounded-xl py-2 px-3 text-slate-300 focus:outline-none focus:border-cyan-400/50 cursor-pointer"
                    >
                        <option value="ALL">All Risk Levels</option>
                        <option value="CRITICAL">Critical Risk</option>
                        <option value="HIGH">High Risk</option>
                        <option value="MEDIUM">Medium Risk</option>
                        <option value="LOW">Low Risk</option>
                    </select>
                </div>
            </div>

            {/* Detection Cards Table with Enhanced Provenance Columns */}
            <div className="glass-card p-6 text-xs">
                <div className="overflow-x-auto">
                    <table className="w-full text-left border-collapse">
                        <thead>
                            <tr className="border-b border-white/10 text-slate-400 uppercase">
                                <th className="py-3 px-4">FRAME</th>
                                <th className="py-3 px-4">OBJECT</th>
                                <th className="py-3 px-4 text-right">CONFIDENCE</th>
                                <th className="py-3 px-4">RISK</th>
                                <th className="py-3 px-4 text-right">TIME</th>
                                <th className="py-3 px-4">SOURCE</th>
                                <th className="py-3 px-4">QUALITY</th>
                                <th className="py-3 px-4">MOTION</th>
                                <th className="py-3 px-4">POSITION</th>
                            </tr>
                        </thead>
                        <tbody>
                            {filteredData.map((item, idx) => (
                                <tr 
                                    key={idx}
                                    onClick={() => openDetail(item)}
                                    className="border-b border-white/5 hover:bg-white/5 cursor-pointer transition-colors font-mono"
                                >
                                    <td className="py-3.5 px-4 font-bold text-cyan-300">{item.frame_id}</td>
                                    <td className="py-3.5 px-4 text-white font-semibold">{item.class}</td>
                                    <td className="py-3.5 px-4 text-right font-bold text-white">{item.confidence}%</td>
                                    <td className="py-3.5 px-4">
                                        <span className={`px-2.5 py-0.5 rounded text-[10px] font-bold ${
                                            item.risk === 'CRITICAL' ? 'bg-red-600/20 text-red-400 border border-red-500' :
                                            item.risk === 'HIGH' ? 'bg-red-500/20 text-red-300 border border-red-500/30' :
                                            item.risk === 'MEDIUM' ? 'bg-amber-500/20 text-amber-300 border border-amber-500/30' :
                                            'bg-emerald-500/20 text-emerald-300 border border-emerald-500/30'
                                        }`}>
                                            {item.risk}
                                        </span>
                                    </td>
                                    <td className="py-3.5 px-4 text-right text-slate-400">{item.timestamp}</td>
                                    <td className="py-3.5 px-4 text-cyan-400 font-bold uppercase">{item.position_source}</td>
                                    <td className="py-3.5 px-4 text-emerald-400 uppercase font-bold">{item.input_quality || 'GOOD'}</td>
                                    <td className="py-3.5 px-4 text-slate-300 uppercase">{item.motion_correction_state}</td>
                                    <td className="py-3.5 px-4 text-cyan-300 font-bold uppercase">{item.position_source}</td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
            </div>

            {/* Detection Detail Modal with Provenance Specification */}
            {selectedItem && (
                <div className="fixed inset-0 bg-black/80 backdrop-blur-md z-50 flex items-center justify-center p-4">
                    <div className="glass-card max-w-2xl w-full p-6 space-y-6 max-h-[90vh] overflow-y-auto animate-fade-in font-mono">
                        <div className="flex items-center justify-between border-b border-white/10 pb-3">
                            <div className="flex items-center gap-3">
                                <h3 className="font-bold text-white uppercase text-base">
                                    DETECTION DETAIL : {selectedItem.frame_id}
                                </h3>
                                <span className="px-2 py-0.5 rounded bg-cyan-500/20 text-cyan-300 text-xs font-bold">
                                    {selectedItem.mission_id}
                                </span>
                            </div>
                            <button 
                                onClick={() => setSelectedItem(null)}
                                className="text-slate-400 hover:text-white p-1 cursor-pointer"
                            >
                                <X size={20} />
                            </button>
                        </div>

                        {/* Sonar Frame Preview */}
                        <div className="relative h-64 rounded-xl overflow-hidden bg-black border border-white/15">
                            <img src={sonarImg} alt="Detection Detail Frame" className="w-full h-full object-cover filter brightness-90 contrast-125" />
                            <div className="absolute top-1/4 left-1/3 w-44 h-28 border-2 border-cyan-400 bg-cyan-400/10 rounded flex items-start p-1.5 shadow-[0_0_20px_rgba(0,240,255,0.3)]">
                                <span className="bg-cyan-500 text-black font-extrabold text-[10px] px-1.5 py-0.5 rounded uppercase">
                                    {selectedItem.class} {selectedItem.confidence}%
                                </span>
                            </div>
                        </div>

                        {/* Detailed Provenance Specifications */}
                        <div className="space-y-2 border-b border-white/10 pb-4">
                            <div className="text-slate-400 text-[11px] font-bold uppercase">PROVENANCE & HARDWARE METADATA</div>
                            <div className="grid grid-cols-2 sm:grid-cols-3 gap-3 text-xs">
                                <div className="p-3 rounded-lg bg-black/40 border border-white/5 space-y-0.5">
                                    <div className="text-slate-400 text-[10px]">MODEL</div>
                                    <div className="text-white font-bold">{selectedItem.model_version}</div>
                                </div>
                                <div className="p-3 rounded-lg bg-black/40 border border-white/5 space-y-0.5">
                                    <div className="text-slate-400 text-[10px]">INPUT</div>
                                    <div className="text-cyan-300 font-bold uppercase">Simulated SSS</div>
                                </div>
                                <div className="p-3 rounded-lg bg-black/40 border border-white/5 space-y-0.5">
                                    <div className="text-slate-400 text-[10px]">TELEMETRY</div>
                                    <div className="text-cyan-300 font-bold uppercase">Simulation</div>
                                </div>
                                <div className="p-3 rounded-lg bg-black/40 border border-white/5 space-y-0.5">
                                    <div className="text-slate-400 text-[10px]">MOTION</div>
                                    <div className="text-white font-bold uppercase">Metadata Only</div>
                                </div>
                                <div className="p-3 rounded-lg bg-black/40 border border-white/5 space-y-0.5">
                                    <div className="text-slate-400 text-[10px]">POSITION</div>
                                    <div className="text-cyan-300 font-bold uppercase">Simulated</div>
                                </div>
                                <div className="p-3 rounded-lg bg-black/40 border border-white/5 space-y-0.5">
                                    <div className="text-slate-400 text-[10px]">FRAME QUALITY</div>
                                    <div className="text-emerald-400 font-bold uppercase">Good</div>
                                </div>
                            </div>
                        </div>

                        {/* GPS and Depth */}
                        <div className="p-3 rounded-lg bg-black/40 border border-white/10 text-xs flex justify-between items-center text-slate-300">
                            <span>COORDINATES: <strong className="text-white">{selectedItem.latitude}, {selectedItem.longitude}</strong> (SIMULATED)</span>
                            <span>DEPTH: <strong className="text-white">{selectedItem.depth}</strong></span>
                        </div>

                        {/* Operator Notes */}
                        <div className="space-y-1.5 text-xs">
                            <label className="text-slate-400 uppercase font-bold">Operator Notes:</label>
                            <textarea 
                                value={operatorNotes}
                                onChange={(e) => setOperatorNotes(e.target.value)}
                                rows={3}
                                className="w-full bg-black/50 border border-white/10 rounded-xl p-3 text-white placeholder-slate-500 focus:outline-none focus:border-cyan-400/50"
                            />
                        </div>

                        {/* Footer Buttons */}
                        <div className="flex justify-end gap-3 pt-2 text-xs">
                            <button 
                                onClick={() => setSelectedItem(null)}
                                className="px-4 py-2 rounded-xl bg-white/10 hover:bg-white/20 text-white font-bold cursor-pointer"
                            >
                                Close
                            </button>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
}
