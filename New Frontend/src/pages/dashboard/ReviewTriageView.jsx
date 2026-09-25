import React, { useState } from 'react';
import { 
    CheckCircle2, 
    XCircle, 
    Flag, 
    ShieldAlert, 
    Check, 
    X, 
    Edit3, 
    Save, 
    Layers, 
    ChevronLeft, 
    ChevronRight,
    MessageSquare,
    Info
} from 'lucide-react';
import { useSimulation } from '../../simulation/SimulationContext.jsx';
import sonarImg from '../../assets/sim2d/ocean.jpg';

export default function ReviewTriageView() {
    const { 
        detections, 
        confirmDetection, 
        dismissDetection, 
        flagDetection, 
        updateNotes 
    } = useSimulation();

    // Filter pending items or items in review queue
    const reviewQueue = detections.filter(d => d.review_status === 'pending' || d.review_status === 'flagged');
    const [currentIndex, setCurrentIndex] = useState(0);

    const activeItem = reviewQueue[currentIndex] || reviewQueue[0] || detections[0];
    const [notesInput, setNotesInput] = useState(activeItem?.operator_notes || '');

    // Synchronize local notes state when active item changes
    React.useEffect(() => {
        if (activeItem) {
            setNotesInput(activeItem.operator_notes || '');
        }
    }, [activeItem?.frame_id]);

    const handleConfirm = () => {
        if (!activeItem) return;
        confirmDetection(activeItem.frame_id);
        if (currentIndex > 0) setCurrentIndex(prev => prev - 1);
    };

    const handleDismiss = () => {
        if (!activeItem) return;
        dismissDetection(activeItem.frame_id);
        if (currentIndex > 0) setCurrentIndex(prev => prev - 1);
    };

    const handleFlag = () => {
        if (!activeItem) return;
        flagDetection(activeItem.frame_id);
    };

    const handleSaveNotes = () => {
        if (!activeItem) return;
        updateNotes(activeItem.frame_id, notesInput);
        alert('Operator notes updated successfully.');
    };

    return (
        <div className="space-y-8 animate-fade-in pb-12 font-mono">
            {/* Header Banner */}
            <div className="glass-card p-6 flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
                <div className="space-y-1">
                    <div className="flex items-center gap-3">
                        <ShieldAlert className="text-amber-400" size={24} />
                        <h2 className="text-2xl font-bold text-white tracking-wide uppercase">
                            REVIEW & TRIAGE
                        </h2>
                        <span className="px-3 py-1 rounded-full bg-amber-500/20 border border-amber-400/50 text-amber-300 text-xs font-bold uppercase tracking-wider">
                            HUMAN-IN-THE-LOOP TRIAGE
                        </span>
                    </div>
                    <p className="text-slate-300 text-xs">
                        Operator verification pipeline for candidate sonar debris detections. Model output is advisory.
                    </p>
                </div>

                <div className="flex items-center gap-3 text-xs">
                    <div className="bg-black/50 px-4 py-2 rounded-xl border border-amber-500/40 text-amber-300 font-bold flex items-center gap-2">
                        <span className="w-2 h-2 rounded-full bg-amber-400 animate-ping"></span>
                        {reviewQueue.length} DETECTIONS REQUIRE REVIEW
                    </div>
                </div>
            </div>

            {/* Main Stage: Triage Card + Queue List */}
            {reviewQueue.length === 0 ? (
                <div className="glass-card p-12 text-center space-y-4">
                    <div className="w-16 h-16 rounded-full bg-emerald-500/20 border border-emerald-400 flex items-center justify-center text-emerald-400 mx-auto">
                        <CheckCircle2 size={36} />
                    </div>
                    <h3 className="text-xl font-bold text-white">QUEUE CLEARED</h3>
                    <p className="text-slate-300 text-xs max-w-md mx-auto">
                        All simulated side-scan sonar detections have been verified or triaged by operator command.
                    </p>
                </div>
            ) : (
                <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
                    {/* Left 2 Columns: Active Detection Inspection Box */}
                    <div className="lg:col-span-2 glass-card p-6 space-y-6">
                        <div className="flex items-center justify-between border-b border-white/10 pb-4">
                            <div className="space-y-1">
                                <div className="text-slate-400 text-xs">FRAME ID</div>
                                <div className="text-xl font-bold text-white">{activeItem.frame_id}</div>
                            </div>

                            <div className="flex items-center gap-3">
                                <span className={`px-3 py-1 rounded text-xs font-bold uppercase ${
                                    activeItem.risk === 'CRITICAL' ? 'bg-red-600/20 text-red-400 border border-red-500' :
                                    activeItem.risk === 'HIGH' ? 'bg-red-500/20 text-red-300 border border-red-500/40' :
                                    'bg-amber-500/20 text-amber-300 border border-amber-500/40'
                                }`}>
                                    RISK: {activeItem.risk}
                                </span>
                            </div>
                        </div>

                        {/* Sonar Image Frame Viewport */}
                        <div className="relative h-72 rounded-2xl overflow-hidden bg-black border border-cyan-500/30">
                            <img 
                                src={sonarImg} 
                                alt="Side Scan Sonar Frame" 
                                className="w-full h-full object-cover filter brightness-90 contrast-125"
                            />

                            {/* Bounding Box Overlay */}
                            <div className="absolute top-1/4 left-1/3 w-56 h-32 border-2 border-cyan-400 bg-cyan-400/10 rounded flex flex-col justify-between p-2 shadow-[0_0_25px_rgba(0,240,255,0.4)]">
                                <span className="bg-cyan-500 text-black font-extrabold text-[11px] px-2 py-0.5 rounded self-start uppercase tracking-wider">
                                    {activeItem.class} {activeItem.confidence}%
                                </span>
                                <div className="text-[10px] text-cyan-300 bg-black/80 px-2 py-0.5 rounded self-start font-mono">
                                    {activeItem.latitude} | {activeItem.longitude}
                                </div>
                            </div>

                            <div className="absolute bottom-3 left-3 bg-black/80 backdrop-blur-md px-3 py-1.5 rounded-lg border border-white/10 text-slate-300 text-xs flex items-center gap-2">
                                <span>DEPTH: <strong className="text-white">{activeItem.depth}</strong></span>
                                <span>|</span>
                                <span className="text-cyan-300 font-bold">SIMULATED SSS</span>
                            </div>
                        </div>

                        {/* Provenance Metadata Grid */}
                        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs">
                            <div className="p-3 rounded-xl bg-black/40 border border-white/5 space-y-0.5">
                                <div className="text-slate-400 text-[10px]">INPUT QUALITY</div>
                                <div className="text-emerald-400 font-bold uppercase">{activeItem.input_quality || 'GOOD'}</div>
                            </div>

                            <div className="p-3 rounded-xl bg-black/40 border border-white/5 space-y-0.5">
                                <div className="text-slate-400 text-[10px]">MOTION STATE</div>
                                <div className="text-cyan-300 font-bold uppercase">{activeItem.motion_correction_state}</div>
                            </div>

                            <div className="p-3 rounded-xl bg-black/40 border border-white/5 space-y-0.5">
                                <div className="text-slate-400 text-[10px]">POSITION SOURCE</div>
                                <div className="text-cyan-300 font-bold uppercase">{activeItem.position_source}</div>
                            </div>

                            <div className="p-3 rounded-xl bg-black/40 border border-white/5 space-y-0.5">
                                <div className="text-slate-400 text-[10px]">TELEMETRY SOURCE</div>
                                <div className="text-cyan-300 font-bold uppercase">{activeItem.telemetry_source}</div>
                            </div>
                        </div>

                        {/* Operator Notes Box */}
                        <div className="space-y-2 text-xs">
                            <div className="flex items-center justify-between text-slate-300 font-bold uppercase">
                                <span className="flex items-center gap-2">
                                    <MessageSquare size={14} className="text-cyan-400" />
                                    OPERATOR NOTES
                                </span>
                                <button 
                                    onClick={handleSaveNotes}
                                    className="text-cyan-400 hover:underline text-[11px] flex items-center gap-1 cursor-pointer"
                                >
                                    <Save size={12} /> Save Note
                                </button>
                            </div>
                            <textarea 
                                value={notesInput}
                                onChange={(e) => setNotesInput(e.target.value)}
                                rows={3}
                                placeholder="Enter human operator notes..."
                                className="w-full bg-black/50 border border-white/10 rounded-xl p-3 text-white placeholder-slate-500 focus:outline-none focus:border-cyan-400/50"
                            />
                        </div>

                        {/* Action Buttons Bar */}
                        <div className="grid grid-cols-3 gap-4 pt-2 text-xs">
                            <button
                                onClick={handleConfirm}
                                className="py-3 px-4 rounded-xl bg-emerald-500 hover:bg-emerald-400 text-black font-bold uppercase transition-all flex items-center justify-center gap-2 cursor-pointer shadow-lg shadow-emerald-500/20"
                            >
                                <CheckCircle2 size={16} /> CONFIRM
                            </button>

                            <button
                                onClick={handleDismiss}
                                className="py-3 px-4 rounded-xl bg-white/10 hover:bg-white/20 border border-white/20 text-slate-200 font-bold uppercase transition-all flex items-center justify-center gap-2 cursor-pointer"
                            >
                                <XCircle size={16} /> DISMISS
                            </button>

                            <button
                                onClick={handleFlag}
                                className="py-3 px-4 rounded-xl bg-amber-500/20 hover:bg-amber-500/30 border border-amber-400/50 text-amber-300 font-bold uppercase transition-all flex items-center justify-center gap-2 cursor-pointer"
                            >
                                <Flag size={16} /> FLAG FOR REVIEW
                            </button>
                        </div>
                    </div>

                    {/* Right 1 Column: Queue List Sidebar */}
                    <div className="lg:col-span-1 glass-card p-6 space-y-4 text-xs">
                        <div className="flex items-center justify-between border-b border-white/10 pb-3 font-bold text-white uppercase">
                            <span>REVIEW QUEUE ({reviewQueue.length})</span>
                            <span className="text-slate-400 text-[11px]">{currentIndex + 1} of {reviewQueue.length}</span>
                        </div>

                        <div className="space-y-2 max-h-[500px] overflow-y-auto pr-1">
                            {reviewQueue.map((item, idx) => {
                                const isSelected = idx === currentIndex;
                                return (
                                    <div
                                        key={item.frame_id}
                                        onClick={() => setCurrentIndex(idx)}
                                        className={`p-3.5 rounded-xl border transition-all cursor-pointer ${
                                            isSelected 
                                                ? 'bg-cyan-500/20 border-cyan-400 text-white shadow-[0_0_12px_rgba(0,240,255,0.2)]' 
                                                : 'bg-black/40 border-white/5 text-slate-300 hover:bg-white/5'
                                        }`}
                                    >
                                        <div className="flex justify-between items-center font-bold">
                                            <span>{item.frame_id}</span>
                                            <span className="text-cyan-300">{item.confidence}%</span>
                                        </div>

                                        <div className="flex justify-between items-center mt-1 text-[11px]">
                                            <span className="text-white font-semibold">{item.class}</span>
                                            <span className={`px-2 py-0.5 rounded text-[10px] font-bold ${
                                                item.risk === 'CRITICAL' ? 'text-red-400 bg-red-500/10' :
                                                item.risk === 'HIGH' ? 'text-red-300 bg-red-500/10' :
                                                'text-amber-300 bg-amber-500/10'
                                            }`}>
                                                {item.risk}
                                            </span>
                                        </div>
                                    </div>
                                );
                            })}
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
}
