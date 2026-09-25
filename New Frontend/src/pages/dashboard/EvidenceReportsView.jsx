import React from 'react';
import { 
    FileText, 
    Download, 
    FileCode, 
    FileSpreadsheet, 
    CheckCircle2, 
    ShieldCheck, 
    Clock, 
    Database,
    ExternalLink
} from 'lucide-react';
import { useSimulation } from '../../simulation/SimulationContext.jsx';

export default function EvidenceReportsView() {
    const { missionState, detections } = useSimulation();

    const reviewedCount = detections.filter(d => d.review_status === 'confirmed').length;
    const pendingCount = detections.filter(d => d.review_status === 'pending' || d.review_status === 'flagged').length;

    // Standardized JSON export handler
    const handleExportJson = () => {
        const jsonContent = JSON.stringify({
            mission_id: missionState.missionId,
            survey_area: missionState.surveyArea,
            mode: 'SIMULATION',
            generated_at: new Date().toISOString(),
            metrics: {
                total_frames: missionState.framesProcessed,
                total_detections: missionState.detectionsCount,
                high_risk_count: missionState.highRiskCount,
                reviewed_count: reviewedCount,
                pending_count: pendingCount
            },
            detections: detections.map(d => ({
                mission_id: d.mission_id,
                frame_id: d.frame_id,
                timestamp: d.timestamp,
                class: d.class,
                confidence: d.confidence,
                risk: d.risk,
                risk_source: d.risk_source || 'not_available',
                depth: d.depth,
                latitude: d.latitude,
                longitude: d.longitude,
                position_source: d.position_source,
                telemetry_source: d.telemetry_source,
                heave: d.heave,
                pitch: d.pitch,
                roll: d.roll,
                motion_state: 'available',
                correction_state: d.motion_correction_state,
                input_quality: d.input_quality,
                model_version: d.model_version,
                review_status: d.review_status,
                operator_notes: d.operator_notes
            }))
        }, null, 2);

        const blob = new Blob([jsonContent], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `WaveSentinel_${missionState.missionId}_Evidence.json`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    };

    // Standardized CSV export handler
    const handleExportCsv = () => {
        const headers = [
            'mission_id', 'frame_id', 'timestamp', 'class', 'confidence', 'risk', 
            'depth', 'latitude', 'longitude', 'position_source', 'telemetry_source', 
            'motion_state', 'correction_state', 'input_quality', 'model_version', 'review_status'
        ];

        const rows = detections.map(d => [
            d.mission_id, d.frame_id, d.timestamp, d.class, d.confidence, d.risk,
            d.depth, d.latitude, d.longitude, d.position_source, d.telemetry_source,
            'available', d.motion_correction_state, d.input_quality, d.model_version, d.review_status
        ]);

        const csvContent = [headers.join(','), ...rows.map(r => r.map(val => `"${val}"`).join(','))].join('\n');
        const blob = new Blob([csvContent], { type: 'text/csv' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `WaveSentinel_${missionState.missionId}_Detections.csv`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    };

    // Standardized Mission Text Report Export handler
    const handleExportReport = () => {
        const reportText = `================================================================
WAVESENTINEL SUBSEA SONAR INTELLIGENCE & DEBRIS REPORT
================================================================
MISSION ID:           ${missionState.missionId}
SURVEY AREA:          ${missionState.surveyArea}
OPERATIONAL MODE:     ${missionState.operationalMode} (SIMULATION DEMO)
GENERATED TIMESTAMP:  ${new Date().toLocaleString()}

SUMMARY METRICS:
- Processed Frames:   ${missionState.framesProcessed} / ${missionState.framesTotal}
- Total Detections:   ${missionState.detectionsCount}
- Simulated Priority Flags:  ${missionState.highRiskCount}
- Operator Reviewed:  ${reviewedCount}
- Pending Review:     ${pendingCount}

SIMULATION METADATA PROVENANCE:
- Position Source:    simulated
- Telemetry Source:   simulation
- Motion Correction:  metadata_only
- AI Model Version:   Phase-D Robust-V2-HN (YOLOv8s)

DETECTION SUMMARY RECORD SAMPLE:
${detections.map((d, i) => `${i + 1}. [${d.frame_id}] Class: ${d.class} | Confidence: ${d.confidence}% | Risk: ${d.risk} | Status: ${d.review_status.toUpperCase()}
   Location: ${d.latitude}, ${d.longitude} (${d.position_source}) | Notes: ${d.operator_notes}`).join('\n')}

================================================================
END OF REPORT - WAVESENTINEL COMMAND CENTRE
================================================================`;

        const blob = new Blob([reportText], { type: 'text/plain' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `WaveSentinel_${missionState.missionId}_Executive_Report.txt`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    };

    return (
        <div className="space-y-8 animate-fade-in pb-12 font-mono">
            {/* Header Banner */}
            <div className="glass-card p-6 flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
                <div className="space-y-1">
                    <div className="flex items-center gap-3">
                        <FileText className="text-cyan-400" size={24} />
                        <h2 className="text-2xl font-bold text-white tracking-wide uppercase">
                            EVIDENCE & REPORTS
                        </h2>
                        <span className="px-3 py-1 rounded-full bg-cyan-500/20 border border-cyan-400/40 text-cyan-300 text-xs font-bold uppercase">
                            MISSION OUTPUT PACKAGES
                        </span>
                    </div>
                    <p className="text-slate-300 text-xs">
                        Export operational sonar evidence packages, detection telemetry manifests, and compliance audit reports.
                    </p>
                </div>
            </div>

            {/* Mission Summary Card */}
            <div className="glass-card p-6 space-y-6">
                <div className="flex items-center justify-between border-b border-white/10 pb-4">
                    <h3 className="font-bold text-white uppercase text-base">
                        MISSION EVIDENCE SUMMARY
                    </h3>
                    <span className="text-cyan-300 font-bold text-sm">MISSION ID: {missionState.missionId}</span>
                </div>

                <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-4 text-xs">
                    <div className="p-4 rounded-xl bg-black/40 border border-white/5 space-y-1">
                        <div className="text-slate-400 text-[10px] uppercase">MISSION</div>
                        <div className="text-white font-bold text-base">{missionState.missionId}</div>
                    </div>

                    <div className="p-4 rounded-xl bg-black/40 border border-white/5 space-y-1">
                        <div className="text-slate-400 text-[10px] uppercase">FRAMES</div>
                        <div className="text-white font-bold text-base">{missionState.framesProcessed.toLocaleString()}</div>
                    </div>

                    <div className="p-4 rounded-xl bg-black/40 border border-white/5 space-y-1">
                        <div className="text-slate-400 text-[10px] uppercase">DETECTIONS</div>
                        <div className="text-cyan-300 font-bold text-base">{missionState.detectionsCount}</div>
                    </div>

                    <div className="p-4 rounded-xl bg-black/40 border border-white/5 space-y-1">
                        <div className="text-slate-400 text-[10px] uppercase">HIGH RISK</div>
                        <div className="text-red-400 font-bold text-base">{missionState.highRiskCount}</div>
                    </div>

                    <div className="p-4 rounded-xl bg-black/40 border border-white/5 space-y-1">
                        <div className="text-slate-400 text-[10px] uppercase">REVIEWED</div>
                        <div className="text-emerald-400 font-bold text-base">{reviewedCount}</div>
                    </div>

                    <div className="p-4 rounded-xl bg-black/40 border border-white/5 space-y-1">
                        <div className="text-slate-400 text-[10px] uppercase">PENDING REVIEW</div>
                        <div className="text-amber-400 font-bold text-base">{pendingCount}</div>
                    </div>
                </div>

                {/* Export Buttons */}
                <div className="flex flex-col sm:flex-row items-center justify-end gap-4 pt-2 border-t border-white/10 text-xs">
                    <button
                        onClick={handleExportJson}
                        className="w-full sm:w-auto px-5 py-3 rounded-xl bg-cyan-500/20 hover:bg-cyan-500/30 border border-cyan-400/50 text-cyan-300 font-bold uppercase transition-all flex items-center justify-center gap-2 cursor-pointer"
                    >
                        <FileCode size={16} /> EXPORT JSON
                    </button>

                    <button
                        onClick={handleExportCsv}
                        className="w-full sm:w-auto px-5 py-3 rounded-xl bg-emerald-500/20 hover:bg-emerald-500/30 border border-emerald-400/50 text-emerald-300 font-bold uppercase transition-all flex items-center justify-center gap-2 cursor-pointer"
                    >
                        <FileSpreadsheet size={16} /> EXPORT CSV
                    </button>

                    <button
                        onClick={handleExportReport}
                        className="w-full sm:w-auto px-6 py-3 rounded-xl bg-cyan-500 hover:bg-cyan-400 text-black font-bold uppercase transition-all flex items-center justify-center gap-2 cursor-pointer shadow-lg shadow-cyan-500/20"
                    >
                        <Download size={16} /> EXPORT MISSION REPORT
                    </button>
                </div>
            </div>

            {/* Detection Records Preview Table */}
            <div className="glass-card p-6 space-y-4 text-xs">
                <div className="flex items-center justify-between border-b border-white/10 pb-3">
                    <h3 className="font-bold text-white uppercase text-sm">
                        EXPORTABLE DETECTION MANIFEST
                    </h3>
                    <span className="text-slate-400">All records contain standardized simulation metadata fields</span>
                </div>

                <div className="overflow-x-auto">
                    <table className="w-full text-left border-collapse">
                        <thead>
                            <tr className="border-b border-white/10 text-slate-400 uppercase">
                                <th className="py-3 px-4">FRAME</th>
                                <th className="py-3 px-4">OBJECT CLASS</th>
                                <th className="py-3 px-4 text-right">CONFIDENCE</th>
                                <th className="py-3 px-4">RISK</th>
                                <th className="py-3 px-4">POS SOURCE</th>
                                <th className="py-3 px-4">MOTION CORRECTION</th>
                                <th className="py-3 px-4 text-right">REVIEW STATUS</th>
                            </tr>
                        </thead>
                        <tbody>
                            {detections.map((d, idx) => (
                                <tr key={idx} className="border-b border-white/5 hover:bg-white/5">
                                    <td className="py-3.5 px-4 font-bold text-cyan-300">{d.frame_id}</td>
                                    <td className="py-3.5 px-4 text-white font-semibold">{d.class}</td>
                                    <td className="py-3.5 px-4 text-right font-bold text-white">{d.confidence}%</td>
                                    <td className="py-3.5 px-4">
                                        <span className={`px-2 py-0.5 rounded text-[10px] font-bold ${
                                            d.risk === 'CRITICAL' ? 'bg-red-600/20 text-red-400 border border-red-500' :
                                            d.risk === 'HIGH' ? 'bg-red-500/20 text-red-300 border border-red-500/30' :
                                            'bg-amber-500/20 text-amber-300 border border-amber-500/30'
                                        }`}>
                                            {d.risk}
                                        </span>
                                    </td>
                                    <td className="py-3.5 px-4 text-cyan-300 font-bold uppercase">{d.position_source}</td>
                                    <td className="py-3.5 px-4 text-slate-300 uppercase">{d.motion_correction_state}</td>
                                    <td className="py-3.5 px-4 text-right">
                                        <span className={`px-2 py-0.5 rounded text-[10px] font-bold uppercase ${
                                            d.review_status === 'confirmed' ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/40' :
                                            d.review_status === 'dismissed' ? 'bg-slate-500/20 text-slate-400 border border-slate-500/30' :
                                            'bg-amber-500/20 text-amber-300 border border-amber-500/40'
                                        }`}>
                                            {d.review_status}
                                        </span>
                                    </td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
    );
}
