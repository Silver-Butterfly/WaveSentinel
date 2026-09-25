import React, { useEffect, useRef, useState } from 'react';
import {
    Upload,
    FileUp,
    FileText,
    Save,
    Crosshair
} from 'lucide-react';
import { useSimulation } from '../../simulation/SimulationContext.jsx';
import { inferImage } from '../../api/client.js';
import defaultSonarImg from '../../assets/sim2d/ocean.jpg';

export default function UploadView() {
    const { addDetection, missionState } = useSimulation();

    const [file, setFile] = useState(null);
    const [previewUrl, setPreviewUrl] = useState(null);
    const [processingStep, setProcessingStep] = useState(0);
    const [results, setResults] = useState(null);
    const [archiveSaved, setArchiveSaved] = useState(false);
    const [fileError, setFileError] = useState('');
    const fileInputRef = useRef(null);

    const [metadata, setMetadata] = useState({
        frameId: 'WS_UPLOAD',
        timestamp: new Date().toISOString(),
        range: '100 m',
        depth: 'not provided',
        surveyLine: 'L-04'
    });

    useEffect(() => {
        return () => {
            if (previewUrl?.startsWith('blob:')) URL.revokeObjectURL(previewUrl);
        };
    }, [previewUrl]);

    const handleFileSelect = (selectedFile) => {
        if (!selectedFile) return;

        if (previewUrl?.startsWith('blob:')) {
            URL.revokeObjectURL(previewUrl);
        }

        setFile(selectedFile);
        setProcessingStep(0);
        setResults(null);
        setArchiveSaved(false);
        setFileError('');

        if (selectedFile.type.startsWith('image/')) {
            setPreviewUrl(URL.createObjectURL(selectedFile));
        } else {
            setPreviewUrl(null);
            setFileError('Select a readable SSS image file.');
        }
    };

    const handleDrop = (event) => {
        event.preventDefault();
        const selectedFile = event.dataTransfer.files?.[0];
        if (selectedFile) handleFileSelect(selectedFile);
    };

    const runInference = async () => {
        if (!file) return;
        setProcessingStep(1);
        setFileError('');

        try {
            const response = await inferImage(file, metadata);
            const detections = response.detections.map((item) => ({
                ...item,
                name: item.class,
                confidencePercent: `${(item.confidence * 100).toFixed(1)}%`,
                color: item.class === 'Ghost Net'
                    ? 'text-purple-400 border-purple-400 bg-purple-500/10'
                    : item.class === 'Pipe'
                        ? 'text-amber-400 border-amber-400 bg-amber-500/10'
                        : 'text-cyan-300 border-cyan-400 bg-cyan-500/10'
            }));

            setResults({
                ...response,
                detections,
                objectsCount: response.objects_count
            });
            setProcessingStep(2);
        } catch (error) {
            setFileError(error.message);
            setProcessingStep(3);
        }
    };

    const handleSaveToArchive = async () => {
        if (!results?.detections?.length || archiveSaved) return;

        try {
            for (const item of results.detections) {
                await addDetection({
                    mission_id: missionState.missionId,
                    frame_id: `${metadata.frameId}:${item.detection_id}`,
                    timestamp: metadata.timestamp,
                    class: item.class,
                    confidence: item.confidence,
                    bbox_xyxy: item.bbox_xyxy,
                    depth: metadata.depth,
                    latitude: null,
                    longitude: null,
                    position_source: 'not_available',
                    telemetry_source: 'not_provided',
                    motion_correction_state: 'not_applied',
                    input_quality: 'not_evaluated',
                    operator_notes: 'Uploaded SSS image. Risk and target georeference are not assessed from this image alone.',
                    source: 'uploaded_file'
                });
            }
            setArchiveSaved(true);
        } catch (error) {
            setFileError(`Archive save failed: ${error.message}`);
        }
    };

    const resetUpload = () => {
        if (previewUrl?.startsWith('blob:')) URL.revokeObjectURL(previewUrl);
        setFile(null);
        setPreviewUrl(null);
        setProcessingStep(0);
        setResults(null);
        setArchiveSaved(false);
        setFileError('');
    };

    return (
        <div className="space-y-8 animate-fade-in pb-12 font-mono">
            <div className="glass-card p-6 flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
                <div className="space-y-1">
                    <div className="flex items-center gap-3">
                        <Upload className="text-cyan-400" size={24} />
                        <h2 className="text-2xl font-bold text-white tracking-wide uppercase">
                            UPLOAD SCAN
                        </h2>
                        <span className="px-3 py-1 rounded-full bg-cyan-500/20 border border-cyan-400/50 text-cyan-300 text-xs font-bold uppercase">
                            BACKEND INFERENCE
                        </span>
                    </div>
                    <p className="text-slate-300 text-xs">
                        Upload an SSS image for TensorRT FP32 inference through the WaveSentinel backend.
                    </p>
                </div>

                {file && (
                    <button
                        onClick={resetUpload}
                        className="px-4 py-2 rounded-xl bg-white/10 hover:bg-white/20 border border-white/20 text-slate-200 text-xs font-bold uppercase transition-all cursor-pointer"
                    >
                        UPLOAD NEW SCAN
                    </button>
                )}
            </div>

            {!file && (
                <div className="glass-card p-10 flex flex-col items-center justify-center min-h-[380px] space-y-6 text-center border-2 border-dashed border-cyan-500/30 hover:border-cyan-400/60 transition-all">
                    <input
                        type="file"
                        ref={fileInputRef}
                        onChange={(e) => e.target.files && handleFileSelect(e.target.files[0])}
                        accept="image/*"
                        className="hidden"
                    />

                    <div
                        onDragOver={(e) => e.preventDefault()}
                        onDrop={handleDrop}
                        onClick={() => fileInputRef.current?.click()}
                        className="w-full flex flex-col items-center justify-center space-y-4 cursor-pointer p-8"
                    >
                        <div className="w-20 h-20 rounded-full bg-cyan-500/10 border border-cyan-500/40 flex items-center justify-center text-cyan-400 shadow-[0_0_25px_rgba(0,240,255,0.2)]">
                            <FileUp size={40} />
                        </div>
                        <div className="space-y-1">
                            <h3 className="font-bold text-white text-lg tracking-wide uppercase">
                                STAGE 1: UPLOAD SIDE-SCAN SONAR
                            </h3>
                            <p className="text-slate-400 text-xs">
                                PNG / JPG / TIFF / BMP. Prepared 640×640 or raw SSS frames are supported.
                            </p>
                        </div>
                        <button
                            type="button"
                            className="px-6 py-3 rounded-xl bg-cyan-500 hover:bg-cyan-400 text-black font-bold text-xs uppercase tracking-wider transition-all shadow-lg shadow-cyan-500/20"
                        >
                            Browse Files
                        </button>
                    </div>
                </div>
            )}

            {file && (
                <div className="space-y-8">
                    <div className="glass-card p-6 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-6">
                        <div className="flex items-center gap-4 min-w-0">
                            <div className="p-3 rounded-xl bg-cyan-500/10 border border-cyan-500/30 text-cyan-400">
                                <FileText size={24} />
                            </div>
                            <div className="space-y-1 min-w-0">
                                <div className="text-xs text-slate-400 uppercase">FILE NAME</div>
                                <div className="text-white font-bold text-base truncate">{file.name}</div>
                                <div className="text-xs text-slate-400 flex items-center gap-4">
                                    <span>{(file.size / (1024 * 1024)).toFixed(2)} MB</span>
                                    <span>SSS IMAGE</span>
                                </div>
                            </div>
                        </div>

                        {processingStep === 0 && (
                            <button
                                onClick={runInference}
                                className="px-6 py-3 rounded-xl bg-cyan-500 hover:bg-cyan-400 text-black font-bold text-xs uppercase tracking-wider transition-all shadow-lg shadow-cyan-500/25 flex items-center gap-2 cursor-pointer"
                            >
                                <Crosshair size={16} /> RUN INFERENCE
                            </button>
                        )}
                    </div>

                    <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                        <div className="glass-card p-6 space-y-4">
                            <h3 className="font-bold text-white uppercase text-sm border-b border-white/10 pb-3">
                                SCAN METADATA
                            </h3>

                            <div className="grid grid-cols-2 gap-4 text-xs">
                                <div className="space-y-1">
                                    <label className="text-slate-400 text-[10px]">Frame ID</label>
                                    <input
                                        type="text"
                                        value={metadata.frameId}
                                        onChange={(e) => setMetadata({ ...metadata, frameId: e.target.value })}
                                        className="w-full bg-black/40 border border-white/10 rounded-xl p-2.5 text-white"
                                    />
                                </div>

                                <div className="space-y-1">
                                    <label className="text-slate-400 text-[10px]">Timestamp</label>
                                    <input
                                        type="text"
                                        value={metadata.timestamp}
                                        onChange={(e) => setMetadata({ ...metadata, timestamp: e.target.value })}
                                        className="w-full bg-black/40 border border-white/10 rounded-xl p-2.5 text-white"
                                    />
                                </div>

                                <div className="space-y-1">
                                    <label className="text-slate-400 text-[10px]">Range</label>
                                    <input
                                        type="text"
                                        value={metadata.range}
                                        onChange={(e) => setMetadata({ ...metadata, range: e.target.value })}
                                        className="w-full bg-black/40 border border-white/10 rounded-xl p-2.5 text-white"
                                    />
                                </div>

                                <div className="space-y-1">
                                    <label className="text-slate-400 text-[10px]">Depth</label>
                                    <input
                                        type="text"
                                        value={metadata.depth}
                                        onChange={(e) => setMetadata({ ...metadata, depth: e.target.value })}
                                        className="w-full bg-black/40 border border-white/10 rounded-xl p-2.5 text-white"
                                    />
                                </div>

                                <div className="col-span-2 space-y-1">
                                    <label className="text-slate-400 text-[10px]">Survey Line</label>
                                    <input
                                        type="text"
                                        value={metadata.surveyLine}
                                        onChange={(e) => setMetadata({ ...metadata, surveyLine: e.target.value })}
                                        className="w-full bg-black/40 border border-white/10 rounded-xl p-2.5 text-white"
                                    />
                                </div>
                            </div>
                        </div>

                        <div className="glass-card p-6 space-y-4">
                            <h3 className="font-bold text-white uppercase text-sm border-b border-white/10 pb-3">
                                DATA SOURCE
                            </h3>
                            <div className="p-3.5 rounded-xl bg-black/40 border border-cyan-500/40 text-cyan-300 font-bold">
                                UPLOADED SSS FILE
                            </div>
                            <p className="text-slate-400 text-[11px]">
                                This is file-based inference. It is not a physical sonar or real-time telemetry connection.
                            </p>
                        </div>
                    </div>

                    <div className="glass-card p-6 space-y-4 text-xs">
                        <h3 className="font-bold text-white uppercase text-sm border-b border-white/10 pb-3">
                            INPUT GATE
                        </h3>
                        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-center font-bold">
                            <div className="p-3 rounded-xl bg-black/40 border border-emerald-500/30">
                                <div className="text-slate-400 text-[9px] uppercase">READABILITY</div>
                                <div className="text-emerald-400 mt-1">{fileError ? 'CHECK' : 'READY'}</div>
                            </div>
                            <div className="p-3 rounded-xl bg-black/40 border border-cyan-500/30">
                                <div className="text-slate-400 text-[9px] uppercase">SOURCE</div>
                                <div className="text-cyan-300 mt-1">UPLOAD</div>
                            </div>
                            <div className="p-3 rounded-xl bg-black/40 border border-cyan-500/30">
                                <div className="text-slate-400 text-[9px] uppercase">TELEMETRY</div>
                                <div className="text-cyan-300 mt-1">NOT PROVIDED</div>
                            </div>
                            <div className="p-3 rounded-xl bg-black/40 border border-cyan-500/30">
                                <div className="text-slate-400 text-[9px] uppercase">POSITION</div>
                                <div className="text-cyan-300 mt-1">NOT PROVIDED</div>
                            </div>
                        </div>
                    </div>

                    {processingStep === 1 && (
                        <div className="glass-card p-8 text-center space-y-4">
                            <div className="w-12 h-12 rounded-full border-2 border-cyan-400 border-t-transparent animate-spin mx-auto"></div>
                            <div className="text-cyan-300 font-bold uppercase text-sm">RUNNING TENSORRT INFERENCE...</div>
                            <p className="text-slate-400 text-xs">
                                Preparing the SSS image, executing the frozen FP32 engine, and applying frozen class thresholds.
                            </p>
                        </div>
                    )}

                    {fileError && processingStep === 3 && (
                        <div className="glass-card p-6 border border-red-500/30 bg-red-500/5">
                            <div className="text-red-300 font-bold uppercase text-sm">BACKEND INFERENCE ERROR</div>
                            <p className="text-slate-300 text-xs mt-2">{fileError}</p>
                        </div>
                    )}

                    {processingStep === 2 && results && (
                        <div className="glass-card p-8 space-y-6 animate-fade-in">
                            <div className="flex items-center justify-between border-b border-white/10 pb-4">
                                <div>
                                    <h3 className="font-bold text-white uppercase text-base">
                                        INFERENCE RESULTS
                                    </h3>
                                    <div className="text-[11px] text-slate-400 mt-1">
                                        TensorRT FP32 • {results.input.mode}
                                    </div>
                                </div>
                                <span className="px-3 py-1 rounded-full bg-emerald-500/20 text-emerald-400 text-xs font-bold">
                                    {results.objectsCount} Objects Detected
                                </span>
                            </div>

                            <div className="text-[11px] text-slate-400 flex flex-wrap gap-4">
                                <span>INPUT: <strong className="text-cyan-300">{results.input.width}×{results.input.height}</strong></span>
                                <span>INFERENCE: <strong className="text-cyan-300">{results.runtime.total_backend_ms.toFixed(2)} ms</strong></span>
                                <span>RISK: <strong className="text-amber-300">UNASSESSED</strong></span>
                                <span>GEOREFERENCE: <strong className="text-amber-300">NOT AVAILABLE</strong></span>
                            </div>

                            <div className="relative h-80 rounded-2xl overflow-hidden bg-black border border-white/15">
                                <img
                                    src={previewUrl || defaultSonarImg}
                                    alt="Detection Result"
                                    className="w-full h-full object-contain filter brightness-90 contrast-125"
                                />

                                {(results.detections || []).map((item, index) => {
                                    const [x1, y1, x2, y2] = item.bbox_xyxy || [0, 0, 0, 0];
                                    const w = results.input.width || 640;
                                    const h = results.input.height || 640;

                                    return (
                                        <div
                                            key={item.detection_id || index}
                                            className="absolute border-2 border-cyan-400 bg-cyan-400/10 rounded pointer-events-none"
                                            style={{
                                                left: `${(x1 / w) * 100}%`,
                                                top: `${(y1 / h) * 100}%`,
                                                width: `${Math.max(0, ((x2 - x1) / w) * 100)}%`,
                                                height: `${Math.max(0, ((y2 - y1) / h) * 100)}%`
                                            }}
                                        >
                                            <span className="bg-cyan-500 text-black font-extrabold text-[10px] px-2 py-0.5 rounded self-start tracking-wider uppercase">
                                                {item.class} {(item.confidence * 100).toFixed(1)}%
                                            </span>
                                        </div>
                                    );
                                })}
                            </div>

                            <div className="space-y-3 text-xs">
                                <div className="text-slate-400 font-bold uppercase text-[11px]">EXTRACTED CANDIDATE OBJECTS</div>

                                {results.detections.length === 0 ? (
                                    <div className="p-4 rounded-xl bg-black/40 border border-white/5 text-slate-400">
                                        No detections passed the frozen class-wise thresholds.
                                    </div>
                                ) : (
                                    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
                                        {results.detections.map((item, i) => (
                                            <div key={i} className={`p-4 rounded-xl border flex justify-between items-center ${item.color}`}>
                                                <div>
                                                    <div className="text-white font-bold text-sm">{item.name}</div>
                                                    <div className="text-slate-300 text-[11px]">Confidence: {item.confidencePercent}</div>
                                                </div>
                                                <span className="px-2.5 py-1 rounded text-[10px] font-bold bg-black/60 border border-white/10">
                                                    {item.risk}
                                                </span>
                                            </div>
                                        ))}
                                    </div>
                                )}
                            </div>

                            <div className="flex flex-col sm:flex-row items-center justify-end gap-4 pt-4 border-t border-white/10 text-xs">
                                <button
                                    onClick={handleSaveToArchive}
                                    disabled={archiveSaved || !results.detections.length}
                                    className="w-full sm:w-auto px-5 py-2.5 rounded-xl bg-cyan-500/20 hover:bg-cyan-500/30 border border-cyan-400/40 text-cyan-300 font-bold uppercase transition-all flex items-center justify-center gap-2 cursor-pointer disabled:opacity-40 disabled:cursor-not-allowed"
                                >
                                    <Save size={14} /> {archiveSaved ? 'Saved to Archive' : 'Save to Archive'}
                                </button>
                            </div>
                        </div>
                    )}
                </div>
            )}
        </div>
    );
}
