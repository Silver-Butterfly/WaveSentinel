import React, { useState } from 'react';
import { MapPin, Navigation, Eye, Compass, Layers, ShieldCheck, Activity } from 'lucide-react';
import { useSimulation } from '../../simulation/SimulationContext.jsx';

export default function SurveyMapView({ onNavigateTab }) {
    const { detections, telemetry, missionState } = useSimulation();

    const [selectedMarker, setSelectedMarker] = useState(null);

    const [layers, setLayers] = useState({
        shipwreck: true,
        pipe: true,
        mine: true,
        ghostnet: true,
        vesselTrack: true,
        sonarCoverage: true,
        detectionZones: true,
        depth: true,
        highRisk: true
    });

    const geoDetections = detections.filter(item =>
        item.latitude && item.longitude && item.position_source && item.position_source !== 'not_available'
    );

    const parseCoord = (value) => {
        const match = String(value).match(/-?\d+(?:\.\d+)?/);
        return match ? Number(match[0]) : NaN;
    };

    const coordValues = geoDetections.map(item => ({
        item,
        lat: parseCoord(item.latitude),
        lon: parseCoord(item.longitude)
    })).filter(x => Number.isFinite(x.lat) && Number.isFinite(x.lon));

    const lats = coordValues.map(x => x.lat);
    const lons = coordValues.map(x => x.lon);
    const minLat = Math.min(...lats, 20.43);
    const maxLat = Math.max(...lats, 20.453);
    const minLon = Math.min(...lons, 85.10);
    const maxLon = Math.max(...lons, 85.13);

    const markers = coordValues.map(({ item, lat, lon }, index) => {
        const x = ((lon - minLon) / Math.max(0.000001, maxLon - minLon)) * 80 + 10;
        const y = 90 - ((lat - minLat) / Math.max(0.000001, maxLat - minLat)) * 80;
        const objectKey = String(item.class || '').toLowerCase().replaceAll(' ', '');
        const color = objectKey === 'mine' ? 'bg-red-600 border-red-500 font-extrabold' :
            objectKey === 'shipwreck' ? 'bg-red-500 border-red-400' :
            objectKey === 'ghostnet' ? 'bg-purple-500 border-purple-400' :
            'bg-amber-400 border-amber-300';
        return {
            id: item.frame_id || `DET-${index}`,
            object: item.class,
            confidence: `${Number(item.confidence).toFixed(1)}%`,
            risk: item.risk || 'UNASSESSED',
            coords: `${item.latitude}, ${item.longitude}`,
            depth: item.depth || 'NOT PROVIDED',
            frame: item.frame_id,
            x, y, color,
            source: item.position_source,
            item
        };
    });

    const toggleLayer = (key) => {
        setLayers(prev => ({ ...prev, [key]: !prev[key] }));
    };

    React.useEffect(() => {
        if (!selectedMarker && markers.length) setSelectedMarker(markers[0]);
        if (selectedMarker && !markers.some(m => m.id === selectedMarker.id)) setSelectedMarker(markers[0] || null);
    }, [markers.length, selectedMarker]);

    return (
        <div className="space-y-8 animate-fade-in pb-12 font-mono">
            {/* Header */}
            <div className="glass-card p-6 flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
                <div className="space-y-1">
                    <div className="flex items-center gap-3">
                        <MapPin className="text-cyan-400" size={24} />
                        <h2 className="text-2xl font-bold text-white tracking-wide uppercase">
                            SURVEY MAP
                        </h2>
                        <span className="px-3 py-1 rounded-full bg-cyan-500/20 border border-cyan-400/50 text-cyan-300 text-xs font-bold uppercase">
                            SPATIAL GEOLOCATION INTELLIGENCE
                        </span>
                    </div>
                    <p className="text-slate-300 text-xs">
                        Subsea Survey Track & Simulated Acoustic Coverage Map
                    </p>
                </div>

                <div className="flex flex-wrap items-center gap-3 text-xs">
                    <div className="flex items-center gap-2 bg-black/50 px-3.5 py-2 rounded-xl border border-cyan-500/40 text-cyan-300 font-bold">
                        <span>POSITION SOURCE:</span>
                        <span className="text-cyan-400 font-extrabold">SIMULATED</span>
                    </div>

                    <div className="flex items-center gap-2 bg-black/50 px-3.5 py-2 rounded-xl border border-emerald-500/40 text-emerald-400 font-bold">
                        <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span>
                        SURVEY TRACK: ACTIVE
                    </div>
                </div>
            </div>

            {/* Map Canvas + Filters & Inspector Grid */}
            <div className="grid grid-cols-1 lg:grid-cols-4 gap-8">
                {/* Main Map Canvas (3 Columns) */}
                <div className="lg:col-span-3 glass-card p-4 relative flex flex-col justify-between min-h-[520px] bg-[#020c1b] rounded-2xl overflow-hidden border border-cyan-500/20">
                    
                    {/* Layer Toggles Overlay Header */}
                    <div className="z-10 bg-black/70 backdrop-blur-md p-3.5 rounded-xl border border-white/10 flex flex-wrap items-center gap-4 text-xs text-slate-300">
                        <div className="font-bold text-white uppercase text-[11px] flex items-center gap-1.5">
                            <Layers size={14} className="text-cyan-400" /> MAP LAYERS:
                        </div>
                        
                        <label className="flex items-center gap-1.5 cursor-pointer">
                            <input type="checkbox" checked={layers.shipwreck} onChange={() => toggleLayer('shipwreck')} className="accent-cyan-400" />
                            <span>Shipwreck</span>
                        </label>
                        <label className="flex items-center gap-1.5 cursor-pointer">
                            <input type="checkbox" checked={layers.pipe} onChange={() => toggleLayer('pipe')} className="accent-cyan-400" />
                            <span>Pipe</span>
                        </label>
                        <label className="flex items-center gap-1.5 cursor-pointer">
                            <input type="checkbox" checked={layers.mine} onChange={() => toggleLayer('mine')} className="accent-cyan-400" />
                            <span>Mine</span>
                        </label>
                        <label className="flex items-center gap-1.5 cursor-pointer">
                            <input type="checkbox" checked={layers.ghostnet} onChange={() => toggleLayer('ghostnet')} className="accent-cyan-400" />
                            <span>Ghost Net</span>
                        </label>
                        <label className="flex items-center gap-1.5 cursor-pointer">
                            <input type="checkbox" checked={layers.vesselTrack} onChange={() => toggleLayer('vesselTrack')} className="accent-cyan-400" />
                            <span>Vessel Track</span>
                        </label>
                        <label className="flex items-center gap-1.5 cursor-pointer">
                            <input type="checkbox" checked={layers.sonarCoverage} onChange={() => toggleLayer('sonarCoverage')} className="accent-cyan-400" />
                            <span>Sonar Coverage</span>
                        </label>
                        <label className="flex items-center gap-1.5 cursor-pointer">
                            <input type="checkbox" checked={layers.depth} onChange={() => toggleLayer('depth')} className="accent-cyan-400" />
                            <span>Depth</span>
                        </label>
                        <label className="flex items-center gap-1.5 cursor-pointer">
                            <input type="checkbox" checked={layers.highRisk} onChange={() => toggleLayer('highRisk')} className="accent-cyan-400" />
                            <span>High Risk</span>
                        </label>
                    </div>

                    {/* Simulated Bathymetric Grid Canvas */}
                    <div className="relative w-full h-[420px] my-3 rounded-xl overflow-hidden bg-[radial-gradient(#00f0ff_1px,transparent_1px)] [background-size:24px_24px] border border-white/10 flex items-center justify-center">
                        
                        {/* Sonar Coverage Corridor Polygon */}
                        {layers.sonarCoverage && (
                            <svg className="absolute inset-0 w-full h-full pointer-events-none">
                                <polygon 
                                    points="80,350 180,310 300,210 460,270 580,170 620,110 580,170 460,210 300,150 180,250 80,290" 
                                    fill="rgba(0, 240, 255, 0.08)"
                                    stroke="rgba(0, 240, 255, 0.25)"
                                    strokeWidth="1"
                                />
                            </svg>
                        )}

                        {/* Survey Track Line SVG */}
                        {layers.vesselTrack && (
                            <svg className="absolute inset-0 w-full h-full pointer-events-none">
                                <polyline
                                    points="100,320 200,280 320,180 480,240 600,140"
                                    fill="none"
                                    stroke="rgba(0, 240, 255, 0.6)"
                                    strokeWidth="2.5"
                                    strokeDasharray="6, 6"
                                />
                                <circle cx="100" cy="320" r="4" fill="#00F0FF" />
                                <circle cx="200" cy="280" r="4" fill="#00F0FF" />
                                <circle cx="320" cy="180" r="4" fill="#00F0FF" />
                                <circle cx="480" cy="240" r="4" fill="#00F0FF" />
                                <circle cx="600" cy="140" r="4" fill="#00F0FF" />
                            </svg>
                        )}

                        {/* Detection Markers */}
                        {markers.map((m) => {
                            const isSelected = selectedMarker?.id === m.id;
                            const classKey = String(m.object).toLowerCase().replaceAll(' ', '');
                            if (layers[classKey] === false) return null;

                            return (
                                <button
                                    key={m.id}
                                    onClick={() => setSelectedMarker(m)}
                                    style={{ left: `${m.x}%`, top: `${m.y}%` }}
                                    className="absolute -translate-x-1/2 -translate-y-1/2 group cursor-pointer transition-all transform hover:scale-125 z-20"
                                >
                                    <div className={`w-7 h-7 rounded-full ${m.color} flex items-center justify-center text-black font-extrabold text-[10px] shadow-lg ${isSelected ? 'ring-4 ring-cyan-400 scale-125 animate-bounce' : ''}`}>
                                        ▲
                                    </div>
                                    <span className="absolute left-1/2 -translate-x-1/2 -bottom-5 bg-black/80 backdrop-blur-md px-1.5 py-0.5 rounded text-[9px] text-cyan-300 whitespace-nowrap border border-white/10 opacity-80 group-hover:opacity-100 font-mono">
                                        {m.object} ({m.confidence})
                                    </span>
                                </button>
                            );
                        })}

                        <div className="absolute bottom-3 left-3 bg-black/80 backdrop-blur-md px-3 py-1.5 rounded-lg border border-white/10 text-xs text-slate-300 flex items-center gap-2">
                            <span>SURVEY TRACK: BAY SECTOR 04 (L-04)</span>
                            <span>|</span>
                            <span className="text-cyan-300 font-bold">SIMULATED POSITION</span>
                        </div>
                    </div>
                </div>

                {/* Right Side Panel: Detection Inspector */}
                <div className="lg:col-span-1 glass-card p-6 space-y-6 text-xs">
                    <div className="flex items-center justify-between border-b border-white/10 pb-3">
                        <h3 className="font-bold text-white uppercase text-sm">
                            DETECTION #{selectedMarker?.id || '0042'}
                        </h3>
                        <span className="px-2 py-0.5 rounded bg-red-500/20 text-red-300 text-xs font-bold">
                            {selectedMarker?.risk || 'UNASSESSED'}
                        </span>
                    </div>

                    <div className="space-y-3.5">
                        <div className="space-y-1">
                            <div className="text-slate-400 text-[10px]">OBJECT CLASS</div>
                            <div className="text-xl font-bold text-white">{selectedMarker?.object || 'Shipwreck'}</div>
                        </div>

                        <div className="flex justify-between items-center bg-black/40 p-2.5 rounded-lg border border-white/5">
                            <span className="text-slate-400">POSITION SOURCE</span>
                            <span className="text-cyan-400 font-bold">SIMULATED</span>
                        </div>

                        <div className="flex justify-between items-center bg-black/40 p-2.5 rounded-lg border border-white/5">
                            <span className="text-slate-400">MOTION STATE</span>
                            <span className="text-emerald-400 font-bold">{selectedMarker?.item?.motion_correction_state || 'metadata_only'}</span>
                        </div>

                        <div className="flex justify-between items-center bg-black/40 p-2.5 rounded-lg border border-white/5">
                            <span className="text-slate-400">FRAME</span>
                            <span className="text-cyan-300 font-bold">{selectedMarker?.frame || 'WS_001284'}</span>
                        </div>

                        <div className="flex justify-between items-center bg-black/40 p-2.5 rounded-lg border border-white/5">
                            <span className="text-slate-400">CONFIDENCE</span>
                            <span className="text-cyan-300 font-bold">{selectedMarker?.confidence || '87%'}</span>
                        </div>

                        <div className="flex justify-between items-center bg-black/40 p-2.5 rounded-lg border border-white/5">
                            <span className="text-slate-400">DEPTH</span>
                            <span className="text-white font-bold">{selectedMarker?.depth || '42.3 m'}</span>
                        </div>

                        <div className="space-y-1 bg-black/40 p-3 rounded-lg border border-white/5">
                            <div className="text-slate-400 text-[10px]">SCHEMATIC COORDINATES</div>
                            <div className="text-white font-bold text-xs">{selectedMarker?.coords || 'NO GEOREFERENCED DETECTIONS'}</div>
                        </div>
                    </div>

                    <button 
                        onClick={() => onNavigateTab('archive')}
                        className="w-full py-3 rounded-xl bg-cyan-500 hover:bg-cyan-400 text-black font-bold text-xs uppercase tracking-wider transition-all shadow-lg shadow-cyan-500/20 flex items-center justify-center gap-2 cursor-pointer"
                    >
                        <Eye size={14} /> View Frame in Archive
                    </button>
                </div>
            </div>
        </div>
    );
}
