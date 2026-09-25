import React, { createContext, useContext, useEffect, useRef, useState } from 'react';
import {
    addDetection as apiAddDetection,
    endMission as apiEndMission,
    getHealth,
    getState,
    getSystemStatus,
    pauseMission as apiPauseMission,
    resetBackend,
    resumeMission as apiResumeMission,
    reviewDetection,
    startMission as apiStartMission
} from '../api/client.js';
import { INITIAL_DETECTIONS, INITIAL_MISSION_STATE, INITIAL_SETTINGS } from './simulationData.js';

const SimulationContext = createContext(null);

function formatTelemetry(raw, fallback = {}) {
    if (!raw) return fallback;
    return {
        ...fallback,
        heave: raw.heave ?? fallback.heave,
        pitch: raw.pitch ?? fallback.pitch,
        roll: raw.roll ?? fallback.roll,
        heading: raw.heading ?? fallback.heading,
        depth: raw.depth ?? fallback.depth,
        vesselSpeed: raw.vesselSpeed ?? fallback.vesselSpeed,
        latency: raw.latency ?? fallback.latency,
        timestamp: raw.timestamp ?? fallback.timestamp,
        position: raw.position ?? fallback.position,
        frameId: raw.frameId ?? fallback.frameId,
        source: raw.source ?? 'simulation',
        valid: raw.valid ?? true,
    };
}

const FALLBACK_TELEMETRY = {
    heave: '+0.18 m',
    pitch: '+2.4°',
    roll: '-1.1°',
    heading: '142°',
    depth: '42.3 m',
    vesselSpeed: '4.2 kts',
    latency: 'SIMULATION',
    timestamp: '11:37:42',
    position: '20.4521° N, 85.1294° E',
    frameId: 'SIM_001284',
    source: 'simulation',
    valid: true,
};

export function SimulationProvider({ children }) {
    const [missionState, setMissionState] = useState(INITIAL_MISSION_STATE);
    const [detections, setDetections] = useState(INITIAL_DETECTIONS);
    const [settings, setSettingsState] = useState(INITIAL_SETTINGS);
    const [telemetry, setTelemetry] = useState(FALLBACK_TELEMETRY);
    const [telemetryLog, setTelemetryLog] = useState([]);
    const [classCounts, setClassCounts] = useState({
        SHIPWRECK: 142,
        PIPE: 98,
        MINE: 64,
        GHOST_NET: 43,
    });

    const [backendOnline, setBackendOnline] = useState(false);
    const [backendHealth, setBackendHealth] = useState(null);
    const [runtimeBenchmark, setRuntimeBenchmark] = useState(null);
    const [backendError, setBackendError] = useState(null);
    const lastGoodState = useRef(null);

    const applyBackendState = (state) => {
        if (!state) return;
        if (state.mission) setMissionState(state.mission);
        if (Array.isArray(state.detections)) setDetections(state.detections);
        if (state.settings) setSettingsState(prev => ({ ...prev, ...state.settings }));
        if (state.telemetry) setTelemetry(prev => formatTelemetry(state.telemetry, prev));
        if (Array.isArray(state.telemetryLog)) setTelemetryLog(state.telemetryLog);
        if (state.classCounts) setClassCounts(state.classCounts);
        lastGoodState.current = state;
    };

    const refreshBackend = async ({ quiet = false } = {}) => {
        try {
            const [state, health, system] = await Promise.all([
                getState(),
                getHealth(),
                getSystemStatus()
            ]);
            applyBackendState(state);
            setBackendHealth(health);
            setRuntimeBenchmark(system.g3_benchmark || null);
            setBackendOnline(true);
            setBackendError(null);
            return state;
        } catch (error) {
            setBackendOnline(false);
            if (!quiet) setBackendError(error.message);
            return null;
        }
    };

    useEffect(() => {
        let cancelled = false;
        refreshBackend({ quiet: false }).then(() => {
            if (cancelled) return;
        });
        return () => { cancelled = true; };
    }, []);

    useEffect(() => {
        if (!backendOnline) return undefined;
        const interval = setInterval(async () => {
            try {
                const state = await getState();
                applyBackendState(state);
                setBackendOnline(true);
            } catch (error) {
                setBackendOnline(false);
                setBackendError(error.message);
            }
        }, 1500);
        return () => clearInterval(interval);
    }, [backendOnline]);

    useEffect(() => {
        if (!backendOnline) return undefined;
        const interval = setInterval(async () => {
            try {
                const system = await getSystemStatus();
                setRuntimeBenchmark(system.g3_benchmark || null);
            } catch {}
        }, 10000);
        return () => clearInterval(interval);
    }, [backendOnline]);

    useEffect(() => {
        if (backendOnline) return undefined;
        if (missionState.status !== 'ACTIVE') return undefined;

        const interval = setInterval(() => {
            setTelemetry(prev => {
                const now = new Date();
                const t = now.getTime() / 1000;
                const h = 0.18 + Math.sin(t * 0.7) * 0.04;
                const p = 2.4 + Math.cos(t * 0.55) * 0.3;
                const r = -1.1 + Math.sin(t * 0.35) * 0.3;
                return formatTelemetry({
                    ...prev,
                    heave: `${h >= 0 ? '+' : ''}${h.toFixed(2)} m`,
                    pitch: `${p >= 0 ? '+' : ''}${p.toFixed(1)}°`,
                    roll: `${r.toFixed(1)}°`,
                    timestamp: now.toTimeString().split(' ')[0],
                    source: 'simulation',
                    valid: true,
                }, prev);
            });
        }, 1500);
        return () => clearInterval(interval);
    }, [backendOnline, missionState.status]);

    const syncMission = (nextMission) => nextMission && setMissionState(nextMission);

    const startMission = async () => {
        try { syncMission((await apiStartMission()).mission); }
        catch { setMissionState(prev => ({ ...prev, status: 'ACTIVE' })); }
    };
    const pauseMission = async () => {
        try { syncMission((await apiPauseMission()).mission); }
        catch { setMissionState(prev => ({ ...prev, status: 'PAUSED' })); }
    };
    const resumeMission = async () => {
        try { syncMission((await apiResumeMission()).mission); }
        catch { setMissionState(prev => ({ ...prev, status: 'ACTIVE' })); }
    };
    const endMission = async () => {
        try { syncMission((await apiEndMission()).mission); }
        catch { setMissionState(prev => ({ ...prev, status: 'ENDED' })); }
    };

    const patchReview = async (frameId, status, notes = null) => {
        try {
            const response = await reviewDetection(frameId, status, notes);
            if (response?.detection) {
                setDetections(prev => prev.map(item =>
                    item.frame_id === frameId ? { ...item, ...response.detection } : item
                ));
            }
            setBackendOnline(true);
        } catch (error) {
            setBackendOnline(false);
            setDetections(prev => prev.map(item => item.frame_id === frameId
                ? { ...item, review_status: status, ...(notes !== null ? { operator_notes: notes } : {}) }
                : item
            ));
        }
    };

    const confirmDetection = (frameId) => patchReview(frameId, 'confirmed');
    const dismissDetection = (frameId) => patchReview(frameId, 'dismissed');
    const flagDetection = (frameId) => patchReview(frameId, 'flagged');
    const updateNotes = (frameId, notes) => patchReview(frameId, 'pending', notes);

    const addDetection = async (newDet) => {
        const result = await apiAddDetection(newDet);
        const persisted = result?.detection || newDet;
        setDetections(prev => [persisted, ...prev]);
        setMissionState(prev => ({
            ...prev,
            detectionsCount: prev.detectionsCount + 1,
        }));
        setBackendOnline(true);
        return persisted;
    };

    const setSettings = (updater) => {
        setSettingsState(prev => typeof updater === 'function' ? updater(prev) : updater);
    };

    const resetDemo = async () => {
        try {
            const result = await resetBackend();
            applyBackendState(result.state);
            setBackendOnline(true);
        } catch {
            setMissionState(INITIAL_MISSION_STATE);
            setDetections(INITIAL_DETECTIONS);
            setSettingsState(INITIAL_SETTINGS);
        }
    };

    const value = {
        missionState,
        detections,
        telemetry,
        telemetryLog,
        settings,
        classCounts,
        setSettings,
        startMission,
        pauseMission,
        resumeMission,
        endMission,
        confirmDetection,
        dismissDetection,
        flagDetection,
        updateNotes,
        addDetection,
        resetDemo,
        refreshBackend,
        backendOnline,
        backendHealth,
        backendError,
        runtimeBenchmark,
        lastGoodState: lastGoodState.current,
    };

    return <SimulationContext.Provider value={value}>{children}</SimulationContext.Provider>;
}

export function useSimulation() {
    const context = useContext(SimulationContext);
    if (!context) throw new Error('useSimulation must be used within a SimulationProvider');
    return context;
}
