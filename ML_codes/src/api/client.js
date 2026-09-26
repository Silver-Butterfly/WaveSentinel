const API_BASE_URL =
    import.meta.env.VITE_API_BASE_URL?.replace(/\/$/, '') ||
    'http://127.0.0.1:8000';

async function request(path, options = {}) {
    const response = await fetch(`${API_BASE_URL}${path}`, options);
    let payload = null;
    try {
        payload = await response.json();
    } catch {
        payload = null;
    }
    if (!response.ok) {
        const detail = payload?.detail || `Backend request failed (${response.status})`;
        throw new Error(detail);
    }
    return payload;
}

export function getApiBaseUrl() { return API_BASE_URL; }
export function getHealth() { return request('/api/health'); }
export function getState() { return request('/api/state'); }
export function getSystemStatus() { return request('/api/system/status'); }
export function getTelemetry() { return request('/api/telemetry'); }
export function getLiveFrame() { return request('/api/live/frame'); }
export function getBatchStatus() { return request('/api/batch/status'); }
export function startBatch() { return request('/api/batch/start', { method: 'POST' }); }

export function startMission() { return request('/api/mission/start', { method: 'POST' }); }
export function pauseMission() { return request('/api/mission/pause', { method: 'POST' }); }
export function resumeMission() { return request('/api/mission/resume', { method: 'POST' }); }
export function endMission() { return request('/api/mission/end', { method: 'POST' }); }

export function reviewDetection(frameId, status, notes = null) {
    return request(`/api/detections/${encodeURIComponent(frameId)}/review`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ status, notes })
    });
}

export function addDetection(detection) {
    return request('/api/detections', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            mission_id: detection.mission_id,
            frame_id: detection.frame_id,
            timestamp: detection.timestamp,
            class_name: detection.class,
            confidence: detection.confidence,
            bbox_xyxy: detection.bbox_xyxy || detection.bbox || [0, 0, 0, 0],
            depth: detection.depth,
            latitude: detection.latitude,
            longitude: detection.longitude,
            position_source: detection.position_source || 'not_available',
            telemetry_source: detection.telemetry_source || 'not_provided',
            motion_correction_state: detection.motion_correction_state || 'not_applied',
            input_quality: detection.input_quality || 'not_evaluated',
            operator_notes: detection.operator_notes || '',
            source: detection.source || 'uploaded_file'
        })
    });
}

export function resetBackend() { return request('/api/settings/reset', { method: 'POST' }); }

export async function inferImage(file, metadata = {}) {
    const form = new FormData();
    form.append('file', file);
    form.append('frame_id', metadata.frameId || 'WS_UPLOAD');
    form.append('timestamp', metadata.timestamp || new Date().toISOString());
    form.append('range_m', metadata.range || 'not_provided');
    form.append('depth', metadata.depth || 'not_provided');
    form.append('survey_line', metadata.surveyLine || 'not_provided');
    return request('/api/inference/image', { method: 'POST', body: form });
}
