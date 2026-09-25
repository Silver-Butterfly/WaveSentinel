import React, { useEffect, useState, useRef } from 'react';
import { ArrowLeft, Play, Pause, RotateCcw } from 'lucide-react';
import './Sim2DPage.css';

// Fallback images since the original assets are missing
const oceanBg = "https://images.unsplash.com/photo-1518837695005-2083093ee35b?auto=format&fit=crop&w=1920&q=80";
const shipImg = "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='80' height='30'%3E%3Cpath d='M0,15 L15,5 L65,5 L80,15 L65,25 L15,25 Z' fill='%23777' stroke='%23333'/%3E%3Ctext x='40' y='19' font-size='12' text-anchor='middle' fill='%23fff' font-family='sans-serif'%3ESHIP%3C/text%3E%3C/svg%3E";
const fishImg = "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='40' height='20'%3E%3Cpath d='M40,10 C30,20 10,20 0,10 C10,0 30,0 40,10 Z' fill='%2300d8ff'/%3E%3Cpolygon points='0,10 -10,0 -10,20' fill='%2300d8ff'/%3E%3C/svg%3E";

export default function Sim2DPage({ onBack }) {
  const [dots, setDots] = useState([]);
  const [shipPath, setShipPath] = useState([]);
  const [isDrawing, setIsDrawing] = useState(false);
  const [isFishFollowing, setIsFishFollowing] = useState(false);
  const [shipAnimating, setShipAnimating] = useState(false);
  const [pathIndex, setPathIndex] = useState(0);
  const [isPaused, setIsPaused] = useState(false);
  const [hasStarted, setHasStarted] = useState(false);
  const [alertVisible, setAlertVisible] = useState(false);
  const [dotStates, setDotStates] = useState([]);

  // Positions
  const initialShipPos = useRef({ x: window.innerWidth * 0.12, y: window.innerHeight * 0.85 });
  const initialFishPos = useRef({ x: window.innerWidth * 0.12, y: window.innerHeight * 0.35 });

  const shipPosRef = useRef({ ...initialShipPos.current });
  const fishPosRef = useRef({ ...initialFishPos.current });

  const [shipPos, setShipPos] = useState({ ...initialShipPos.current });
  const [fishPos, setFishPos] = useState({ ...initialFishPos.current });

  const pathLineRef = useRef(null);
  const animFrameRef = useRef(null);
  const simLoopFrameRef = useRef(null);

  const shipPathRef = useRef([]);
  shipPathRef.current = shipPath;

  const pathIndexRef = useRef(pathIndex);
  pathIndexRef.current = pathIndex;

  const shipAnimatingRef = useRef(shipAnimating);
  shipAnimatingRef.current = shipAnimating;

  const isPausedRef = useRef(isPaused);
  isPausedRef.current = isPaused;

  const alertTriggeredRef = useRef(false);
  const lastAlertTimeRef = useRef(0);

  // Initialize Semicircle Hydrophones
  useEffect(() => {
    const numDots = 6;
    const centerX = window.innerWidth;
    const centerY = window.innerHeight / 2;
    const radius = Math.min(window.innerWidth, window.innerHeight) * 0.4;
    const newDots = [];

    for (let i = 0; i < numDots; i++) {
      const angle = Math.PI / 2 + (Math.PI * (i / (numDots - 1)));
      const x = centerX + radius * Math.cos(angle);
      const y = centerY + radius * Math.sin(angle);
      newDots.push({ id: `h${i + 1}`, label: `h${i + 1}`, x, y });
    }

    setDots(newDots);
    setDotStates(new Array(numDots).fill('none'));
  }, []);

  // Update positions helper
  const updateShipPos = (pos) => {
    shipPosRef.current = pos;
    setShipPos({ ...pos });
  };

  const updateFishPos = (pos) => {
    fishPosRef.current = pos;
    setFishPos({ ...pos });
  };

  // Distance calculator
  const distance = (p1, p2) => Math.hypot(p1.x - p2.x, p1.y - p2.y);
  // Helper to check if ship and fish are in collision/red-alert zone for any dot
  const checkRedAlert = (sPos, fPos, dotsList) => {
    if (!sPos || !fPos || !dotsList || dotsList.length === 0) return false;
    return dotsList.some((dot) => {
      const dFish = Math.hypot(fPos.x - dot.x, fPos.y - dot.y);
      const dShip = Math.hypot(sPos.x - dot.x, sPos.y - dot.y);
      return dFish < 140 && dShip < 140;
    });
  };

  // Core Proximity Simulation Loop
  useEffect(() => {
    if (dots.length === 0) return;

    const runSimulationLoop = () => {
      const currentDotStates = new Array(dots.length).fill('none');
      let shouldStopShip = false;

      const PRIMARY_RANGE = 90;
      const SECONDARY_RANGE = 140;

      dots.forEach((dot, index) => {
        const dFish = distance(fishPosRef.current, dot);
        const dShip = distance(shipPosRef.current, dot);

        const isFishPrimary = dFish < PRIMARY_RANGE;
        const isFishSecondary = dFish >= PRIMARY_RANGE && dFish < SECONDARY_RANGE;

        const isShipPrimary = dShip < PRIMARY_RANGE;
        const isShipSecondary = dShip >= PRIMARY_RANGE && dShip < SECONDARY_RANGE;

        const isFishInRange = isFishPrimary || isFishSecondary;
        const isShipInRange = isShipPrimary || isShipSecondary;

        if (isShipPrimary && isFishPrimary) {
          currentDotStates[index] = 'glow-deep-red';
          shouldStopShip = true;
        } else if (isShipInRange && isFishInRange) {
          currentDotStates[index] = 'glow-red';
          shouldStopShip = true;
        } else if (isShipPrimary) {
          currentDotStates[index] = 'glow-orange';
        } else if (isFishPrimary) {
          currentDotStates[index] = 'glow-green';
        } else if (isFishSecondary) {
          currentDotStates[index] = 'glow-yellow';
        } else {
          currentDotStates[index] = 'none';
        }
      });

      if (shouldStopShip && (shipAnimatingRef.current || !isPausedRef.current)) {
        if (animFrameRef.current) {
          cancelAnimationFrame(animFrameRef.current);
          animFrameRef.current = null;
        }
        setShipAnimating(false);
        shipAnimatingRef.current = false;
        setIsPaused(true);
        isPausedRef.current = true;

        if (!alertTriggeredRef.current && Date.now() - lastAlertTimeRef.current > 5000) {
          alertTriggeredRef.current = true;
          lastAlertTimeRef.current = Date.now();
          setAlertVisible(true);
          setTimeout(() => {
            setAlertVisible(false);
            alertTriggeredRef.current = false;
          }, 3000);
        }
      }

      setDotStates(currentDotStates);
      simLoopFrameRef.current = requestAnimationFrame(runSimulationLoop);
    };

    simLoopFrameRef.current = requestAnimationFrame(runSimulationLoop);

    return () => {
      if (simLoopFrameRef.current) cancelAnimationFrame(simLoopFrameRef.current);
    };
  }, [dots]);

  // Ship Movement Animation Loop
  const animateShip = () => {
    if (!shipAnimatingRef.current || isPausedRef.current) return;

    const speed = 2.5; // pixels per frame
    const path = shipPathRef.current;
    let idx = pathIndexRef.current;

    if (idx < path.length - 1) {
      let target = path[idx + 1];
      let current = { ...shipPosRef.current };
      let dist = distance(current, target);

      if (dist <= speed) {
        current = { ...target };
        idx++;
        setPathIndex(idx);
        pathIndexRef.current = idx;
      } else {
        let ratio = speed / dist;
        current.x += (target.x - current.x) * ratio;
        current.y += (target.y - current.y) * ratio;
      }

      updateShipPos(current);
      animFrameRef.current = requestAnimationFrame(animateShip);
    } else {
      setShipAnimating(false);
      shipAnimatingRef.current = false;
      setIsPaused(true);
      isPausedRef.current = true;
    }
  };

  useEffect(() => {
    if (shipAnimating && !isPaused) {
      animFrameRef.current = requestAnimationFrame(animateShip);
    }
    return () => {
      if (animFrameRef.current) cancelAnimationFrame(animFrameRef.current);
    };
  }, [shipAnimating, isPaused]);

  // Mouse Handlers for drawing ship path & fish following
  const handleMouseMove = (e) => {
    if (isDrawing) {
      setShipPath((prev) => [...prev, { x: e.clientX, y: e.clientY }]);
    }
    if (isFishFollowing) {
      updateFishPos({ x: e.clientX, y: e.clientY });
    }
  };

  const handleMouseUp = () => {
    if (isDrawing) {
      setIsDrawing(false);
    }
  };

  // Ship double click to start drawing
  const handleShipDoubleClick = (e) => {
    e.stopPropagation();
    setIsDrawing(true);
    setShipPath([{ ...shipPosRef.current }]);
  };

  // Fish click to toggle following
  const handleFishClick = (e) => {
    e.stopPropagation();
    setIsFishFollowing((prev) => !prev);
    if (!isFishFollowing) {
      updateFishPos({ x: e.clientX, y: e.clientY });
    }
  };

  // Action Button Handler (Start Simulation / Reset)
  const handleActionClick = () => {
    if (!hasStarted) {
      if (shipPath.length === 0) return;
      setHasStarted(true);
      setShipAnimating(true);
      shipAnimatingRef.current = true;
      setIsPaused(false);
      isPausedRef.current = false;
      setPathIndex(0);
      pathIndexRef.current = 0;
    } else {
      // Reset
      setHasStarted(false);
      setShipAnimating(false);
      shipAnimatingRef.current = false;
      setIsDrawing(false);
      setIsFishFollowing(false);
      setIsPaused(false);
      isPausedRef.current = false;
      setPathIndex(0);
      pathIndexRef.current = 0;
      setShipPath([]);
      updateShipPos({ ...initialShipPos.current });
      updateFishPos({ ...initialFishPos.current });
      setAlertVisible(false);
      alertTriggeredRef.current = false;
    }
  };

  // Play/Pause Handler
  const togglePlayPause = () => {
    const nextPaused = !isPaused;
    setIsPaused(nextPaused);
    isPausedRef.current = nextPaused;
    if (!nextPaused) {
      setShipAnimating(true);
      shipAnimatingRef.current = true;
    } else {
      setShipAnimating(false);
      shipAnimatingRef.current = false;
    }
  };

  // Timeline Slider Scrubbing
  const handleSliderChange = (e) => {
    if (shipPath.length === 0) return;
    let targetIndex = parseInt(e.target.value, 10);

    if (targetIndex > pathIndex) {
      let collisionIndex = -1;
      for (let i = pathIndex; i <= targetIndex; i++) {
        let tempPos = shipPath[i];
        let collision = dots.some((dot) => {
          return distance(fishPosRef.current, dot) < 140 && distance(tempPos, dot) < 140;
        });
        if (collision) {
          collisionIndex = i;
          break;
        }
      }
      if (collisionIndex !== -1) {
        targetIndex = collisionIndex;
      }
    }

    setPathIndex(targetIndex);
    pathIndexRef.current = targetIndex;
    if (targetIndex < shipPath.length) {
      updateShipPos({ ...shipPath[targetIndex] });
    }
  };

  const pointsString = shipPath.map((p) => `${p.x},${p.y}`).join(' ');

  return (
    <div
      className="sim2d-page"
      onMouseMove={handleMouseMove}
      onMouseUp={handleMouseUp}
    >
      {/* Ocean Background */}
      <div
        className="sim2d-bg-container"
        style={{ backgroundImage: `url(${oceanBg})` }}
      />

      {/* Top Nav */}
      <div className="sim2d-nav-bar">
        <button className="sim2d-back-btn" onClick={onBack}>
          <ArrowLeft size={16} />
          RETURN TO COMMAND HUB
        </button>
      </div>

      {/* SVG Canvas overlay for drawn route & detection radii */}
      <svg id="path-canvas" className="fullscreen-overlay">
        {dots.map((dot, index) => {
          const state = dotStates[index];
          if (!state || state === 'none') return null;

          let ringColor = 'transparent';
          let radius = 90;

          if (state === 'glow-green') {
            ringColor = 'rgba(0, 255, 0, 0.35)';
            radius = 90;
          } else if (state === 'glow-yellow') {
            ringColor = 'rgba(255, 255, 0, 0.3)';
            radius = 140;
          } else if (state === 'glow-orange') {
            ringColor = 'rgba(255, 165, 0, 0.35)';
            radius = 90;
          } else if (state === 'glow-deep-red') {
            ringColor = 'rgba(255, 0, 0, 0.45)';
            radius = 90;
          } else if (state === 'glow-red') {
            ringColor = 'rgba(255, 0, 0, 0.3)';
            radius = 140;
          }

          return (
            <circle
              key={`radius-${dot.id}`}
              cx={dot.x}
              cy={dot.y}
              r={radius}
              fill="none"
              stroke={ringColor}
              strokeWidth="1.5"
              strokeDasharray="4, 4"
            />
          );
        })}
        {shipPath.length > 1 && (
          <polyline
            ref={pathLineRef}
            points={pointsString}
            className="sim2d-path-line"
          />
        )}
      </svg>

      {/* Ship Image */}
      <img
        src={shipImg}
        alt="Tactical Subsea Vessel"
        className="draggable-entity"
        style={{ left: `${shipPos.x}px`, top: `${shipPos.y}px` }}
        onDoubleClick={handleShipDoubleClick}
        title="Double-click to draw path"
      />

      {/* Fish Image */}
      <div
        className="interactive-entity"
        style={{ position: 'absolute', left: `${fishPos.x}px`, top: `${fishPos.y}px`, zIndex: 30 }}
        onClick={handleFishClick}
        title="Click to toggle follow mouse"
      >
        <img src={fishImg} alt="Marine Life Target" style={{ width: '100%', height: 'auto' }} />
        {isFishFollowing && <span className="sim2d-fish-badge">TRACKING</span>}
      </div>

      {/* Hydrophone Dots */}
      <div id="dots-container" className="fullscreen-overlay">
        {dots.map((dot, index) => {
          const stateClass = dotStates[index] && dotStates[index] !== 'none' ? dotStates[index] : '';
          return (
            <div
              key={dot.id}
              className={`dot ${stateClass}`}
              style={{ left: `${dot.x}px`, top: `${dot.y}px` }}
            >
              <div className="dot-label">{dot.label}</div>
            </div>
          );
        })}
      </div>

      {/* Action Controls */}
      <div className="sim2d-controls">
        <button className="action-btn" onClick={handleActionClick}>
          {hasStarted ? 'RESET SIMULATION' : 'START SIMULATION'}
        </button>
      </div>

      {/* Color Key Legend */}
      <div className="color-key-panel">
        <ul className="color-key-list">
          <li>
            <span className="legend-color" style={{ background: '#111', border: '1px solid #555' }}></span> Default (Black)
          </li>
          <li>
            <span className="legend-color" style={{ background: '#00ff00', boxShadow: '0 0 8px #00ff00' }}></span> Mammal Nearby
          </li>
          <li>
            <span className="legend-color" style={{ background: '#ffff00', boxShadow: '0 0 8px #ffff00' }}></span> Mammal around proximity
          </li>
          <li>
            <span className="legend-color" style={{ background: '#ffa500', boxShadow: '0 0 8px #ffa500' }}></span> Vessel Nearby
          </li>
          <li>
            <span className="legend-color" style={{ background: '#810000', boxShadow: '0 0 8px #ff0000' }}></span> Collision Risk
          </li>
          <li>
            <span className="legend-color" style={{ background: '#ff0000', boxShadow: '0 0 8px #ff0000' }}></span> Collision Alert
          </li>
        </ul>
      </div>

      {/* Timeline Controls (Visible when started) */}
      {hasStarted && (
        <div className="timeline-panel">
          <button className="play-pause-btn" onClick={togglePlayPause}>
            {isPaused ? <Play size={16} /> : <Pause size={16} />}
          </button>
          <input
            type="range"
            className="timeline-slider"
            min="0"
            max={shipPath.length > 0 ? shipPath.length - 1 : 0}
            value={pathIndex}
            onChange={handleSliderChange}
          />
        </div>
      )}

      {/* Custom Alert Box */}
      {alertVisible && (
        <div className="custom-alert-box">
          ⚠️ ALERT! Ship and Marine Mammal are both near a hydrophone!
        </div>
      )}
    </div>
  );
}
