@echo off
setlocal

set PY="D:\debris_envs\debris\Scripts\python.exe"
set SCRIPT="K:\Debris model\src\phase_g3_runtime_benchmark.py"
set ONNX="K:\Debris model\runs\phase_g1_onnx\best.onnx"
set ENGINE="K:\Debris model\runs\phase_g2_tensorrt\best_fp32.engine"
set IMAGES="D:\Datasets\DRISHTI-SSS-ROBUST-V2-HN\val\images"
set OUT="K:\Debris model\runs\phase_g3_runtime"

%PY% %SCRIPT% preflight --onnx %ONNX% --engine %ENGINE% --output %OUT%\preflight.json
if errorlevel 1 exit /b 1

%PY% %SCRIPT% benchmark --onnx %ONNX% --engine %ENGINE% --images %IMAGES% --output-dir %OUT%
if errorlevel 1 exit /b 1

echo.
echo G3 benchmark complete. Evidence: %OUT%
endlocal
