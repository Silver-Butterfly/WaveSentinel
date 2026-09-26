@echo off
setlocal
cd /d "%~dp0"
python phase_g4_deployment.py preflight ^
  --onnx "K:\Debris model\runs\phase_g1_onnx\best.onnx" ^
  --fp32 "K:\Debris model\runs\phase_g2_tensorrt\best_fp32.engine" ^
  --output "K:\Debris model\runs\phase_g4_deployment\preflight.json"
if errorlevel 1 exit /b 1

python phase_g4_deployment.py validate ^
  --onnx "K:\Debris model\runs\phase_g1_onnx\best.onnx" ^
  --fp32 "K:\Debris model\runs\phase_g2_tensorrt\best_fp32.engine" ^
  --output "K:\Debris model\runs\phase_g4_deployment\g4_deployment_integrity.json"
if errorlevel 1 exit /b 1

echo.
echo G4 FINAL DEPLOYMENT INTEGRITY: PASS
echo Report: K:\Debris model\runs\phase_g4_deployment\g4_deployment_integrity.json
