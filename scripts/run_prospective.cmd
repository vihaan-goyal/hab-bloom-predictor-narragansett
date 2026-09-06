@echo off
rem run_prospective.cmd -- score due ledger rows, then issue this week's forecast.
rem Both steps append to data\prospective\logs\last_run.log. Base conda env only.
rem Usage: scripts\run_prospective.cmd   (schedule weekly once ISEF Form 1A is signed)
cd /d "%~dp0.."
if not exist data\prospective\logs mkdir data\prospective\logs
set PY=C:\Users\vihaa\anaconda3\python.exe
echo ==== %date% %time% score_ledger >> data\prospective\logs\last_run.log
"%PY%" -m src.deploy.score_ledger >> data\prospective\logs\last_run.log 2>&1
echo ==== %date% %time% prospective_forecast >> data\prospective\logs\last_run.log
"%PY%" -m src.deploy.prospective_forecast >> data\prospective\logs\last_run.log 2>&1
