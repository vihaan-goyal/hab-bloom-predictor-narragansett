# run_satellite_pipeline.ps1 -- detached driver for the satellite feasibility test
# Runs viability (two product groups in parallel), then the full pull, then the
# evaluation and figure. Logs under data\transfer\satellite\. Launch detached:
#   Start-Process powershell -ArgumentList '-NoProfile','-ExecutionPolicy','Bypass','-File','src\transfer\run_satellite_pipeline.ps1' -WindowStyle Hidden
$ErrorActionPreference = "Continue"
Set-Location "C:\Users\vihaa\hab-bloom-predictor-narragansett"
$py = "C:\Users\vihaa\anaconda3\python.exe"
$log = "data\transfer\satellite"
New-Item -ItemType Directory -Force $log | Out-Null
"START $(Get-Date -Format s)" | Out-File "$log\pipeline.log" -Append

$a = Start-Process $py -ArgumentList "-u -m src.transfer.satellite_fetch --stage viability --products viirs4k,dineof2k,olci4k --max-wait 300" -RedirectStandardOutput "$log\viability.log" -RedirectStandardError "$log\viability.err" -PassThru -WindowStyle Hidden
$b = Start-Process $py -ArgumentList "-u -m src.transfer.satellite_fetch --stage viability --products olci300" -RedirectStandardOutput "$log\viability_olci300.log" -RedirectStandardError "$log\viability_olci300.err" -PassThru -WindowStyle Hidden
$a.WaitForExit(); $b.WaitForExit()
"VIABILITY DONE $(Get-Date -Format s)" | Out-File "$log\pipeline.log" -Append

$c = Start-Process $py -ArgumentList "-u -m src.transfer.satellite_fetch --stage pull --products viirs4k,dineof2k,olci4k,sst --start 2016-01-01 --end 2023-12-31 --max-wait 600" -RedirectStandardOutput "$log\pull.log" -RedirectStandardError "$log\pull.err" -PassThru -WindowStyle Hidden
$d = Start-Process $py -ArgumentList "-u -m src.transfer.satellite_fetch --stage pull --products olci300 --start 2016-01-01 --end 2023-12-31" -RedirectStandardOutput "$log\pull_olci300.log" -RedirectStandardError "$log\pull_olci300.err" -PassThru -WindowStyle Hidden
$c.WaitForExit(); $d.WaitForExit()
"PULL DONE $(Get-Date -Format s)" | Out-File "$log\pipeline.log" -Append

& $py -u -m src.transfer.satellite_eval *> "$log\eval.log"
"EVAL DONE $(Get-Date -Format s)" | Out-File "$log\pipeline.log" -Append
& $py -u src\viz\satellite_figure.py *> "$log\figure.log"
"FIGURE DONE $(Get-Date -Format s)" | Out-File "$log\pipeline.log" -Append
"END $(Get-Date -Format s)" | Out-File "$log\pipeline.log" -Append
