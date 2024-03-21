rem @echo off

rem Array of parameters
rem set "si_lambda=0.001 0.005 0.01 0.05 0.1 0.5"
rem set "ewc_lambda=0.001 0.005 0.01 0.05 0.1 0.5"
rem set "alpha=0.1 0.2 0.3 0.4 0.5"
rem set "temperature=2 3 4 5"

set "si_lambda=0.001 0.5"
set "ewc_lambda=0.001 0.5"
set "alpha=0.1 0.5"
set "temperature=2 5"

rem Run with -s SI and -si_lambda
echo Running with -s SI and -si_lambda
for %%i in (%si_lambda%) do (
    python main.py -s SI -si_lambda %%i -e 0
)

rem Run with -s EWC and -ewc_lambda
echo Running with -s EWC and -ewc_lambda
for %%i in (%ewc_lambda%) do (
    python main.py -s EWC -ewc_lambda %%i -e 0
)

rem Run with -s LWF and -alpha and -temperature
echo Running with -s LWF and -alpha and -temperature
for %%a in (%alpha%) do (
    for %%t in (%temperature%) do (
        python main.py -s LWF -alpha %%a -temperature %%t -e 0
    )
)

rem Run with -s ICARL
rem echo Running with -s ICARL
rem python main.py -s ICARL -e 0