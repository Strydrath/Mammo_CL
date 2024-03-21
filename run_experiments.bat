rem @echo off

rem Array of parameters
set "si_lambda=0.001 0.005 0.01 0.05 0.1 0.5"
set "ewc_lambda=0.001 0.005 0.01 0.05 0.1 0.5"
set "alpha=0.1 0.2 0.3 0.4 0.5"
set "temperature=2 3 4 5"
set "db_o=VRD VDR RVD RDV DVR DRV"

rem Run with -s SI and -si_lambda and -db_o
echo Running with -s SI and -si_lambda and -db_o
for %%s in (SI) do (
    for %%i in (%si_lambda%) do (
        for %%db in (%db_o%) do (
            python main.py -s %%s -si_lambda %%i -db_o %%db
        )
    )
)

rem Run with -s EWC and -ewc_lambda and -db_o
echo Running with -s EWC and -ewc_lambda and -db_o
for %%s in (EWC) do (
    for %%i in (%ewc_lambda%) do (
        for %%db in (%db_o%) do (
            python main.py -s %%s -ewc_lambda %%i -db_o %%db
        )
    )
)

rem Run with -s LWF and -alpha and -temperature and -db_o
echo Running with -s LWF and -alpha and -temperature and -db_o
for %%s in (LWF) do (
    for %%a in (%alpha%) do (
        for %%t in (%temperature%) do (
            for %%db in (%db_o%) do (
                python main.py -s %%s -alpha %%a -temperature %%t -db_o %%db
            )
        )
    )
)

rem Run with -s ICARL
echo Running with -s ICARL
python main.py -s ICARL
