@ECHO OFF
REM ╠Да©ясЁы
SETLOCAL ENABLEDELAYEDEXPANSION

REM File List
SET LIST=list.txt

REM Log File
SET LOG=exec.log

REM Get file list
IF EXIST "%LIST%" (
  del %LIST%  >> %LOG%
)
dir/b/on/a *.jpg >> %LIST%

FOR /F "tokens=1,2 delims=."  %%a IN (%LIST%) DO (

  SET /a x=!x!+1
  SET nm=00000!x!

  REM file rename
  ren %%a.%%b in!nm:~-6!.jpg
  ECHO ren %%a.%%b in2!nm:~-6!.jpg >> %LOG%

)

ECHO Success!!! >> %LOG%