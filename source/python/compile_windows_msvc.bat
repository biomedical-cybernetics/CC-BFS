@echo off
SETLOCAL
REM ============================================================================
REM  Compile script for Windows using MSVC (Visual Studio)
REM  This script compiles all C++ source files into DLLs for Python usage.
REM
REM  PREREQUISITE: Run this script from an MSVC-enabled command prompt, e.g.:
REM    "x64 Native Tools Command Prompt for VS 2022"
REM    "Developer Command Prompt for VS 2022"
REM  These can be found in the Start Menu under "Visual Studio 2022".
REM  Do NOT run from a plain cmd or Conda prompt, or the compiler will not be found.
REM ============================================================================

REM --- Check that the MSVC compiler is available ---
where cl >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo ERROR: MSVC compiler ^(cl.exe^) not found in PATH.
    echo Please run this script from one of the following terminals:
    echo   - "x64 Native Tools Command Prompt for VS 2022"
    echo   - "Developer Command Prompt for VS 2022"
    echo These are available in the Start Menu under "Visual Studio 2022".
    echo.
    exit /b 1
)
echo MSVC compiler found: OK
echo.

REM --- Configuration ---
IF NOT "%~1"=="" SET "EIGEN_PATH=%~1"
IF NOT DEFINED EIGEN_PATH (
    echo ERROR: Eigen include directory is required.
    echo Usage: compile_windows_msvc.bat C:\path\to\eigen-3.4.0
    exit /b 1
)
IF NOT EXIST "%EIGEN_PATH%\Eigen\Dense" (
    echo ERROR: Expected to find "%EIGEN_PATH%\Eigen\Dense".
    exit /b 1
)
REM Resolve a relative Eigen path before entering the build directory.
FOR %%I IN ("%EIGEN_PATH%") DO SET "EIGEN_PATH=%%~fI"

REM --- Paths ---
REM This script assumes it is located in the source/python/ directory
SET "SCRIPT_DIR=%~dp0"
SET "OUTPUT_DIR=%SCRIPT_DIR%lib"
SET "BUILD_DIR=%SCRIPT_DIR%build\windows"
SET "SOURCE_DIR=%SCRIPT_DIR%..\cpp\\"
SET "CC_BFS_P_SOURCE_DIR=%SOURCE_DIR%cc_bfs_p\\"
SET "PYTHON_WRAPPER_DIR=%SCRIPT_DIR%wrappers\\"
SET "DO_BFS_SOURCE_DIR=%SCRIPT_DIR%..\..\third_party\gapbs-do-bfs\\"

REM --- Compiler and Flags ---
SET "COMPILER=cl"
SET "COMMON_FLAGS=/LD /EHsc /O2 /std:c++17 /DPYTHON_BUILD"

REM --- Create output directory ---
echo Creating output directory at %OUTPUT_DIR%...
IF NOT EXIST "%OUTPUT_DIR%" (
    MKDIR "%OUTPUT_DIR%"
)

REM Keep compiler intermediates away from the caller's files. Never delete
REM wildcard-matched objects or libraries from the caller's working directory.
IF NOT EXIST "%BUILD_DIR%" MKDIR "%BUILD_DIR%"
PUSHD "%BUILD_DIR%"
IF %ERRORLEVEL% NEQ 0 exit /b 1
CALL :build
SET "BUILD_RESULT=%ERRORLEVEL%"
POPD
exit /b %BUILD_RESULT%

:build

echo.
echo ============================================================================
echo  Starting compilation with MSVC...
echo ============================================================================
echo.

REM --- Compile bfs_apsp ---
echo Compiling bfs_apsp_l_cpp.cpp...
%COMPILER% %COMMON_FLAGS% /I"%EIGEN_PATH%" "%SOURCE_DIR%bfs_apsp_l_cpp.cpp" /link /OUT:"%OUTPUT_DIR%\bfs_apsp.dll"
IF %ERRORLEVEL% NEQ 0 (
    echo ERROR: Compilation of bfs_apsp_l_cpp.cpp failed.
    exit /b 1
)
echo   -^> Success: bfs_apsp.dll created in lib/
echo.

REM --- Compile cc_bfs_apsp ---
echo Compiling cc_bfs_apsp_l_cpp.cpp...
%COMPILER% %COMMON_FLAGS% /I"%EIGEN_PATH%" /I"%PYTHON_WRAPPER_DIR%" "%SOURCE_DIR%cc_bfs_apsp_l_cpp.cpp" /link /OUT:"%OUTPUT_DIR%\cc_bfs_apsp.dll"
IF %ERRORLEVEL% NEQ 0 (
    echo ERROR: Compilation of cc_bfs_apsp_l_cpp.cpp failed.
    exit /b 1
)
echo   -^> Success: cc_bfs_apsp.dll created in lib/
echo.

REM --- Compile cc_bfs_p ---
echo Compiling CC-BFS-P...
%COMPILER% %COMMON_FLAGS% /openmp /I"%EIGEN_PATH%" /I"%CC_BFS_P_SOURCE_DIR%" /I"%CC_BFS_P_SOURCE_DIR%include" "%CC_BFS_P_SOURCE_DIR%cc_bfs_p.cpp" "%PYTHON_WRAPPER_DIR%cc_bfs_p_python.cpp" /link /OUT:"%OUTPUT_DIR%\cc_bfs_p.dll"
IF %ERRORLEVEL% NEQ 0 (
    echo ERROR: Compilation of CC-BFS-P failed.
    exit /b 1
)
echo   -^> Success: cc_bfs_p.dll created in lib/
echo.

REM --- Compile ess ---
echo Compiling ess_cpp.cpp...
%COMPILER% %COMMON_FLAGS% /I"%EIGEN_PATH%" "%SOURCE_DIR%ess_cpp.cpp" /link /OUT:"%OUTPUT_DIR%\ess.dll"
IF %ERRORLEVEL% NEQ 0 (
    echo ERROR: Compilation of ess_cpp.cpp failed.
    exit /b 1
)
echo   -^> Success: ess.dll created in lib/
echo.

REM --- Compile cc_ess ---
echo Compiling cc_ess_cpp.cpp...
%COMPILER% %COMMON_FLAGS% /I"%EIGEN_PATH%" "%SOURCE_DIR%cc_ess_cpp.cpp" /link /OUT:"%OUTPUT_DIR%\cc_ess.dll"
IF %ERRORLEVEL% NEQ 0 (
    echo ERROR: Compilation of cc_ess_cpp.cpp failed.
    exit /b 1
)
echo   -^> Success: cc_ess.dll created in lib/
echo.

REM --- Compile do_bfs_gapbs_apsp (third-party baseline) ---
echo Compiling do_bfs_gapbs_apsp_l_cpp.cpp (DO-BFS, third-party GAPBS)...
%COMPILER% %COMMON_FLAGS% /I"%EIGEN_PATH%" /I"%DO_BFS_SOURCE_DIR%" "%DO_BFS_SOURCE_DIR%do_bfs_gapbs_apsp_l_cpp.cpp" /link /OUT:"%OUTPUT_DIR%\do_bfs_gapbs_apsp.dll"
IF %ERRORLEVEL% NEQ 0 (
    echo ERROR: Compilation of do_bfs_gapbs_apsp_l_cpp.cpp failed.
    exit /b 1
)
echo   -^> Success: do_bfs_gapbs_apsp.dll created in lib/
echo.

echo ============================================================================
echo  All compilations finished successfully!
echo ============================================================================
echo.

echo Intermediate files are retained in "%BUILD_DIR%".
echo.

exit /b 0
