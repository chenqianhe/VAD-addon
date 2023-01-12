@echo off

set __script_action_type=%1
set __fastdeploy_sdk_dir=%2
set __another_target_dir=%3
set __install_infos_flag=%4

@rem help
if "%__script_action_type%" == "help" (
    call:__print_long_line
    echo [1] [help]    print help information:                      fastdeploy_init.bat help
    echo [2] [show]    show all dlls/libs/include paths:            fastdeploy_init.bat show fastdeploy-sdk-dir
    echo [3] [init]    init all dlls paths for current terminal:    fastdeploy_init.bat init fastdeploy-sdk-dir  [WARNING: need copy onnxruntime.dll manually]
    echo [4] [setup]   setup path env for current terminal:         fastdeploy_init.bat setup fastdeploy-sdk-dir [WARNING: need copy onnxruntime.dll manually]
    echo [5] [install] install all dlls to a specific dir:          fastdeploy_init.bat install fastdeploy-sdk-dir another-dir-to-install-dlls **[RECOMMEND]**
    echo [6] [install] install all dlls with logging infos:         fastdeploy_init.bat install fastdeploy-sdk-dir another-dir-to-install-dlls info
    call:__print_long_line
    goto:eof
)

@rem show dlls and libs
if "%__script_action_type%" == "show" (

    call:__print_long_line
    echo [SDK] %__fastdeploy_sdk_dir%

    call:__print_long_line
    set __have_openvino_flag=false
    set __fastdeploy_lib_dir=%__fastdeploy_sdk_dir%\lib

    @setlocal enabledelayedexpansion
    echo [DLL] !__fastdeploy_lib_dir!\fastdeploy.dll **[NEEDED]**
    for /f "delims= " %%a in ('dir /s /b %__fastdeploy_sdk_dir%\third_libs ^| findstr /e \.dll ^| findstr /v "vc14\\bin\\opencv"') do ( 
        set __3rd_dll_file=%%a && set __3rd_needed_flag=true
        echo !__3rd_dll_file! | findstr "openvino">nul && set __have_openvino_flag=true
        echo !__3rd_dll_file! | findstr d\.dll>nul && set __3rd_needed_flag=false
        if "!__3rd_needed_flag!"=="false" (echo !__3rd_dll_file! | findstr /v opencv>nul && set __3rd_needed_flag=true)
        echo !__3rd_dll_file! | findstr debug\.dll>nul && set __3rd_needed_flag=false
        if "!__3rd_needed_flag!"=="true" (echo [DLL] !__3rd_dll_file! **[NEEDED]**) else (echo [DLL] !__3rd_dll_file!)
    )

    call:__print_long_line
    echo [Lib] !__fastdeploy_lib_dir!\fastdeploy.lib **[NEEDED][fastdeploy]**
    for /f "delims= " %%a in ('dir /s /b %__fastdeploy_sdk_dir%\third_libs ^| findstr /e \.lib ^| findstr /v "vc14\\lib\\opencv"') do ( 
        set __3rd_lib_file=%%a && set __3rd_needed_flag=false && set __api_tag=fastdeploy
        echo !__3rd_lib_file! | findstr "opencv">nul && set __3rd_needed_flag=true
        echo !__3rd_lib_file! | findstr "opencv">nul && set __api_tag=!__api_tag!::vision
        if "!__3rd_needed_flag!"=="true" (echo !__3rd_lib_file! | findstr d\.lib>nul && set __3rd_needed_flag=false)
        echo !__3rd_lib_file! | findstr "fast_tokenizer">nul && set __3rd_needed_flag=true
        echo !__3rd_lib_file! | findstr "fast_tokenizer">nul && set __api_tag=!__api_tag!::text
        if "!__3rd_needed_flag!"=="true" (echo [Lib] !__3rd_lib_file! **[NEEDED][!__api_tag!]**) else (echo [Lib] !__3rd_lib_file!)
    )

    call:__print_long_line
    set __fastdeploy_include_dir=%__fastdeploy_sdk_dir%\include
    echo [Include] !__fastdeploy_include_dir! **[NEEDED][fastdeploy]**
    for /f "delims= " %%a in ('dir /s /b %__fastdeploy_sdk_dir%\third_libs ^| findstr /e include ^| findstr /v "vc14\\bin\\opencv"') do ( 
        set __3rd_include_dir=%%a && set __3rd_needed_flag=false && set __api_tag=fastdeploy
        echo !__3rd_include_dir! | findstr "opencv">nul && set __3rd_needed_flag=true
        echo !__3rd_include_dir! | findstr "opencv">nul && set __api_tag=!__api_tag!::vision
        echo !__3rd_include_dir! | findstr "fast_tokenizer">nul && set __3rd_needed_flag=true
        echo !__3rd_include_dir! | findstr "fast_tokenizer">nul && set __api_tag=!__api_tag!::text
        if "!__3rd_needed_flag!"=="true" (echo [Include] !__3rd_include_dir! **[NEEDED][!__api_tag!]**) else (echo [Include] !__3rd_include_dir!)
    )

    call:__print_long_line  
    if "!__have_openvino_flag!"=="true" (
        for /f "delims= " %%a in ('dir /s /b %__fastdeploy_sdk_dir%\third_libs ^| findstr /e \.xml ^| findstr "openvino"') do ( 
            set __openvino_plugin_xml=%%a
            echo [XML] !__openvino_plugin_xml! **[NEEDED]**
        )
        call:__print_long_line
    )
    @setlocal disabledelayedexpansion
    goto:eof
)

@rem init all paths for dlls
if "%__script_action_type%" == "init" (
    @setlocal enabledelayedexpansion
    set /p yes_or_no=Init dll paths for FastDeploy in current terminal: [y/n]
    if "!yes_or_no!"=="y" (echo YES.) else (echo NO. && pause && goto:eof)
    @setlocal disabledelayedexpansion
    if exist bin.txt (del /Q bin.txt)
    if exist lib.txt (del /Q lib.txt)
    for /f "delims= " %%a in ('dir /s /b /A:D %__fastdeploy_sdk_dir% ^| findstr /v include ^| findstr /e bin ^| findstr /v "vc14\\bin"') do (>>bin.txt set /p=%%a;<nul)
    for /f "delims= " %%a in ('dir /s /b /A:D %__fastdeploy_sdk_dir% ^| findstr /v include ^| findstr /e lib ^| findstr /v "vc14\\lib"') do (>>lib.txt set /p=%%a;<nul)
    for /f %%i in (bin.txt) do (endlocal & set __fastdeploy_bin_paths=%%i)
    for /f %%j in (lib.txt) do (endlocal & set __fastdeploy_lib_paths=%%j)
    pause
    call:__print_long_line
    echo [INFO] FastDeploy dlls paths init done! Please run: [fastdeploy_init.bat setup fastdeploy-sdk-dir]
    echo [INFO] command to push these dlls paths into PATH ENV in current terminal.
    call:__print_long_line
    goto:eof
)

@rem setup PATH ENV for all dlls
if "%__script_action_type%" == "setup" (
    @setlocal enabledelayedexpansion
    set /p yes_or_no=Setup PATH ENV for FastDeploy in current terminal: [y/n]
    if "!yes_or_no!"=="y" (echo YES.) else (echo NO. && pause && goto:eof)
    @setlocal disabledelayedexpansion
    if not exist bin.txt (echo Can not found bin.txt, Please run init before setup && goto:eof)
    if not exist lib.txt (echo Can not found lib.txt, Please run init before setup && goto:eof)
    for /f %%i in (bin.txt) do (endlocal & set __fastdeploy_bin_paths=%%i)
    for /f %%j in (lib.txt) do (endlocal & set __fastdeploy_lib_paths=%%j)
    set "PATH=%__fastdeploy_bin_paths%%__fastdeploy_lib_paths%%PATH%"
    pause
    call:__print_long_line
    echo [INFO] FastDeploy PATH ENV setup done! Please use [set PATH] to check PATH ENV in current terminal.
    echo [INFO] Just setup once again if the paths of FastDeploy can not be found in your PATH ENV.
    call:__print_long_line
    echo [WARN] Please copy all onnxruntime dlls manually to your-exe-or-custom-dll-dir if ENABLE_ORT_BACKEND=ON.
    echo [WARN] Use [fastdeploy_init.bat show fastdeploy-sdk-dir] to find the dll's location of onnxruntime.
    call:__print_long_line
    goto:eof
)

@rem copy all dlls to a specific location  
if "%__script_action_type%" == "install" (
    @setlocal enabledelayedexpansion
    if "!__install_infos_flag!"=="noconfirm" (
        echo [INFO] Installing all FastDeploy dlls ...
    ) else (
        echo [INFO] Do you want to install all FastDeploy dlls ?
        echo [INFO] From: !__fastdeploy_sdk_dir!
        echo [INFO]   To: !__another_target_dir!
        set /p yes_or_no=Choose y means YES, n means NO: [y/n]
        if "!yes_or_no!"=="y" (echo YES.) else (echo NO. && pause && goto:eof)
        pause
    )
    @setlocal disabledelayedexpansion
    if not exist %__fastdeploy_sdk_dir% ( echo [ERROR] %__fastdeploy_sdk_dir% is not exist ! && goto:eof )
    if not exist %__another_target_dir% ( mkdir %__another_target_dir% && echo [INFO] Created %__another_target_dir% done!)
    set __have_openvino_flag=false
    @setlocal enabledelayedexpansion
    for /f "delims= " %%a in ('dir /s /b %__fastdeploy_sdk_dir% ^| findstr /e \.dll ^| findstr /v "vc14\\bin\\opencv"') do ( 
        set __3rd_or_fd_dll_file=%%a && set __3rd_or_fd_needed_flag=true
        echo !__3rd_or_fd_dll_file! | findstr "openvino">nul && set __have_openvino_flag=true
        echo !__3rd_or_fd_dll_file! | findstr d\.dll>nul && set __3rd_or_fd_needed_flag=false
        if "!__3rd_or_fd_needed_flag!"=="false" ( echo !__3rd_or_fd_dll_file! | findstr /v opencv>nul && set __3rd_or_fd_needed_flag=true)
        echo !__3rd_or_fd_dll_file! | findstr debug\.dll>nul && set __3rd_or_fd_needed_flag=false
        if "!__3rd_or_fd_needed_flag!"=="true" (
            copy /Y !__3rd_or_fd_dll_file! %__another_target_dir%
            if "!__install_infos_flag!"=="info" ( echo [Installed][DLL] !__3rd_or_fd_dll_file! "--->" %__another_target_dir%)
        )
    )
    if "!__have_openvino_flag!"=="true" (
        for /f "delims= " %%a in ('dir /s /b %__fastdeploy_sdk_dir% ^| findstr /e \.xml ^| findstr "openvino"') do ( 
            set __openvino_plugin_xml=%%a
            copy /Y !__openvino_plugin_xml! %__another_target_dir%
            if "!__install_infos_flag!"=="info" ( echo [Installed][XML] !__openvino_plugin_xml! "--->" %__another_target_dir% )
        )
    )
    @setlocal disabledelayedexpansion
    goto:eof
)
goto:eof

@rem helpers
:__print_long_line
echo ------------------------------------------------------------------------------------------------------------------------------------------------------------
goto:eof
@rem end

@echo on
