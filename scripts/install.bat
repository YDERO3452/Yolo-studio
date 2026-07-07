@echo off
chcp 65001 >nul 2>&1
setlocal enabledelayedexpansion

echo ============================================
echo   YOLO Studio - 一键安装脚本 (Windows)
echo ============================================
echo.

REM ---- 配置国内镜像源 ----
set PIP_MIRROR=https://mirrors.aliyun.com/pypi/simple
set PIP_TRUSTED_HOST=mirrors.aliyun.com
REM 加大重试和超时, 避免镜像源断连导致安装失败
set PIP_EXTRA=--retries 5 --timeout 120

REM ---- 检测 Python ----
python --version >nul 2>&1
if errorlevel 1 (
    echo [错误] 未找到 Python，请先安装 Python 3.10+
    echo 下载地址: https://www.python.org/downloads/
    pause
    exit /b 1
)
for /f "tokens=2 delims= " %%v in ('python --version 2^>^&1') do set PYVER=%%v
echo [OK] Python %PYVER%

REM ---- 检测 pip ----
python -m pip --version >nul 2>&1
if errorlevel 1 (
    echo [错误] pip 不可用，请重新安装 Python 并勾选 pip
    pause
    exit /b 1
)
echo [OK] pip 可用
echo.

REM ---- 配置国内镜像源 ----
echo [1/3] 配置 pip 国内镜像源 -- 阿里云...
python -m pip config set global.index-url %PIP_MIRROR%
python -m pip config set global.trusted-host %PIP_TRUSTED_HOST%
python -m pip config set global.timeout 120
echo [OK] 镜像源: 阿里云
echo.

REM ---- 升级 pip ----
echo [2/3] 升级 pip...
python -m pip install --upgrade pip
echo.

REM ---- 检测 NVIDIA GPU ----
set HAS_NVIDIA=0
nvidia-smi >nul 2>&1
if errorlevel 1 goto :no_gpu

set HAS_NVIDIA=1
echo [检测到] NVIDIA GPU
nvidia-smi 2>&1 | findstr /C:"CUDA Version"
goto :gpu_done

:no_gpu
echo [信息] 未检测到 NVIDIA GPU，将安装 CPU 版本

:gpu_done
echo.

REM ---- 选择安装模式 ----
if not "!HAS_NVIDIA!"=="1" goto :force_cpu

echo nvidia-smi 输出的 CUDA Version 即驱动支持的最高 CUDA 版本:
echo   CUDA 12.4+ -^> cu124    CUDA 12.1+ -^> cu121    CUDA 11.8+ -^> cu118
echo.
set /p "choice=安装 CUDA 版 PyTorch? [Y/n] : "
if /i "!choice!"=="n" goto :force_cpu

set INSTALL_MODE=cuda
set /p "cu_tag=请输入 CUDA 版本标签 cu118/cu121/cu124, 默认 cu121: "
if "!cu_tag!"=="" set cu_tag=cu121
goto :install_torch

:force_cpu
set INSTALL_MODE=cpu

:install_torch
echo.
echo [3/3] 安装 PyTorch -- !INSTALL_MODE! 模式...

if not "!INSTALL_MODE!"=="cuda" goto :install_cpu_torch

echo 正在安装 CUDA 版 PyTorch !cu_tag! -- 约 2.5 GB...
echo   PyTorch 官方源, 国内镜像无 CUDA wheel
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/!cu_tag!
if not errorlevel 1 goto :torch_done

echo.
echo [警告] !cu_tag! 安装失败, 尝试 cu121...
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
if not errorlevel 1 goto :torch_done

echo.
echo [警告] cu121 也失败, 尝试 cu118...
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
if not errorlevel 1 goto :torch_done

echo.
echo [警告] CUDA 版全部失败, 回退 CPU 版...
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
goto :torch_done

:install_cpu_torch
echo 正在安装 CPU 版 PyTorch -- 约 200 MB...
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

:torch_done
echo.

REM ---- 安装项目依赖 ----
echo 安装项目依赖 -- 国内镜像源...
python -m pip install -r requirements.txt %PIP_EXTRA%
if errorlevel 1 (
    echo.
    echo [错误] 依赖安装失败, 请检查网络或手动执行:
    echo   pip install -r requirements.txt
    pause
    exit /b 1
)
echo.

REM ---- 验证安装 ----
echo ============================================
echo   验证安装结果
echo ============================================
python -c "import torch; print(f'  PyTorch: {torch.__version__}'); print(f'  CUDA available: {torch.cuda.is_available()}'); print(f'  CUDA version: {torch.version.cuda or \"N/A (CPU)\"}')"
python -c "import ultralytics; print(f'  Ultralytics: {ultralytics.__version__}')"
python -c "import PyQt6; print(f'  PyQt6: OK')"
python -c "import cv2; print(f'  OpenCV: {cv2.__version__}')"

echo.
echo ============================================
echo   安装完成！
echo ============================================
echo.
echo 启动:  python main.py
echo.
pause
