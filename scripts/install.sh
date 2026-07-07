#!/usr/bin/env bash
# YOLO Studio - 一键安装脚本 (Linux / macOS)
set -e

echo "============================================"
echo "  YOLO Studio - 安装脚本 (Linux / macOS)"
echo "============================================"
echo

# ---- 配置国内镜像源（阿里云）----
PIP_MIRROR="https://mirrors.aliyun.com/pypi/simple"
PIP_TRUSTED_HOST="mirrors.aliyun.com"
PIP_EXTRA="--retries 5 --timeout 120"

# ---- 检测 Python ----
PYTHON=""
for cmd in python3 python; do
    if command -v "$cmd" &>/dev/null; then
        PYTHON="$cmd"
        break
    fi
done

if [ -z "$PYTHON" ]; then
    echo "[错误] 未找到 Python，请先安装 Python 3.10+"
    exit 1
fi

PYVER=$($PYTHON --version 2>&1)
echo "[OK] $PYVER ($PYTHON)"

# ---- 检测 pip ----
if ! $PYTHON -m pip --version &>/dev/null; then
    echo "[错误] pip 不可用，请安装 python3-pip"
    exit 1
fi
echo "[OK] pip 可用"
echo

# ---- 配置国内镜像源 ----
echo "[1/3] 配置 pip 国内镜像源 (阿里云)..."
$PYTHON -m pip config set global.index-url "$PIP_MIRROR"
$PYTHON -m pip config set global.trusted-host "$PIP_TRUSTED_HOST"
$PYTHON -m pip config set global.timeout 120
echo "[OK] 镜像源: 阿里云"
echo

# ---- 升级 pip ----
echo "[2/3] 升级 pip..."
$PYTHON -m pip install --upgrade pip
echo

# ---- 检测 NVIDIA GPU ----
HAS_NVIDIA=0
if command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null; then
    HAS_NVIDIA=1
    echo "[检测到] NVIDIA GPU"
    nvidia-smi | grep "CUDA Version" || true
else
    echo "[信息] 未检测到 NVIDIA GPU，将安装 CPU 版本"
fi
echo

# ---- 选择安装模式 ----
INSTALL_MODE="cpu"
if [ "$HAS_NVIDIA" = "1" ]; then
    echo "nvidia-smi 输出的 CUDA Version 即驱动支持的最高 CUDA 版本:"
    echo "  CUDA 12.4+ → cu124    CUDA 12.1+ → cu121    CUDA 11.8+ → cu118"
    echo
    read -p "安装 CUDA 版 PyTorch? [Y/n] (n=CPU版): " choice
    if [ "$choice" != "n" ] && [ "$choice" != "N" ]; then
        read -p "请输入 CUDA 版本标签 (cu118/cu121/cu124，默认 cu121): " cu_tag
        cu_tag="${cu_tag:-cu121}"
        INSTALL_MODE="cuda"
    fi
fi
echo

# ---- 安装 PyTorch ----
echo "[3/3] 安装 PyTorch ($INSTALL_MODE 模式)..."

if [ "$INSTALL_MODE" = "cuda" ]; then
    echo "正在安装 CUDA 版 PyTorch ($cu_tag, 约 2.5 GB)..."
    echo "  (PyTorch 官方源，国内镜像无 CUDA wheel)"
    if ! $PYTHON -m pip install torch torchvision --index-url "https://download.pytorch.org/whl/$cu_tag"; then
        echo
        echo "[警告] $cu_tag 安装失败，尝试 cu121..."
        if ! $PYTHON -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121; then
            echo
            echo "[警告] cu121 也失败，尝试 cu118..."
            if ! $PYTHON -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118; then
                echo
                echo "[警告] CUDA 版全部失败，回退 CPU 版..."
                $PYTHON -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
            fi
        fi
    fi
else
    echo "正在安装 CPU 版 PyTorch (约 200 MB)..."
    $PYTHON -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
fi
echo

# ---- 安装项目依赖 ----
echo "安装项目依赖 (国内镜像源)..."
$PYTHON -m pip install -r requirements.txt $PIP_EXTRA
echo

# ---- 验证安装 ----
echo "============================================"
echo "  验证安装结果"
echo "============================================"
$PYTHON -c "import torch; print(f'  PyTorch: {torch.__version__}'); print(f'  CUDA available: {torch.cuda.is_available()}'); print(f'  CUDA version: {torch.version.cuda or \"N/A (CPU)\"}')"
$PYTHON -c "import ultralytics; print(f'  Ultralytics: {ultralytics.__version__}')"
$PYTHON -c "import PyQt6; print(f'  PyQt6: OK')"
$PYTHON -c "import cv2; print(f'  OpenCV: {cv2.__version__}')"

echo
echo "============================================"
echo "  安装完成！"
echo "============================================"
echo
echo "启动:  $PYTHON main.py"
echo
