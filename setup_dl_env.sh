#!/usr/bin/env bash
set -e

# --- Mensaje de bienvenida ---
echo ""
echo "  (˶ᵔ ᵕ ᵔ˶)"
echo "  ¡Bienvenido al instalador de entornos de Deep Learning!"
echo "  =================================================="
echo ""

# --- Selección de frameworks ---
echo "[?] ¿Qué framework deseas instalar?"
echo "    1) TensorFlow"
echo "    2) PyTorch"
echo "    3) Ambos (TensorFlow y PyTorch)"
echo ""
read -p "Selecciona una opción [1/2/3]: " FRAMEWORK_CHOICE

case $FRAMEWORK_CHOICE in
    1)
        INSTALL_TF=true
        INSTALL_TORCH=false
        echo "[*] Se instalará TensorFlow"
        ;;
    2)
        INSTALL_TF=false
        INSTALL_TORCH=true
        echo "[*] Se instalará PyTorch"
        ;;
    3)
        INSTALL_TF=true
        INSTALL_TORCH=true
        echo "[*] Se instalarán TensorFlow y PyTorch"
        ;;
    *)
        echo "[!] Opción inválida. Saliendo..."
        exit 1
        ;;
esac

echo ""
read -p "[?] ¿Deseas añadir los entornos como kernels de Jupyter? [Y/n]: " JUPYTER_CHOICE
JUPYTER_CHOICE=${JUPYTER_CHOICE:-Y}

if [[ "$JUPYTER_CHOICE" =~ ^[Yy]$ ]]; then
    ADD_JUPYTER=true
    echo "[+] Se configurarán los kernels de Jupyter"
else
    ADD_JUPYTER=false
    echo "[i] No se configurarán kernels de Jupyter"
fi

# Detectar Python
PYTHON_BIN=$(command -v python || command -v python3)

# --- Detección de GPU ---
echo ""
echo "[?] Detectando hardware GPU..."

GPU_TYPE="none"
GPU_INFO=""

# Detectar NVIDIA
if command -v nvidia-smi &> /dev/null; then
    if nvidia-smi &> /dev/null; then
        GPU_TYPE="nvidia"
        GPU_INFO=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
        echo "[+] GPU NVIDIA detectada: $GPU_INFO"
    fi
fi

# Detectar AMD
if [ "$GPU_TYPE" = "none" ]; then
    if command -v rocm-smi &> /dev/null; then
        if rocm-smi &> /dev/null; then
            GPU_TYPE="amd"
            GPU_INFO=$(rocm-smi --showproductname | grep -i "GPU" | head -n 1 || echo "AMD GPU")
            echo "[+] GPU AMD detectada: $GPU_INFO"
        fi
    elif lspci 2>/dev/null | grep -i "vga\|3d\|display" | grep -i "amd\|radeon" &> /dev/null; then
        GPU_TYPE="amd"
        GPU_INFO=$(lspci | grep -i "vga\|3d" | grep -i "amd\|radeon" | head -n 1)
        echo "[!] GPU AMD detectada pero ROCm no esta instalado: $GPU_INFO"
        echo "    Para soporte GPU con AMD, instala ROCm: https://rocm.docs.amd.com/"
    fi
fi

# Sin GPU detectada
if [ "$GPU_TYPE" = "none" ]; then
    echo "[!] No se detecto GPU compatible (NVIDIA/AMD)"
    echo "    Se instalaran versiones CPU de los frameworks"
fi

# --- Función para añadir kernel de Jupyter ---
add_jupyter_kernel() {
    local venv_path=$1
    local kernel_name=$2
    local display_name=$3
    
    echo "[*] Configurando kernel de Jupyter: $display_name"
    source "$venv_path/bin/activate"
    
    pip install ipykernel jupyter -q
    python -m ipykernel install --user --name="$kernel_name" --display-name="$display_name"
    
    echo "[+] Kernel '$display_name' añadido correctamente"
    deactivate
}

# --- TensorFlow ---
if [ "$INSTALL_TF" = true ]; then
    echo ""
    echo "=== Configurando TensorFlow ==="
    
    if [ ! -d ".tf_venv" ]; then
        echo "[*] Creando entorno .tf_venv..."
        $PYTHON_BIN -m venv .tf_venv
    else
        echo "[!] El entorno .tf_venv ya existe, se reutilizara."
    fi
    
    source .tf_venv/bin/activate
    
    echo "[>] Instalando TensorFlow..."
    pip install --upgrade pip -q
    
    case $GPU_TYPE in
        nvidia)
            echo "    -> Instalando TensorFlow 2.20.0 con soporte CUDA..."
            pip install tensorflow[and-cuda]==2.20.0
            ;;
        amd)
            echo "    -> Instalando tensorflow-rocm (soporte AMD ROCm)..."
            if command -v rocm-smi &> /dev/null; then
                pip install tensorflow-rocm
            else
                echo "    [!] ROCm no detectado, instalando version CPU..."
                pip install tensorflow==2.20.0
            fi
            ;;
        *)
            echo "    -> Instalando TensorFlow 2.20.0 (CPU)..."
            pip install tensorflow==2.20.0
            ;;
    esac
    
    echo "[?] Verificando TensorFlow..."
    python - <<'EOF'
import tensorflow as tf
print("\n=== TensorFlow ===")
print("Version:", tf.__version__)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print("[+] GPU detectada:", gpus)
    for gpu in gpus:
        print(f"    - {gpu}")
else:
    print("[i] Ejecutando en modo CPU")
EOF
    
    deactivate
    
    # Añadir kernel de Jupyter si se solicitó
    if [ "$ADD_JUPYTER" = true ]; then
        add_jupyter_kernel ".tf_venv" "tensorflow-env" "Python (TensorFlow)"
    fi
fi

# --- PyTorch ---
if [ "$INSTALL_TORCH" = true ]; then
    echo ""
    echo "=== Configurando PyTorch ==="
    
    if [ ! -d ".torch_venv" ]; then
        echo "[*] Creando entorno .torch_venv..."
        $PYTHON_BIN -m venv .torch_venv
    else
        echo "[!] El entorno .torch_venv ya existe, se reutilizara."
    fi
    
    source .torch_venv/bin/activate
    
    echo "[>] Instalando PyTorch..."
    pip install --upgrade pip -q
    
    case $GPU_TYPE in
        nvidia)
            echo "    -> Instalando PyTorch 2.6.0 con CUDA 12.4..."
            pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
            ;;
        amd)
            echo "    -> Instalando PyTorch con soporte ROCm..."
            if command -v rocm-smi &> /dev/null; then
                # Detectar versión de ROCm
                ROCM_VERSION=$(rocm-smi --version 2>/dev/null | grep -oP "ROCm version: \K[\d.]+" || echo "6.0")
                ROCM_MAJOR=$(echo $ROCM_VERSION | cut -d. -f1)
                echo "    ROCm version detectada: $ROCM_VERSION"
                pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm${ROCM_MAJOR}.2
            else
                echo "    [!] ROCm no detectado, instalando version CPU..."
                pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
            fi
            ;;
        *)
            echo "    -> Instalando PyTorch 2.6.0 (CPU)..."
            pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
            ;;
    esac
    
    echo "[?] Verificando PyTorch..."
    python - <<'EOF'
import torch
print("\n=== PyTorch ===")
print("Version:", torch.__version__)
print("CUDA disponible:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("[+] GPU CUDA:")
    for i in range(torch.cuda.device_count()):
        print(f"    - {torch.cuda.get_device_name(i)}")
elif hasattr(torch.version, 'hip') and torch.version.hip:
    print("[+] ROCm disponible:", torch.version.hip)
else:
    print("[i] Ejecutando en modo CPU")
EOF
    
    deactivate
    
    # Añadir kernel de Jupyter si se solicitó
    if [ "$ADD_JUPYTER" = true ]; then
        add_jupyter_kernel ".torch_venv" "pytorch-env" "Python (PyTorch)"
    fi
fi

# --- Resumen final ---
echo ""
echo "[OK] Instalacion completada con exito."
echo ""
echo "--- Resumen de configuracion:"
echo "    * GPU detectada: $GPU_TYPE"
if [ "$GPU_TYPE" != "none" ]; then
    echo "    * Info: $GPU_INFO"
fi
if [ "$INSTALL_TF" = true ]; then
    echo "    * TensorFlow: instalado en .tf_venv"
fi
if [ "$INSTALL_TORCH" = true ]; then
    echo "    * PyTorch: instalado en .torch_venv"
fi
if [ "$ADD_JUPYTER" = true ]; then
    echo "    * Kernels de Jupyter: configurados"
fi
echo ""
echo "==> Activa los entornos con:"
if [ "$INSTALL_TF" = true ]; then
    echo "    source .tf_venv/bin/activate   # para TensorFlow"
fi
if [ "$INSTALL_TORCH" = true ]; then
    echo "    source .torch_venv/bin/activate # para PyTorch"
fi

if [ "$ADD_JUPYTER" = true ]; then
    echo ""
    echo "[i] Para usar los kernels en Jupyter:"
    echo "    1. Inicia Jupyter: jupyter notebook"
    echo "    2. Crea un nuevo notebook"
    echo "    3. Selecciona el kernel deseado desde: Kernel > Change kernel"
    echo ""
    echo "[i] Kernels disponibles:"
    jupyter kernelspec list
fi

echo ""
echo "[!] RECORDATORIO IMPORTANTE:"
echo "    Si necesitas instalar librerias adicionales (numpy, pandas, etc.),"
echo "    debes hacerlo DENTRO del entorno virtual correspondiente:"
echo ""
if [ "$INSTALL_TF" = true ]; then
    echo "    # Para TensorFlow:"
    echo "    source .tf_venv/bin/activate"
    echo "    pip install <nombre-libreria>"
    echo "    deactivate"
    echo ""
fi
if [ "$INSTALL_TORCH" = true ]; then
    echo "    # Para PyTorch:"
    echo "    source .torch_venv/bin/activate"
    echo "    pip install <nombre-libreria>"
    echo "    deactivate"
fi

if [ "$GPU_TYPE" = "amd" ] && ! command -v rocm-smi &> /dev/null; then
    echo ""
    echo "[!] NOTA: Para aprovechar tu GPU AMD, instala ROCm:"
    echo "    https://rocm.docs.amd.com/projects/install-on-linux/en/latest/"
fi

echo ""
echo "  (˶ᵔ ᵕ ᵔ˶) ¡Feliz entrenamiento!"
echo ""
