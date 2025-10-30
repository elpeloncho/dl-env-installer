#!/usr/bin/env bash
set -e

# =============================================================================
# CONFIGURACIÓN Y CONSTANTES
# =============================================================================

readonly SCRIPT_NAME="Deep Learning Installer"
readonly TF_VENV=".tf_venv"
readonly TORCH_VENV=".torch_venv"
readonly TF_VERSION="2.20.0"
readonly TORCH_VERSION="2.6.0"
readonly CUDA_VERSION="cu124"

# Variables globales
INSTALL_TF=false
INSTALL_TORCH=false
ADD_JUPYTER=false
GPU_TYPE="none"
GPU_INFO=""
PYTHON_BIN=""
MODE="install"  # install o remove

# =============================================================================
# FUNCIONES DE UTILIDAD
# =============================================================================

print_header() {
    echo ""
    echo "  (˶ᵔ ᵕ ᵔ˶)"
    echo "  ¡Bienvenido al gestor de entornos para Deep Learning!"
    echo "  =================================================="
    echo ""
}

print_section() {
    echo ""
    echo "=== $1 ==="
}

log_info() {
    echo "[i] $1"
}

log_success() {
    echo "[+] $1"
}

log_warning() {
    echo "[!] $1"
}

log_action() {
    echo "[*] $1"
}

log_question() {
    echo "[?] $1"
}

log_process() {
    echo "[>] $1"
}

# =============================================================================
# FUNCIONES DE DETECCIÓN
# =============================================================================

detect_python() {
    PYTHON_BIN=$(command -v python || command -v python3)
    
    if [ -z "$PYTHON_BIN" ]; then
        log_warning "Python no encontrado en el sistema. Saliendo..."
        exit 1
    fi
    
    log_success "Python detectado: $PYTHON_BIN"
}

detect_jupyter() {
    JUPYTER_BIN=$(command -v jupyter)
    
    if [ -z "$JUPYTER_BIN" ]; then
        log_warning "Jupyter no encontrado en el sistema. Saliendo..."
        exit 1
    fi
    
    log_success "Jupyter detectado: $JUPYTER_BIN"
}

detect_nvidia_gpu() {
    if command -v nvidia-smi &> /dev/null && nvidia-smi &> /dev/null; then
        GPU_TYPE="nvidia"
        GPU_INFO=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
        log_success "GPU NVIDIA detectada: $GPU_INFO"
        return 0
    fi
    return 1
}

detect_amd_gpu() {
    if command -v rocm-smi &> /dev/null && rocm-smi &> /dev/null; then
        GPU_TYPE="amd"
        GPU_INFO=$(rocm-smi --showproductname | grep -i "GPU" | head -n 1 || echo "AMD GPU")
        log_success "GPU AMD detectada: $GPU_INFO"
        return 0
    elif lspci 2>/dev/null | grep -i "vga\|3d\|display" | grep -i "amd\|radeon" &> /dev/null; then
        GPU_TYPE="amd"
        GPU_INFO=$(lspci | grep -i "vga\|3d" | grep -i "amd\|radeon" | head -n 1)
        log_warning "GPU AMD detectada pero ROCm no está instalado: $GPU_INFO"
        log_info "Para soporte GPU con AMD, instala ROCm: https://rocm.docs.amd.com/"
        return 0
    fi
    return 1
}

detect_gpu() {
    print_section "Detectando hardware GPU"
    
    if detect_nvidia_gpu; then
        return
    fi
    
    if detect_amd_gpu; then
        return
    fi
    
    log_warning "No se detectó GPU compatible (NVIDIA/AMD)"
    log_info "Se instalarán versiones CPU de los frameworks"
}

# =============================================================================
# FUNCIONES DE CONFIGURACIÓN DE USUARIO
# =============================================================================

prompt_mode_selection() {
    log_question "¿Qué deseas hacer?"
    echo "    1) Instalar entornos"
    echo "    2) Eliminar entornos"
    echo ""
    read -p "Selecciona una opción [1/2]: " choice
    
    case $choice in
        1)
            MODE="install"
            log_action "Modo: Instalación"
            ;;
        2)
            MODE="remove"
            log_action "Modo: Eliminación"
            ;;
        *)
            log_warning "Opción inválida. Saliendo..."
            exit 1
            ;;
    esac
}

check_existing_venv() {
    local venv_path=$1
    local venv_name=$2
    
    if [ -d "$venv_path" ]; then
        echo ""
        log_warning "El entorno $venv_name ya existe en: $venv_path"
        log_warning "Si continúas, se ELIMINARÁ completamente y se reinstalará desde cero."
        echo ""
        read -p "[?] ¿Deseas SOBREESCRIBIR el entorno $venv_name? [y/N]: " choice
        choice=${choice:-N}
        
        if [[ "$choice" =~ ^[Yy]$ ]]; then
            log_action "Eliminando entorno antiguo: $venv_path"
            rm -rf "$venv_path"
            log_success "Entorno eliminado correctamente"
            return 0
        else
            log_info "Operación cancelada por el usuario"
            exit 0
        fi
    fi
    return 0
}

prompt_framework_selection() {
    log_question "¿Qué framework deseas instalar?"
    echo "    1) TensorFlow"
    echo "    2) PyTorch"
    echo "    3) Ambos (TensorFlow y PyTorch)"
    echo ""
    read -p "Selecciona una opción [1/2/3]: " choice
    
    case $choice in
        1)
            INSTALL_TF=true
            log_action "Se instalará TensorFlow"
            check_existing_venv "$TF_VENV" "TensorFlow"
            ;;
        2)
            INSTALL_TORCH=true
            log_action "Se instalará PyTorch"
            check_existing_venv "$TORCH_VENV" "PyTorch"
            ;;
        3)
            INSTALL_TF=true
            INSTALL_TORCH=true
            log_action "Se instalarán TensorFlow y PyTorch"
            check_existing_venv "$TF_VENV" "TensorFlow"
            check_existing_venv "$TORCH_VENV" "PyTorch"
            ;;
        *)
            log_warning "Opción inválida. Saliendo..."
            exit 1
            ;;
    esac
}

prompt_jupyter_configuration() {
    echo ""
    read -p "[?] ¿Deseas añadir los entornos como kernels de Jupyter? [Y/n]: " choice
    choice=${choice:-Y}
    
    if [[ "$choice" =~ ^[Yy]$ ]]; then
        ADD_JUPYTER=true
        log_success "Se configurarán los kernels de Jupyter"
    else
        log_info "No se configurarán kernels de Jupyter"
    fi
}

# =============================================================================
# FUNCIONES DE ELIMINACIÓN
# =============================================================================

detect_existing_venvs() {
    local tf_exists=false
    local torch_exists=false
    
    [ -d "$TF_VENV" ] && tf_exists=true
    [ -d "$TORCH_VENV" ] && torch_exists=true
    
    if [ "$tf_exists" = false ] && [ "$torch_exists" = false ]; then
        echo ""
        log_warning "No se encontraron entornos virtuales existentes."
        log_info "No hay nada que eliminar. Saliendo..."
        exit 0
    fi
    
    echo ""
    log_info "Entornos virtuales detectados:"
    if [ "$tf_exists" = true ]; then
        echo "    ✓ TensorFlow: $TF_VENV"
    fi
    if [ "$torch_exists" = true ]; then
        echo "    ✓ PyTorch: $TORCH_VENV"
    fi
}

prompt_removal_selection() {
    echo ""
    log_question "¿Qué entorno(s) deseas eliminar?"
    
    local options=()
    local option_num=1
    
    if [ -d "$TF_VENV" ]; then
        echo "    $option_num) TensorFlow ($TF_VENV)"
        options[$option_num]="tf"
        ((option_num++))
    fi
    
    if [ -d "$TORCH_VENV" ]; then
        echo "    $option_num) PyTorch ($TORCH_VENV)"
        options[$option_num]="torch"
        ((option_num++))
    fi
    
    if [ -d "$TF_VENV" ] && [ -d "$TORCH_VENV" ]; then
        echo "    $option_num) Ambos"
        options[$option_num]="both"
        ((option_num++))
    fi
    
    echo "    0) Cancelar"
    echo ""
    read -p "Selecciona una opción: " choice
    
    if [ "$choice" = "0" ]; then
        log_info "Operación cancelada"
        exit 0
    fi
    
    local selected="${options[$choice]}"
    
    if [ -z "$selected" ]; then
        log_warning "Opción inválida. Saliendo..."
        exit 1
    fi
    
    case $selected in
        tf)
            remove_venv "$TF_VENV" "TensorFlow"
            ;;
        torch)
            remove_venv "$TORCH_VENV" "PyTorch"
            ;;
        both)
            remove_venv "$TF_VENV" "TensorFlow"
            remove_venv "$TORCH_VENV" "PyTorch"
            ;;
    esac
}

remove_venv() {
    local venv_path=$1
    local venv_name=$2
    
    echo ""
    log_warning "Vas a eliminar el entorno: $venv_name ($venv_path)"
    log_warning "Esta acción NO se puede deshacer."
    echo ""
    read -p "[?] ¿Estás seguro de que deseas eliminar $venv_name? [y/N]: " confirm
    confirm=${confirm:-N}
    
    if [[ "$confirm" =~ ^[Yy]$ ]]; then
        log_action "Eliminando $venv_name..."
        
        # Eliminar el kernel de Jupyter si existe
        remove_jupyter_kernel "$venv_name"
        
        # Eliminar el entorno virtual
        rm -rf "$venv_path"
        log_success "Entorno $venv_name eliminado correctamente"
    else
        log_info "Eliminación de $venv_name cancelada"
    fi
}

remove_jupyter_kernel() {
    local venv_name=$1
    local kernel_name=""
    
    case $venv_name in
        "TensorFlow")
            kernel_name="tensorflow-env"
            ;;
        "PyTorch")
            kernel_name="pytorch-env"
            ;;
    esac
    
    if [ -n "$kernel_name" ] && jupyter kernelspec list 2>/dev/null | grep -q "$kernel_name"; then
        log_action "Eliminando kernel de Jupyter: $kernel_name"
        jupyter kernelspec uninstall "$kernel_name" -f 2>/dev/null || true
        log_success "Kernel eliminado"
    fi
}

print_removal_summary() {
    echo ""
    log_success "Operación de eliminación completada"
    echo ""
    log_info "Kernels disponibles:"
    jupyter kernelspec list
    echo ""
    echo "  (˶ᵔ ᵕ ᵔ˶) ¡Hasta luego!"
    echo ""
}

# =============================================================================
# FUNCIONES DE INSTALACIÓN
# =============================================================================

create_venv_if_needed() {
    local venv_path=$1
    local venv_name=$2
    
    if [ ! -d "$venv_path" ]; then
        log_action "Creando entorno $venv_path..."
        $PYTHON_BIN -m venv "$venv_path"
    fi
}

upgrade_pip() {
    log_process "Actualizando pip..."
    pip install --upgrade pip -q
}

install_tensorflow_gpu() {
    case $GPU_TYPE in
        nvidia)
            echo "    -> Instalando TensorFlow $TF_VERSION con soporte CUDA..."
            pip install "tensorflow[and-cuda]==$TF_VERSION"
            ;;
        amd)
            if command -v rocm-smi &> /dev/null; then
                echo "    -> Instalando tensorflow-rocm (soporte AMD ROCm)..."
                pip install tensorflow-rocm
            else
                log_warning "ROCm no detectado, instalando versión CPU..."
                pip install "tensorflow==$TF_VERSION"
            fi
            ;;
        *)
            echo "    -> Instalando TensorFlow $TF_VERSION (CPU)..."
            pip install "tensorflow==$TF_VERSION"
            ;;
    esac
}

verify_tensorflow() {
    log_question "Verificando TensorFlow..."
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
}

install_tensorflow() {
    print_section "Configurando TensorFlow"
    
    create_venv_if_needed "$TF_VENV" "TensorFlow"
    source "$TF_VENV/bin/activate"
    
    upgrade_pip
    install_tensorflow_gpu
    verify_tensorflow
    
    deactivate
    
    if [ "$ADD_JUPYTER" = true ]; then
        add_jupyter_kernel "$TF_VENV" "tensorflow-env" "Python (TensorFlow)"
    fi
}

install_pytorch_gpu() {
    case $GPU_TYPE in
        nvidia)
            echo "    -> Instalando PyTorch $TORCH_VERSION con CUDA 12.4..."
            pip install torch torchvision torchaudio --index-url "https://download.pytorch.org/whl/$CUDA_VERSION"
            ;;
        amd)
            if command -v rocm-smi &> /dev/null; then
                local rocm_version=$(rocm-smi --version 2>/dev/null | grep -oP "ROCm version: \K[\d.]+" || echo "6.0")
                local rocm_major=$(echo "$rocm_version" | cut -d. -f1)
                echo "    -> ROCm version detectada: $rocm_version"
                pip install torch torchvision torchaudio --index-url "https://download.pytorch.org/whl/rocm${rocm_major}.2"
            else
                log_warning "ROCm no detectado, instalando versión CPU..."
                pip install torch torchvision torchaudio --index-url "https://download.pytorch.org/whl/cpu"
            fi
            ;;
        *)
            echo "    -> Instalando PyTorch $TORCH_VERSION (CPU)..."
            pip install torch torchvision torchaudio --index-url "https://download.pytorch.org/whl/cpu"
            ;;
    esac
}

verify_pytorch() {
    log_question "Verificando PyTorch..."
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
}

install_pytorch() {
    print_section "Configurando PyTorch"
    
    create_venv_if_needed "$TORCH_VENV" "PyTorch"
    source "$TORCH_VENV/bin/activate"
    
    upgrade_pip
    install_pytorch_gpu
    verify_pytorch
    
    deactivate
    
    if [ "$ADD_JUPYTER" = true ]; then
        add_jupyter_kernel "$TORCH_VENV" "pytorch-env" "Python (PyTorch)"
    fi
}

add_jupyter_kernel() {
    local venv_path=$1
    local kernel_name=$2
    local display_name=$3
    
    log_action "Configurando kernel de Jupyter: $display_name"
    source "$venv_path/bin/activate"
    
    pip install ipykernel jupyter -q
    python -m ipykernel install --user --name="$kernel_name" --display-name="$display_name"
    
    log_success "Kernel '$display_name' añadido correctamente"
    deactivate
}

# =============================================================================
# FUNCIONES DE RESUMEN
# =============================================================================

print_summary() {
    echo ""
    echo "[OK] Instalación completada con éxito."
    echo ""
    echo "--- Resumen de configuración:"
    echo "    * GPU detectada: $GPU_TYPE"
    
    if [ "$GPU_TYPE" != "none" ]; then
        echo "    * Info: $GPU_INFO"
    fi
    
    if [ "$INSTALL_TF" = true ]; then
        echo "    * TensorFlow: instalado en $TF_VENV"
    fi
    
    if [ "$INSTALL_TORCH" = true ]; then
        echo "    * PyTorch: instalado en $TORCH_VENV"
    fi
    
    if [ "$ADD_JUPYTER" = true ]; then
        echo "    * Kernels de Jupyter: configurados"
    fi
}

print_activation_instructions() {
    echo ""
    echo "==> Activa los entornos con:"
    
    if [ "$INSTALL_TF" = true ]; then
        echo "    source $TF_VENV/bin/activate   # para TensorFlow"
    fi
    
    if [ "$INSTALL_TORCH" = true ]; then
        echo "    source $TORCH_VENV/bin/activate # para PyTorch"
    fi
}

print_jupyter_instructions() {
    if [ "$ADD_JUPYTER" = true ]; then
        echo ""
        log_info "Para usar los kernels en Jupyter:"
        echo "    1. Inicia Jupyter: jupyter notebook"
        echo "    2. Crea un nuevo notebook"
        echo "    3. Selecciona el kernel deseado desde: Kernel > Change kernel"
        echo ""
        log_info "Kernels disponibles:"
        jupyter kernelspec list
    fi
}

print_installation_reminder() {
    echo ""
    echo "[!] RECORDATORIO IMPORTANTE:"
    echo "    Si necesitas instalar librerías adicionales (numpy, pandas, etc.),"
    echo "    debes hacerlo DENTRO del entorno virtual correspondiente:"
    echo ""
    
    if [ "$INSTALL_TF" = true ]; then
        echo "    # Para TensorFlow:"
        echo "    source $TF_VENV/bin/activate"
        echo "    pip install <nombre-libreria>"
        echo "    deactivate"
        echo ""
    fi
    
    if [ "$INSTALL_TORCH" = true ]; then
        echo "    # Para PyTorch:"
        echo "    source $TORCH_VENV/bin/activate"
        echo "    pip install <nombre-libreria>"
        echo "    deactivate"
    fi
}

print_rocm_reminder() {
    if [ "$GPU_TYPE" = "amd" ] && ! command -v rocm-smi &> /dev/null; then
        echo ""
        log_warning "NOTA: Para aprovechar tu GPU AMD, instala ROCm:"
        echo "    https://rocm.docs.amd.com/projects/install-on-linux/en/latest/"
    fi
}

print_goodbye() {
    echo ""
    echo "  (˶ᵔ ᵕ ᵔ˶) ¡Feliz entrenamiento!"
    echo ""
}

# =============================================================================
# FUNCIÓN PRINCIPAL
# =============================================================================

main() {
    print_header
    
    # Selección de modo
    prompt_mode_selection
    
    # Detección del sistema
    detect_python
    detect_jupyter
    
    if [ "$MODE" = "remove" ]; then
        # Modo eliminación
        detect_existing_venvs
        prompt_removal_selection
        print_removal_summary
    else
        # Modo instalación
        prompt_framework_selection
        prompt_jupyter_configuration
        # Detección de hardware
        detect_gpu
        # Instalación de frameworks
        if [ "$INSTALL_TF" = true ]; then
            install_tensorflow
        fi
        
        if [ "$INSTALL_TORCH" = true ]; then
            install_pytorch
        fi
        
        # Resumen e instrucciones finales
        print_summary
        print_activation_instructions
        print_jupyter_instructions
        print_installation_reminder
        print_rocm_reminder
        print_goodbye
    fi
}

# =============================================================================
# PUNTO DE ENTRADA
# =============================================================================

main "$@"
