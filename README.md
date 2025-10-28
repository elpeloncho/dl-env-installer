# Instalador de Entornos para Deep Learning (˶ᵔ ᵕ ᵔ˶)

Script automatizado para configurar entornos virtuales de TensorFlow y PyTorch con soporte para GPU NVIDIA/AMD en GNU/Linux.

## Características

- [+] Detección automática de GPU (NVIDIA/AMD)
- [+] Instalación selectiva (TensorFlow, PyTorch o ambos)
- [+] Configuración automática de kernels para Jupyter
- [+] Soporte para CUDA (NVIDIA) y ROCm (AMD)

## Requisitos Previos

### Básicos
- Python 3.8 o superior
- `pip` y `venv`
- Git (para clonar el repositorio)

### Para GPU NVIDIA
- **NVIDIA drivers** (>= 550 recomendado)

### Para GPU AMD
- **Drivers AMDGPU + ROCm** (>= 6.0)
- **GPU compatible:** RX 6000+, RX 7000+, MI series
  
    Verifica que tu GPU AMD sea compatible en:
https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html

## Instalación de Drivers

### 🟢 NVIDIA Drivers
#### Comprueba si tienes los drivers ya instalados:
```bash
nvidia-smi
```
#### Ubuntu / Debian
```bash
sudo apt update
# Ver versiones disponibles
apt search nvidia-driver
sudo apt install -y nvidia-driver-550
sudo reboot
```

#### Fedora / RHEL / CentOS
```bash
# Habilitar RPM Fusion
sudo dnf install -y https://download1.rpmfusion.org/free/fedora/rpmfusion-free-release-$(rpm -E %fedora).noarch.rpm
sudo dnf install -y https://download1.rpmfusion.org/nonfree/fedora/rpmfusion-nonfree-release-$(rpm -E %fedora).noarch.rpm

# Instalar drivers
sudo dnf install -y akmod-nvidia
sudo reboot
```

#### Arch Linux / Manjaro
```bash
sudo pacman -S nvidia nvidia-utils
sudo reboot
```

#### openSUSE
```bash
sudo zypper addrepo --refresh https://download.nvidia.com/opensuse/leap/15.5 nvidia
sudo zypper install -y nvidia-driver-G06-kmp-default
sudo reboot
```

**Verificar instalación:**
```bash
nvidia-smi
```

---

### 🔴 AMD ROCm Drivers
#### Comprueba si tienes los drivers y ROCm ya instalados:
```bash
rocm-smi
rocminfo
```
#### Ubuntu / Debian
```bash
# Ubuntu 22.04 (recomendado)
wget https://repo.radeon.com/amdgpu-install/6.2.4/ubuntu/jammy/amdgpu-install_6.2.60204-1_all.deb
sudo apt install -y ./amdgpu-install_6.2.60204-1_all.deb

# Instalar ROCm
sudo amdgpu-install --usecase=rocm

# Agregar usuario a grupos necesarios
sudo usermod -a -G render,video $USER

sudo reboot
```

#### Fedora / RHEL
```bash
sudo dnf install -y https://repo.radeon.com/amdgpu-install/6.2.4/rhel/9.4/amdgpu-install-6.2.60204-1.el9.noarch.rpm
sudo amdgpu-install --usecase=rocm
sudo usermod -a -G render,video $USER
sudo reboot
```

#### Arch Linux / Manjaro
```bash
sudo pacman -S rocm-hip-sdk rocm-smi-lib
sudo usermod -a -G render,video $USER
sudo reboot
```

#### openSUSE
```bash
sudo zypper addrepo https://repo.radeon.com/rocm/zypp/6.2.4/main rocm
sudo zypper install rocm-hip-sdk
sudo usermod -a -G render,video $USER
sudo reboot
```

**Verificar instalación:**
```bash
rocm-smi
rocminfo
```


---

## Uso del Script

### 1. Clonar el repositorio
```bash
git clone https://github.com/TU-USUARIO/dl-env-installer.git
cd dl-env-installer
```

### 2. Dar permisos de ejecución
```bash
chmod +x setup_dl_env.sh
```

### 3. Ejecutar el script
```bash
./setup_dl_env.sh
```

### 4. Seguir las instrucciones
El script te preguntará:
- ¿Qué framework instalar? (TensorFlow / PyTorch / Ambos)
- ¿Configurar kernels de Jupyter? (Y/n)

## Activar los Entornos

### TensorFlow
```bash
source .tf_venv/bin/activate
# Trabajar con TensorFlow...
deactivate
```

### PyTorch
```bash
source .torch_venv/bin/activate
# Trabajar con PyTorch...
deactivate
```

## Usar con Jupyter

Si configuraste los kernels de Jupyter:
```bash
# Iniciar Jupyter (desde cualquier entorno o el sistema base)
jupyter notebook

# En el navegador:
# 1. Crea un nuevo notebook
# 2. Ve a: Kernel > Change kernel
# 3. Selecciona "Python (TensorFlow)" o "Python (PyTorch)"
```

Ver kernels instalados:
```bash
jupyter kernelspec list
```

## Instalar Librerías Adicionales

Si necesitas instalar más librerías (numpy, pandas, matplotlib, etc.):
```bash
# Para TensorFlow
source .tf_venv/bin/activate
pip install numpy pandas matplotlib scikit-learn
deactivate

# Para PyTorch
source .torch_venv/bin/activate
pip install numpy pandas matplotlib scikit-learn
deactivate
```

## Verificación de GPU

### Verificar TensorFlow
```bash
source .tf_venv/bin/activate
python -c "import tensorflow as tf; print('GPUs:', tf.config.list_physical_devices('GPU'))"
deactivate
```

### Verificar PyTorch
```bash
source .torch_venv/bin/activate
python -c "import torch; print('CUDA disponible:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
deactivate
```

## Problemas comunes

### NVIDIA: nvidia-smi no funciona
```bash
sudo nvidia-modprobe
```

### NVIDIA: Conflicto con nouveau
```bash
sudo bash -c "echo blacklist nouveau > /etc/modprobe.d/blacklist-nvidia-nouveau.conf"
sudo update-initramfs -u
sudo reboot
```

### AMD: rocm-smi no funciona
```bash
# Verificar que el dispositivo existe
ls -la /dev/kfd

# Verificar grupos del usuario
groups  # Debe mostrar 'render' y 'video'

# Si no aparecen, cerrar sesión y volver a entrar
```

### TensorFlow/PyTorch no detecta la GPU
```bash
# NVIDIA: Verificar drivers
nvidia-smi

# AMD: Verificar ROCm
rocm-smi

# Reinstalar el framework en el entorno virtual
source .tf_venv/bin/activate  # o .torch_venv
pip uninstall tensorflow  # o torch
# Ejecutar el script nuevamente
```


## Estructura de Archivos Creados
```
.
├── .tf_venv/          # Entorno virtual de TensorFlow
├── .torch_venv/       # Entorno virtual de PyTorch
└── setup_dl_env.sh    # Este script
```


## Licencia

MIT License - Ver archivo LICENSE para más detalles

## Recursos Útiles

- [TensorFlow GPU Support](https://www.tensorflow.org/install/gpu)
- [PyTorch Get Started](https://pytorch.org/get-started/locally/)
- [NVIDIA CUDA GPUs](https://developer.nvidia.com/cuda-gpus)
- [AMD ROCm Documentation](https://rocm.docs.amd.com/)
- [ROCm GPU Support](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html)

---

(˶ᵔ ᵕ ᵔ˶) ¡Feliz entrenamiento!
