# PARALLEL PROGRAMMING FINAL PROJECT

## Autoencoder-based unsupervised feature learning system

### 1. Directory Structure
```text
.
├── src/                        # Source code
│   ├── train_cpu.cpp           # Main training program (CPU Version)
│   ├── train_gpu.cu            # Main training program (GPU Baseline)
│   ├── train_gpu_optimize.cu   # Main training program (GPU Optimized)
│   ├── kernels_cpu.cpp         # CPU Math kernels
│   ├── kernel.cu               # GPU Kernels (Baseline)
│   ├── optimize_kernel.cu      # GPU Kernels (Optimized)
│   └── cifar10_dataset.cpp     # Data loader implementation
├── include/                    # Header files
│   ├── host.h                  # CPU Helper declarations
│   ├── kernel.h                # GPU Kernel declarations
│   ├── optimize_kernel.h       # Optimized GPU Kernel declarations
│   └── cifar10_dataset.h       # Data loader header
├── data/                       # Dataset directory
│   └── cifar-10-batches-bin/   # Extracted CIFAR-10 binary files
├── build/                      # Compiled executables (output folder)
├── weights/                    # Saved binary weights (output folder)
└── README.md
```

### 2. How to Run

#### Step 1: Preparation

Ensure you have the dataset in `data/cifar-10-batches-bin` and create output directories:

```bash
mkdir -p build
mkdir -p weights
```

#### Step 2: Compile

You can compile different versions of the project depending on your hardware and requirements.

**1. CPU Version**
```bash
!g++ -std=c++17 -O3 src/train_cpu.cpp src/cifar10_dataset.cpp src/kernels_cpu.cpp -I include -o build/cpu_train
```

**2. GPU Baseline Version** (Requires NVIDIA GPU & CUDA Toolkit)
```bash
!nvcc -arch=sm_75 -o build/train_gpu_baseline src/train_gpu_baseline.cu src/kernel.cu src/cifar10_dataset.cpp -I include/
```

**3. GPU Optimized Version** (Requires NVIDIA GPU & CUDA Toolkit)
```bash
!nvcc -arch=sm_75 -o build/train_gpu_optimize_all src/train_gpu_optimize_all.cu src/kernel.cu src/optimize_kernel.cu src/cifar10_dataset.cpp -I include/
```

#### Step 3: Execute

Navigate to the build directory and run the desired executable:

```bash
cd build
```

**Run CPU Version:**
```bash
./cpu_train
```

**Run GPU Baseline:**
```bash
./train_gpu_baseline
```

**Run GPU Optimized:**
```bash
./train_gpu_optimize_all
```

### 3. Notes
- The training process will save weights to the `weights/` directory automatically.
- Ensure CUDA is properly installed and added to your PATH for GPU compilation.