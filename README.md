#  PARALLEL PROGRAMMING FINAL PROJECT

## Autoencoder-based unsupervised feature learning system

### 1. Directory Structure
```text
.
├── src/                    # Source code
│   ├── train.cpp    # Main training program
│   ├── kernels_cpu.cpp     # CPU Math kernels (Conv, ReLU, Pool...)
│   └── cifar10_dataset.cpp # Data loader implementation
├── include/                # Header files
│   ├── kernels.h           # Function declarations
│   └── cifar10_dataset.h   # Data loader header
├── data/                   # Dataset directory
│   └── cifar-10-batches-bin/  # Extracted CIFAR-10 binary files
├── build/                  # Compiled executables (output folder)
├── weights/                # Saved binary weights (output folder)
└── README.md
````

### 2. How to Run

#### Step 1: Preparation

Ensure you have the dataset in `data/` and create output directories:

```bash
mkdir -p build
mkdir -p weights
```

#### Step 2: Compile

Run this command from the project root to compile with optimizations:

```bash
g++ -std=c++17 -O3 src/train.cpp src/cifar10_dataset.cpp src/kernels_cpu.cpp -I include -o build/cpu_train
```

#### Step 3: Execute

Navigate to the build directory and run the program:

```bash
cd build
./cpu_train
```
