# Classical Composer Classification using Multi-Branch CNN

## Project Overview

This project implements a deep learning system for classifying classical music compositions by composer using a multi-branch Convolutional Neural Network (CNN) architecture. The system processes MIDI files and extracts both temporal patterns (piano-roll) and harmonic features (pitch class distribution) to achieve accurate composer identification.

### Key Features
- **Multi-modal Input Processing**: Combines piano-roll sequences (128×1000×1) and pitch class distributions (12-dimensional)
- **Late Fusion Architecture**: Processes different feature types in separate branches before combining
- **Class Imbalance Handling**: Uses class weights to address dataset imbalance
- **High Performance**: Achieves 80.67% overall accuracy on validation set

### Dataset
- **4 Composers**: Bach (1024 files), Beethoven (212 files), Chopin (136 files), Mozart (256 files)
- **Total Samples**: 1,628 MIDI files
- **Data Split**: 80% training, 20% validation

### Input Data Format
- **Piano-Roll**: 2D representation of MIDI data (128 notes × 1000 time steps)
- **Pitch Class Distribution**: 12-dimensional vector representing harmonic content

## Model Architecture

```
┌─────────────────┐    ┌─────────────────┐
│ Piano-Roll      │    │ Pitch Class     │
│ Input           │    │ Input           │
│ (128×1000×1)    │    │ (12,)           │
└─────────┬───────┘    └─────────┬───────┘
          │                      │
    ┌─────▼─────┐           ┌────▼────┐
    │ Conv2D    │           │ Dense   │
    │ (32→64→128)│           │ (64→32→16) │
    │ + BN + ReLU│           │ + BN + ReLU│
    │ + MaxPool  │           │ + Dropout │
    │ + Dropout  │           │           │
    └─────┬─────┘           └────┬────┘
          │                      │
    ┌─────▼─────┐                │
    │Global Avg │                │
    │   Pool    │                │
    └─────┬─────┘                │
          │                      │
    ┌─────▼─────┐                │
    │Dense (256)│                │
    │→ Dense(128)│                │
    └─────┬─────┘                │
          │                      │
          └─────────┬────────────┘
                    │
              ┌─────▼─────┐
              │Concatenate│
              │(128 + 16) │
              └─────┬─────┘
                    │
              ┌─────▼─────┐
              │Dense (64) │
              │+ Dropout  │
              └─────┬─────┘
                    │
              ┌─────▼─────┐
              │ Softmax   │
              │(4 classes)│
              └───────────┘
```

### Performance Metrics
- **Overall Accuracy**: 80.67%
- **Per-Class Performance**:
  - Bach: 92.68% recall, 89.62% precision
  - Beethoven: 34.88% recall, 68.18% precision  
  - Chopin: 81.48% recall, 75.86% precision
  - Mozart: 70.59% recall, 57.14% precision

**Note**: Beethoven's lower recall suggests the model struggles to identify Beethoven pieces correctly, likely due to the smaller dataset size (212 files vs. 1024 for Bach).

---

## Setup Instructions

### Option 1: CPU Setup 

#### Using Conda:
1. **Create and activate a new conda environment:**
   ```bash
   conda create -n composer-classifier python=3.8
   conda activate composer-classifier
   ```

2. **Install requirements:**
   ```bash
   pip install -r requirements.txt
   ```

#### Using Python venv:
1. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   
   # On Windows:
   venv\Scripts\activate
   
   # On macOS/Linux:
   source venv/bin/activate
   ```

2. **Install requirements:**
   ```bash
   pip install -r requirements.txt
   ```

### Option 2: GPU Setup (CUDA) - For faster training

⚠️ **Important**: Only use this option if you have an NVIDIA GPU and want faster training. The CPU setup works fine for running the project.

1. **Run the CUDA setup script:**
   ```bash
   # On Windows PowerShell:
   .\cuda.ps1
   ```

2. **The script will:**
   - Create a new conda environment called `tf-gpu`
   - Install CUDA toolkit 11.2 and cuDNN 8.1
   - Install TensorFlow 2.7.0 with GPU support
   - Set up proper environment variables
   - Verify the installation

3. **After running the script, activate the GPU environment:**
   ```bash
   conda activate tf-gpu
   ```

4. **Install additional requirements:**
   ```bash
   pip install -r requirements.txt
   ```

## Running the Project

1. **Activate your chosen environment:**
   ```bash
   # For CPU setup:
   conda activate composer-classifier
   # OR
   source venv/bin/activate  # (Linux/Mac) or venv\Scripts\activate (Windows)
   
   # For GPU setup:
   conda activate tf-gpu
   ```

2. **Launch Jupyter Notebook:**
   ```bash
   jupyter notebook
   ```

3. **Open and run `Team_project.ipynb`**

## Project Files

- `Team_project.ipynb` - Main Jupyter notebook containing the complete project implementation
- `requirements.txt` - Python dependencies for the project
- `cuda.ps1` - PowerShell script for CUDA/GPU setup (Windows only)
- `midi_subset/` - Directory containing the MIDI dataset organized by composer
- `pianorolls/` - Generated piano-roll representations
- `pitch/` - Generated pitch class distributions

