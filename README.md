# Classical Composer Classification using Multi-Branch CRNN

## Project Overview

This project implements a deep learning system for classifying classical music compositions by composer using a multi-branch Convolutional Recurrent Neural Network (CRNN) architecture. The system processes MIDI files using a chunked data pipeline and combines CNN feature extraction with LSTM temporal modeling to achieve accurate composer identification.

### Key Features
- **CRNN Architecture**: Combines CNN layers for spatial feature extraction with LSTM layers for temporal modeling
- **Chunked Data Pipeline**: Uses overlapping chunks for data augmentation and memory optimization
- **Multi-modal Input Processing**: Combines piano-roll sequences (128×1000×1) and pitch class distributions (12-dimensional)
- **Late Fusion Architecture**: Processes different feature types in separate branches before combining
- **Memory Optimization**: Balanced sampling strategy reduces memory usage while maintaining data integrity
- **Data Leakage Prevention**: Ensures chunks from the same original file don't appear in both train/validation sets

### Dataset
- **4 Composers**: Bach, Beethoven, Chopin, Mozart
- **Original Files**: 1,628 MIDI files
- **Chunked Data**: ~15,000 chunks with overlapping windows for data augmentation
- **Memory Optimization**: Balanced sampling reduces Bach overrepresentation while preserving underrepresented composers
- **Data Split**: 80% training, 20% validation with no-leakage guarantee at file level

### Input Data Format
- **Piano-Roll**: 2D representation of MIDI data (128 notes × 1000 time steps) processed in overlapping chunks
- **Pitch Class Distribution**: 12-dimensional vector representing harmonic content per chunk
- **Chunking Strategy**: 10% overlap between consecutive chunks for data augmentation

## Model Architecture (CRNN)

```
┌─────────────────┐    ┌─────────────────┐
│ Piano-Roll      │    │ Pitch Class     │
│ Input           │    │ Input           │
│ (128×1000×1)    │    │ (12,)           │
└─────────┬───────┘    └─────────┬───────┘
          │                      │
    ┌─────▼─────┐           ┌────▼────┐
    │ CNN Layers│           │ Dense   │
    │ (32→64→128)│           │ (64→32→16) │
    │ + BN + ReLU│           │ + BN + ReLU│
    │ + MaxPool  │           │ + Dropout │
    │ + Dropout  │           │           │
    └─────┬─────┘           └────┬────┘
          │                      │
    ┌─────▼─────┐                │
    │ Reshape   │                │
    │ for LSTM  │                │
    └─────┬─────┘                │
          │                      │
    ┌─────▼─────┐                │
    │LSTM (128) │                │
    │ Temporal  │                │
    │ Modeling  │                │
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
              │Late Fusion│
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

### Architecture Details
- **CNN Branch**: Extracts spatial features from piano-roll data using convolutional layers
- **LSTM Layer**: Captures temporal dependencies and long-term patterns in musical sequences
- **Dense Branch**: Processes pitch class distributions for harmonic analysis
- **Late Fusion**: Combines features from both branches for final classification
- **Total Parameters**: ~700K trainable parameters optimized for musical pattern recognition

### Data Processing Pipeline

1. **MIDI Extraction**: Automated extraction and organization of MIDI files by composer
2. **Chunked Preprocessing**: 
   - Creates overlapping chunks (10% overlap) for data augmentation
   - Handles variable-length pieces without truncation
   - Generates multiple training samples per original file
3. **Feature Extraction**:
   - Piano-roll representation: 128 notes × 1000 time steps per chunk
   - Pitch class distributions: 12-dimensional harmonic features per chunk
4. **Memory Optimization**: Balanced sampling strategy reduces memory footprint
5. **Data Integrity**: File-level train/validation split prevents data leakage

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

3. **Open and run the appropriate notebook:**
   - `composer_classifier_optimized.ipynb` - Latest CRNN implementation with chunked data pipeline
   - `Team_project_cnn.ipynb` - CNN implementation
   - `Team_project_crnn.ipynb` - CRNN implementation

## Project Files

- `composer_classifier_optimized.ipynb` - Latest optimized CRNN implementation with chunked data pipeline
- `Team_project_cnn.ipynb` - CNN implementation
- `Team_project_crnn.ipynb` - CRNN implementation
- `requirements.txt` - Python dependencies for the project
- `cuda.ps1` - PowerShell script for CUDA/GPU setup (Windows only)
- `midi_subset/` - Directory containing the MIDI dataset organized by composer
- `pianorolls/` & `pianorolls2_0/` - Generated piano-roll representations (original and chunked)
- `pitch/` & `pitch2_0/` - Generated pitch class distributions (original and chunked)

