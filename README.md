# Multi-modal Motor Fault Diagnosis and Reporting Assistant

![Build Status](https://img.shields.io/badge/build-passing-brightgreen)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Framework](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![License](https://img.shields.io/badge/License-MIT-green)

## 📖 Abstract
This project presents a novel approach to **Industrial Predictive Maintenance** by combining Signal Processing with Natural Language Generation (NLG). While traditional methods only classify faults (e.g., "Fault Detected"), this system generates **human-readable diagnostic reports** from raw vibration signals.

The system leverages a hybrid **Encoder-Decoder architecture**, utilizing **Audio Spectrogram Transformers (AST)** for feature extraction from vibration data and **GPT-2** for generating context-aware textual descriptions (Signal Captioning).

---

## 🏗️ Methodology & Architecture

The proposed model follows a multi-modal learning pipeline designed to bridge the semantic gap between time-series sensor data and natural language.

### 1. Signal Encoder (The "Eye")
* **Model:** `MIT/ast-finetuned-audioset-10-10-0.4593`
* **Process:** Raw 1D vibration signals are converted into 2D **Spectrograms** using Short-Time Fourier Transform (STFT). These images are fed into the AST, which treats them as visual patches to extract dense semantic embeddings (`e1`).

### 2. The Bridge (Projection Layer)
* A learnable linear mapping layer translates the AST output dimension (768-dim) to the GPT-2 input embedding space, enabling seamless information flow between modalities.

### 3. Text Decoder (The "Voice")
* **Model:** `gpt2` (Pre-trained Transformer Decoder)
* **Process:** The decoder receives the mapped signal embeddings and generates the diagnostic report token-by-token using auto-regressive decoding.

```mermaid
graph LR
    subgraph Signal Processing
    A[Raw Vibration Data] -->|STFT| B(Spectrogram)
    end
    
    subgraph Neural Network
    B -->|Input| C{AST Encoder}
    C -->|Feature Vector| D[Projection Bridge]
    D -->|Semantic Context| E{GPT-2 Decoder}
    end
    
    E --> F[Generated Report: 'Inner race fault detected...']