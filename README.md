# DTFformer
Dual-Granularity Time-Frequency Interaction Transformer for Time Series Forecasting
## 📖 Overview

Time series forecasting often struggles to balance time-domain local dynamics and frequency-domain global patterns. **DTFformer** addresses this by introducing a **dual-granularity interaction mechanism**.

Instead of simple concatenation, this model progressively fuses information from both domains:
* **Micro-Level:** Utilizes a mutual guidance attention mechanism where time and frequency features serve as contexts for each other.
* **Macro-Level:** Employs inter-layer gating networks to dynamically modulate the interaction strength between branches.
* **Noise Reduction:** Features an **AmpT-Filter** (Adaptive Amplitude Transformation Filter) to enhance the signal-to-noise ratio by adaptively weighting frequency components.

## ✨ Key Features

* **Dual-Branch Architecture:** Processes temporal and frequency features in parallel with deep interaction.
* **Adaptive Filtering:** The AmpT-Filter automatically amplifies key frequency components and suppresses noise, making it robust against non-stationary data.
* **High Performance:** Achieves state-of-the-art performance on standard benchmarks including **ETTh, ETTm, Weather, and Wind** datasets.

## 🚀 Getting Started

### 1. Prerequisites

* Python 3.8+
* PyTorch >= 1.8

