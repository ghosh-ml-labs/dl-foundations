# This is under-construction

# dl-foundations
Core deep learning architectures implemented with mathematical clarity and production discipline.

> Scope: **CNNs → RNN/LSTM → Attention/Transformer Encoder → Optimisers**  
> Goals: clean implementations (NumPy & PyTorch), unit tests, reproducible training, and concise, didactic notebooks.

---

## 🔎 Why this repo
- Demonstrate **first-principles understanding** (forward/backprop, shapes, gradients).
- Provide **reference implementations** with unit tests and training scripts.
- Show **industry readiness**: CI, packaging, experiment tracking, and model cards.

---

## 🗂️ Repository structure



# dl-foundations
Core deep learning architectures implemented with mathematical clarity and production discipline.

> Scope: **CNNs → RNN/LSTM → Attention/Transformer Encoder → Optimisers**  
> Goals: clean implementations (NumPy & PyTorch), unit tests, reproducible training, and concise, didactic notebooks.

---

## 🔎 Why this repo
- Demonstrate **first-principles understanding** (forward/backprop, shapes, gradients).
- Provide **reference implementations** with unit tests and training scripts.
- Show **industry readiness**: CI, packaging, experiment tracking, and model cards.

---

## 🗂️ Repository structure
```text
dl-foundations/
 ├─ README.md
 ├─ LICENSE
 ├─ pyproject.toml                # package metadata (optional)
 ├─ requirements.txt              # pinned minimal deps
 ├─ src/dlfoundations/
 │   ├─ models/
 │   │   ├─ cnn.py                # Conv/Pool/FC blocks
 │   │   ├─ rnn.py                # Vanilla RNN, GRU/LSTM
 │   │   ├─ attention.py          # Bahdanau/Luong, scaled dot-prod
 │   │   └─ transformer.py        # Encoder block (MHSA + FFN)
 │   ├─ optimisers/
 │   │   ├─ sgd.py                # SGD, Momentum, Nesterov
 │   │   ├─ rmsprop.py
 │   │   └─ adam.py
 │   ├─ data/
 │   │   ├─ vision.py             # MNIST/CIFAR loaders
 │   │   └─ text.py               # IMDB/AGNews loaders
 │   ├─ train/
 │   │   ├─ train_cnn.py
 │   │   ├─ train_rnn.py
 │   │   └─ train_transformer.py
 │   └─ utils/
 │       ├─ metrics.py            # accuracy, F1, perplexity
 │       ├─ viz.py                # loss curves, filters, attn maps
 │       └─ seed.py
 ├─ notebooks/
 │   ├─ 01_cnn_from_scratch.ipynb
 │   ├─ 02_rnn_lstm_basics.ipynb
 │   ├─ 03_attention_and_transformer.ipynb
 │   └─ 04_optimisers_explained.ipynb
 ├─ tests/                        # pytest unit tests
 │   ├─ test_cnn.py
 │   ├─ test_rnn.py
 │   ├─ test_attention.py
 │   └─ test_optimisers.py
 ├─ configs/                      # YAML configs (Hydra/vanilla)
 │   ├─ cnn_mnist.yaml
 │   ├─ rnn_imdb.yaml
 │   └─ transformer_tiny.yaml
 ├─ assets/                       # diagrams, sample outputs
 └─ .github/workflows/ci.yaml     # lint, tests, (optional) build
```



---

## 🧭 Roadmap (milestones & acceptance criteria)

### 1) CNN Foundations (Vision) — **Milestone M1**
- **Deliverables**
  - NumPy forward pass (Conv2D, ReLU, MaxPool, Linear) + gradient checks on tiny inputs.
  - PyTorch CNN (simple LeNet/ResNet-mini) with training on **MNIST**; optional **CIFAR-10**.
  - Visualisations: learned filters, activation maps, confusion matrix.
- **Targets**
  - MNIST accuracy ≥ **99.0%** (5 epochs, baseline).
  - CIFAR-10 accuracy ≥ **70%** (light model, 50 epochs) *(stretch)*.
- **Artifacts**
  - `notebooks/01_cnn_from_scratch.ipynb`
  - `src/dlfoundations/models/cnn.py`, `train/train_cnn.py`
  - Plots in `assets/`

### 2) RNN / LSTM (Sequence) — **Milestone M2**
- **Deliverables**
  - Vanilla RNN step-by-step (NumPy) + gradient checks.
  - PyTorch **LSTM** for **IMDB** sentiment (or AG News with simple tokenizer).
  - Visuals: loss/accuracy curves, error analysis by sequence length.
- **Targets**
  - IMDB test accuracy ≥ **86%** (baseline).
- **Artifacts**
  - `notebooks/02_rnn_lstm_basics.ipynb`
  - `src/dlfoundations/models/rnn.py`, `train/train_rnn.py`

### 3) Attention & Transformer Encoder — **Milestone M3**
- **Deliverables**
  - Scaled dot-product attention from first principles.
  - **Transformer Encoder block** (MHSA + FFN + LayerNorm + residuals).
  - Tiny text classification with encoder-only model.
  - Visuals: attention heatmaps, layer output norms.
- **Targets**
  - AG News accuracy ≥ **90%** (tiny encoder).
- **Artifacts**
  - `notebooks/03_attention_and_transformer.ipynb`
  - `src/dlfoundations/models/attention.py`, `models/transformer.py`, `train/train_transformer.py`

### 4) Optimisers (SGD/Momentum/RMSProp/Adam) — **Milestone M4**
- **Deliverables**
  - Implement update rules from scratch; unit tests against PyTorch reference for a toy loss.
  - Comparative training curves on the **same CNN** and **same LSTM**.
  - Discussion: stability, learning-rate schedules, weight decay vs AdamW.
- **Artifacts**
  - `notebooks/04_optimisers_explained.ipynb`
  - `src/dlfoundations/optimisers/*.py`, `tests/test_optimisers.py`

---

## ⚙️ Setup

### Requirements
- Python ≥ 3.10  
- `torch`, `torchvision`, `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `tqdm`, `pytest`

```bash
# (optional) create env
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install -r requirements.txt
# or
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install numpy pandas scikit-learn matplotlib tqdm pytest
