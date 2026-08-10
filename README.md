# CNEE — Fine-tuning VLM Models for Case File Evaluation

Supervised training pipeline for Vision-Language Models (VLM) with QLoRA using Unsloth, aimed at the automatic evaluation of electrical outage case files.

---

## Supported Models

| Model | Parameters | VRAM at 4-bit | RTX 3070 Ti (8 GB) | Cloud / DGX Spark |
|---|---|---|---|---|
| `Qwen/Qwen3-VL-2B-Instruct` | 2B | ~3–4 GB | ✅ Comfortable | ✅ |
| `Qwen/Qwen3-VL-4B-Instruct` | 4B | ~5–6 GB | ⚠️ Very tight | ✅ |
| `Qwen/Qwen3-VL-8B-Instruct` | 8B | ~10–12 GB | ❌ | ✅ DGX Spark only |
| `Qwen/Qwen3.5-2B-Instruct` | 2B | ~3.5 GB | ✅ Comfortable | ✅ |
| `Qwen/Qwen3.5-4B-Instruct` | 4B | ~5.5 GB | ✅ Viable | ✅ |
| `Qwen/Qwen3.5-9B` | 9B | ~18 GB (BF16) | ❌ | ✅ DGX Spark only |

> **Note:** Qwen3.5 outperforms Qwen3-VL on document comprehension benchmarks (OCR, long documents). Recommended as the base model for new experiments.
>
> **DGX Spark:** With 130.7 GB of unified memory, the 8B/9B models can be trained with full LoRA (BF16, no quantization required) and with complete case files — no page truncation needed.

### Selected model — Qwen3.5-9B (no thinking)

**`Qwen3.5-9B` is the current base model for this project.** Both a thinking-enabled variant (`train_with_val_loss_35_thinking.py`, `ENABLE_THINKING=True`) and a non-thinking variant (`train_with_val_loss_35.py`, `ENABLE_THINKING=False`) were trained and evaluated. The **non-thinking variant was selected** for production inference (`inference_qwen35.py`, checkpoint `output/qwen35_9b_v9/checkpoint-72`): it produces the JSON decision directly, without a `<think>...</think>` reasoning trace, which keeps inference latency and token usage predictable — thinking traces increased output length substantially without a proportional accuracy gain for this task. See [Reference Metrics](#reference-metrics) below for the comparison.

---

## System Requirements

### Minimum Hardware (training)

- **NVIDIA GPU** with at least **8 GB of VRAM** (recommended: 16 GB+)
- **System RAM:** 16 GB minimum (32 GB recommended)
- **Storage:** 50 GB free (model + dataset + checkpoints)

### Base Software

- **OS:** Ubuntu 22.04 LTS
- **NVIDIA Driver:** >= 580.x
- **CUDA Toolkit:** 12.8 (compatible with PyTorch 2.11+)
- **Python:** 3.11

### NVIDIA DGX Spark

| Component | Specification |
|---|---|
| Superchip | NVIDIA GB10 Grace Blackwell |
| Unified Memory | 130.7 GB LPDDR5X |
| AI Performance | 1 PFLOP FP4 · 1000 TOPS |
| CPU | 20 ARM cores (Grace) |
| Storage | 4 TB NVMe SSD |
| OS | DGX OS (Ubuntu 24.04 LTS) |
| CUDA | 13.0 (cu128 PyTorch wheel) |
| Python | 3.12 (system default) |

---

## Installation

### Ubuntu 22.04 LTS (RTX 3070 Ti / Cloud)

#### 1. Verify GPU and CUDA

```bash
nvidia-smi
nvcc --version
```

#### 2. Install system dependencies

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3.11 python3.11-venv python3.11-dev \
  build-essential git curl wget cmake ninja-build \
  libssl-dev libffi-dev git-lfs
git lfs install
```

#### 3. Create virtual environment

```bash
mkdir ~/CNEE && cd ~/CNEE
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel
```

#### 4. Install PyTorch with CUDA 12.8

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

Verify:

```bash
python3 -c "
import torch
print('PyTorch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
print('GPU:', torch.cuda.get_device_name(0))
print('VRAM:', round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1), 'GB')
"
```

#### 5. Install Unsloth

```bash
pip install "unsloth @ git+https://github.com/unslothai/unsloth.git"
pip install unsloth-zoo
```

Verify:

```bash
python3 -c "import unsloth; print('Unsloth:', unsloth.__version__)"
```

#### 6. Install training dependencies

```bash
pip install transformers accelerate peft bitsandbytes trl datasets
pip install pillow qwen-vl-utils matplotlib
```

#### 7. Install Ollama (for inference)

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama --version
```

---

### NVIDIA DGX Spark (Ubuntu 24.04 LTS)

The DGX Spark runs Ubuntu 24.04, which ships Python 3.12 by default. Python 3.11 is not available in the standard repositories and is not needed. PyTorch is installed using the `cu128` wheel, which runs correctly on CUDA 13.0 drivers due to backward compatibility.

#### 1. Verify GPU and CUDA

```bash
nvcc --version   # Expected: CUDA 13.0
tegrastats       # Monitor unified memory (replaces nvidia-smi for memory usage)
```

> `nvidia-smi` runs on the Spark but does not report VRAM the same way as discrete GPUs due to the unified memory architecture. Use `tegrastats` for memory monitoring, or check from Python (see step 4).

#### 2. Install system dependencies

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3-dev python3-venv python3-pip \
  build-essential git curl wget cmake ninja-build \
  libssl-dev libffi-dev git-lfs \
  libgl1 libglib2.0-0
git lfs install
```

> On Ubuntu 24.04, `libgl1-mesa-glx` has been renamed to `libgl1`.

#### 3. Create virtual environment (Python 3.12)

```bash
mkdir ~/CNEE && cd ~/CNEE
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel
```

Add to `.bashrc` to activate automatically on every session:

```bash
echo "source ~/CNEE/.venv/bin/activate" >> ~/.bashrc
```

#### 4. Install PyTorch with CUDA 12.8 (compatible with CUDA 13.0)

```bash
pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu128
```

Verify GPU access:

```bash
python3 -c "
import torch
print('PyTorch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
print('GPU:', torch.cuda.get_device_name(0))
print('VRAM:', round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1), 'GB')
"
```

Expected output on DGX Spark:

```
PyTorch: 2.11.0+cu128
CUDA available: True
GPU: NVIDIA GB10
VRAM: 130.7 GB
```

#### 5. Install Unsloth

```bash
pip install "unsloth @ git+https://github.com/unslothai/unsloth.git"
pip install unsloth-zoo
```

Verify:

```bash
python3 -c "import unsloth; print('Unsloth:', unsloth.__version__)"
```

#### 6. Install training dependencies

```bash
pip install transformers accelerate peft bitsandbytes trl datasets
pip install pillow qwen-vl-utils matplotlib
```

#### DGX Spark — Verified package versions

| Package | Version |
|---|---|
| Python | 3.12 |
| PyTorch | 2.11.0+cu128 |
| CUDA Toolkit | 13.0 (cu128 wheel) |
| OS | DGX OS — Ubuntu 24.04 LTS |

#### DGX Spark — Training differences vs RTX 3070 Ti

| Parameter | RTX 3070 Ti | DGX Spark |
|---|---|---|
| `load_in_4bit` | `True` (QLoRA required) | `False` (LoRA BF16, full precision) |
| `MAX_IMAGENES` | 3–5 (truncation risk) | Full case file, no limit |
| `max_seq_length` | 8192 | 32768+ |
| LoRA rank | 16–32 | 64–128 |
| Models supported | 2B, 4B | 2B, 4B, 8B |

---

## Verified Package Versions

### Ubuntu 22.04 — RTX 3070 Ti Laptop (8 GB VRAM)

| Package | Version |
|---|---|
| Python | 3.11 |
| PyTorch | 2.11.0+cu128 |
| Unsloth | 2026.3.11 |
| Transformers | 5.2.0 |
| Datasets | 4.3.0 |
| TRL | — (latest compatible with Unsloth) |
| PEFT | — (installed with Unsloth) |
| bitsandbytes | — (installed with Unsloth) |
| CUDA Toolkit | 12.8 |
| NVIDIA Driver | 580.126.09 |

### Ubuntu 24.04 — NVIDIA DGX Spark (130.7 GB unified memory)

| Package | Version |
|---|---|
| Python | 3.12 |
| PyTorch | 2.11.0+cu128 |
| Unsloth | 2026.3.11 |
| Transformers | 5.2.0 |
| Datasets | 4.3.0 |
| CUDA Toolkit | 13.0 (cu128 wheel) |
| GPU | NVIDIA GB10 Blackwell |

---

## Model Download

### Option A — Git LFS (recommended)

```bash
cd ~/CNEE/models

# Qwen3-VL 2B
git clone https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct

# Qwen3-VL 4B
git clone https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct

# Qwen3.5 2B
git clone https://huggingface.co/Qwen/Qwen3.5-2B-Instruct

# Qwen3.5 4B
git clone https://huggingface.co/Qwen/Qwen3.5-4B-Instruct

# Qwen3.5 9B (selected base model)
git clone https://huggingface.co/Qwen/Qwen3.5-9B
```

### Option B — Google Drive (direct download)

The base models **Qwen3-VL-2B-Instruct** and **Qwen3-VL-4B-Instruct**, along with the project dataset, are available on Google Drive:

**[Download from Google Drive](https://drive.google.com/drive/folders/16I3sG4I4UbXSvw9HVDBJjrQbyxAKmnBc?usp=drive_link)**

Download the corresponding folders and place them at:

```
~/CNEE/models/Qwen3-VL-2B-Instruct/
~/CNEE/models/Qwen3-VL-4B-Instruct/
~/CNEE/dataset/
```

### Option C — Python (if git lfs fails)

```bash
source ~/CNEE/.venv/bin/activate
pip install huggingface-hub

python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='Qwen/Qwen3.5-2B-Instruct',
    local_dir='./models/Qwen3.5-2B-Instruct',
    local_dir_use_symlinks=False
)
"
```

---

## Project Structure

```
~/CNEE/
├── .venv/                          # Python virtual environment
├── models/
│   ├── Qwen3-VL-2B-Instruct/       # Qwen3-VL 2B base model
│   ├── Qwen3-VL-4B-Instruct/       # Qwen3-VL 4B base model
│   ├── Qwen3.5-2B-Instruct/        # Qwen3.5 2B base model
│   └── Qwen3.5-4B-Instruct/        # Qwen3.5 4B base model
├── dataset/
│   ├── dataset_info.json           # Registry for LLaMA Factory
│   ├── dataset_FINAL_100casos.json # Main dataset (100 cases)
│   └── DEOCSA_2023S1/
│       ├── approved/               # Images of approved cases
│       │   └── caso_XXX/
│       │       ├── page_001.jpg
│       │       └── ...
│       └── rejected/               # Images of rejected cases
│           └── caso_XXX/
│               └── ...
├── output/
│   └── qwen3vl_2b_cnee/
│       ├── final/                  # Fine-tuned model (safetensors)
│       ├── merged/                 # Merged model (LoRA + base)
│       └── merged_gguf/            # Model converted to GGUF
│           ├── merged.Q4_K_M.gguf  # For inference with llama.cpp
│           └── merged.BF16-mmproj.gguf
├── train_unsloth.py                # Basic training script
├── train_with_val_loss.py          # Script with Train/Val loss + charts
├── model_evaluation.py             # Post-training evaluation script
└── finetuning_config.yaml          # Config for LLaMA Factory (reference)
```

---

## Available Scripts

### Training with metrics

```bash
cd ~/CNEE
source .venv/bin/activate

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python3 train_con_val_loss.py
```

Configurable parameters at the top of the script:

```python
MAX_IMAGENES = 3        # Pages per case (3-5 depending on available VRAM)
NUM_EPOCHS   = 7        # Number of epochs
OUTPUT_DIR   = "..."    # Output directory
MODEL_PATH   = "..."    # Path to base model
```

> **DGX Spark:** Set `MAX_IMAGENES` to the full number of pages in the case file and `load_in_4bit=False` to use LoRA BF16 instead of QLoRA.

### Model evaluation

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python3 evaluar_modelo.py
```

Generates:
- VQA Accuracy and Exact Match table per case
- `metricas_validacion.json` with detailed results
- `evaluacion_modelo.png` with metric charts and confusion matrix

### Conversion to GGUF (for inference)

```bash
python3 -c "
from unsloth import FastVisionModel

# Merge LoRA with base model
model, tokenizer = FastVisionModel.from_pretrained(
    model_name='output/qwen3vl_2b_cnee/final',
    load_in_4bit=True,
)
model.save_pretrained_merged(
    'output/qwen3vl_2b_cnee/merged',
    tokenizer,
    save_method='merged_16bit',
)

# Convert to GGUF
model.save_pretrained_gguf(
    'output/qwen3vl_2b_cnee/merged_gguf',
    tokenizer,
    quantization_method='q4_k_m',
)
"
```

### Inference with llama.cpp (Unsloth)

Unsloth ships its own `llama-mtmd-cli` binary that supports multimodal models with a separate `mmproj` file. This is the recommended inference path for GGUF models.

```bash
# Locate the binary (installed by Unsloth)
find ~/.unsloth -name "llama-mtmd-cli" 2>/dev/null
```

**Key flags:**

| Flag | Description |
|---|---|
| `-m` | Path to the quantized GGUF model |
| `--mmproj` | Path to the multimodal projector (vision encoder) |
| `--image` | Image file to send (repeat for multiple pages) |
| `-p` | Prompt text |
| `-n` | Max tokens to generate |
| `--gpu-layers` | Number of layers to offload to GPU (29 for RTX 3070 Ti) |
| `-c` | Context window size |

#### Single image — full normative analysis

```bash
~/.unsloth/llama.cpp/llama-mtmd-cli \
    -m ~/CNEE/output/qwen3vl_2b_cnee/merged_gguf/merged.Q4_K_M.gguf \
    --mmproj ~/CNEE/output/qwen3vl_2b_cnee/merged_gguf/merged.BF16-mmproj.gguf \
    --image ~/CNEE/dataset/DEOCSA_2023S1/approved/caso_022/page_001.jpg \
    -p "Analiza este expediente y determina si es APROBADO o RECHAZADO segun los 7 criterios normativos." \
    -n 500 \
    --gpu-layers 29 \
    -c 1024
```

#### Two images — concise decision + justification

```bash
~/.unsloth/llama.cpp/llama-mtmd-cli \
    -m ~/CNEE/output/qwen3vl_2b_cnee/merged_gguf/merged.Q4_K_M.gguf \
    --mmproj ~/CNEE/output/qwen3vl_2b_cnee/merged_gguf/merged.BF16-mmproj.gguf \
    --image ~/CNEE/dataset/DEOCSA_2023S1/approved/caso_022/page_001.jpg \
    --image ~/CNEE/dataset/DEOCSA_2023S1/approved/caso_022/page_002.jpg \
    -p "Analiza estas paginas del expediente de fuerza mayor. Responde unicamente con: 1) APROBADO o RECHAZADO, 2) El criterio oficial que justifica la decision." \
    -n 300 \
    --gpu-layers 29 \
    -c 2048
```

#### Three images — approved case (caso_029)

```bash
~/.unsloth/llama.cpp/llama-mtmd-cli \
    -m ~/CNEE/output/qwen3vl_2b_cnee/merged_gguf/merged.Q4_K_M.gguf \
    --mmproj ~/CNEE/output/qwen3vl_2b_cnee/merged_gguf/merged.BF16-mmproj.gguf \
    --image ~/CNEE/dataset/DEOCSA_2023S1/approved/caso_029/page_001.jpg \
    --image ~/CNEE/dataset/DEOCSA_2023S1/approved/caso_029/page_002.jpg \
    --image ~/CNEE/dataset/DEOCSA_2023S1/approved/caso_029/page_003.jpg \
    -p "Analiza estas paginas del expediente de fuerza mayor. Responde unicamente con: 1) APROBADO o RECHAZADO, 2) El criterio oficial que justifica la decision." \
    -n 300 \
    --gpu-layers 29 \
    -c 2048
```

#### Four images — rejected case (caso_240)

```bash
~/.unsloth/llama.cpp/llama-mtmd-cli \
    -m ~/CNEE/output/qwen3vl_2b_cnee/merged_gguf/merged.Q4_K_M.gguf \
    --mmproj ~/CNEE/output/qwen3vl_2b_cnee/merged_gguf/merged.BF16-mmproj.gguf \
    --image ~/CNEE/dataset/DEOCSA_2023S1/rejected/caso_240/page_003.jpg \
    --image ~/CNEE/dataset/DEOCSA_2023S1/rejected/caso_240/page_004.jpg \
    --image ~/CNEE/dataset/DEOCSA_2023S1/rejected/caso_240/page_005.jpg \
    --image ~/CNEE/dataset/DEOCSA_2023S1/rejected/caso_240/page_006.jpg \
    -p "Analiza estas paginas del expediente de fuerza mayor. Responde unicamente con: 1) APROBADO o RECHAZADO, 2) El criterio oficial que justifica la decision." \
    -n 300 \
    --gpu-layers 29 \
    -c 2048
```

> **Note:** Each additional image increases VRAM and context usage. For 4+ images use `-c 2048` or higher. Adjust `--gpu-layers` down if you get out-of-memory errors.

---

## Compatibility Notes

### Ollama and VLM models

Ollama currently **does not support** multimodal models that require a separate `mmproj` file (Qwen3-VL and Qwen3.5). Use `llama-mtmd-cli` from llama.cpp as an alternative until Ollama adds support.

### Recommended environment variables

```bash
# Always activate before training
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export DISABLE_VERSION_CHECK=1
export TORCHINDUCTOR_COMPILE_THREADS=1
```

Or add permanently to `.bashrc`:

```bash
echo 'export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True' >> ~/.bashrc
echo 'export DISABLE_VERSION_CHECK=1' >> ~/.bashrc
source ~/.bashrc
```

### Known Issues

| Error | Cause | Solution |
|---|---|---|
| `CUDA out of memory` | Insufficient VRAM | Reduce `MAX_IMAGENES` to 3, enable `gradient_checkpointing` |
| `Mismatch in image token count` | Context truncation | Increase `max_seq_length` or reduce images |
| `datasets version incompatible` | Version conflict | `pip install "datasets>=3.4.1,<4.4.0,!=4.0.*,!=4.1.0"` |
| `torchao incompatible` | Non-critical warning | Ignore, does not affect training |
| `generate-parameter-library-py` | ROS package without dependencies | `pip install jinja2 typeguard` (optional) |
| `Unable to locate package python3.11` | Ubuntu 24.04 ships with Python 3.12 | Use `python3 -m venv` instead of `python3.11 -m venv` |
| `Package 'libgl1-mesa-glx' not found` | Renamed in Ubuntu 24.04 | Replace with `libgl1` |
| No `cu130` PyTorch wheel | PyTorch does not yet publish cu130 builds | Use `cu128` — fully compatible with CUDA 13.0 drivers |

---

## Reference Metrics

Results obtained with Qwen3-VL-2B, 5 epochs, 3 images/case, RTX 3070 Ti Laptop:

| Metric | Description |
|---|---|
| VQA Accuracy | % of correct APPROVED/REJECTED decisions |
| Exact Match | % of responses identical to ground truth |
| Precision | TP / (TP + FP) for APPROVED class |
| Recall | TP / (TP + FN) for APPROVED class |
| F1-Score | Harmonic mean of Precision and Recall |

Results obtained with Qwen3-VL-4B, 25 epochs, 8 images/case, AWS g5.2xlarge (A10G 24 GB):

| Metric | Value |
|---|---|
| VQA Accuracy | 80.0% |
| Precision | 71.4% |
| Recall | 100.0% |
| F1-Score | 83.3% |
| Best checkpoint | Epoch 21 (checkpoint-126) |
| Training time | ~4 hours |
| Inference time | ~59 s/case (4096 tokens) |

Results obtained with **Qwen3.5-9B, no thinking** (selected model), checkpoint `qwen35_9b_v9/checkpoint-72`, held-out test set (100 cases), DGX Spark:

| Metric | Model only | With escalation gate (threshold 0.75) |
|---|---|---|
| VQA Accuracy | 77.0% | — |
| Precision | 72.1% | 72.7% |
| Recall | 88.0% | 88.9% |
| F1-Score | 79.3% | 80.0% |
| Escalation rate | — | 18.0% |
| Avg. inference time | ~106 s/case (8192 max tokens) | — |

> The escalation gate routes low-confidence predictions to manual review instead of forcing an automated decision; this raises precision/recall on the auto-decided subset at the cost of a ~18% escalation rate. Full breakdown (approved-only, rejected-only splits, train set) is in `output/inferencia_qwen35_9b_v11/resultados_inferencia.json`.
>
> The thinking-enabled variant (`qwen35_9b_v9_thinking`) was evaluated under the same conditions but was not selected: reasoning traces increased tokens/latency per case without a proportional accuracy improvement, so the non-thinking variant was chosen for production.

Results obtained with **Qwen3.5-9B, no thinking**, checkpoint `qwen35_9b_v9/checkpoint-156` (inference run v12), same held-out test set (100 cases), DGX Spark:

| Metric | Model only | With escalation gate (threshold 0.75) |
|---|---|---|
| VQA Accuracy | 86.0% | — |
| Precision | 89.1% | 92.1% |
| Recall | 83.7% | 92.1% |
| F1-Score | 86.3% | 92.1% |
| Escalation rate | — | 34.0% |
| Avg. inference time | ~107 s/case (8192 max tokens) | — |

> Same evaluation setup as the `checkpoint-72` run above (`output/inferencia_qwen35_9b_v12/resultados_inferencia.json`). `checkpoint-156` scores higher on every metric but escalates more cases to manual review (34% vs. 18%). By validation loss during training, the best checkpoint was actually `checkpoint-108` (`trainer_state.json`, loss 0.0385) — `72` and `156` bracket it. Production currently points at `checkpoint-72` (see `inference_qwen35.py`); this `checkpoint-156` result suggests it's worth re-checking whether `108` or `156` should be the production checkpoint instead.

---

## Team

Universidad Galileo — IRE and LSDC-FISICC Laboratories
Project: Implementation of AI tools for case file evaluation in the energy sector
