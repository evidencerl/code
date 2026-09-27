# Evidence-RL: Towards Evidence-intensive Visual Reasoning

Official PyTorch implementation of **Evidence-RL** and **Answer-CED** (NeurIPS 2026).

---

## 📌 Overview

Vision-Language Models (VLMs) frequently suffer from visual hallucinations and shortcut reasoning by relying on language priors or scene context rather than inspecting relevant visual evidence. 

**Evidence-RL** introduces **Counterfactual Evidence Disentanglement (CED)**, an outcome-conditioned causal intervention reward integrated with Group Relative Policy Optimization (GRPO). By evaluating counterfactual likelihood margins under localized feature interventions, Evidence-RL encourages the model to ground its predictions in visual evidence rather than language shortcuts.

<p align="center">
  <img src="https://img.shields.io/badge/Model-Evidence--RL--9B-blue" alt="Model">
  <img src="https://img.shields.io/badge/Framework-PyTorch%20%7C%20Transformers-orange" alt="Framework">
  <img src="https://img.shields.io/badge/License-Apache--2.0-green" alt="License">
</p>

---

## 🚀 Released Models

The post-trained checkpoint based on Qwen3.5-9B is available on Hugging Face:

| Model | Base Backbone | Method | Hugging Face Checkpoint |
| :--- | :--- | :--- | :--- |
| **Evidence-RL-9B** | Qwen3.5-9B | Answer-CED + Gated GRPO | [🤗 hhj-ai/Evidence-RL-9B](https://huggingface.co/hhj-ai/Evidence-RL-9B) |

---

## 🛠️ Environment Setup

### 1. Clone & Environment
```bash
git clone https://github.com/evidencerl/code.git
cd code

# Option A: Conda environment
conda env create -f environment.yml
conda activate evidence_rl

# Option B: Pip installation
pip install -r requirements.txt
```

### 2. (Optional) FlashAttention-2
For optimal training throughput:
```bash
pip install flash-attn --no-build-isolation
```

---

## 📦 Quickstart: Inference Demo

You can run inference using the Hugging Face `transformers` library directly:

```python
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
from PIL import Image

model_id = "hhj-ai/Evidence-RL-9B"

processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForImageTextToText.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)

# Load your image
image = Image.open("example.jpg").convert("RGB")
prompt = "Look at the image carefully and count the objects. Answer with just a number, without any additional text."

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt},
        ],
    }
]

text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = processor(text=[text], images=[image], return_tensors="pt").to(model.device)

with torch.inference_mode():
    outputs = model.generate(**inputs, max_new_tokens=256)

response = processor.batch_decode(outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
print(response)
```

Or run the bundled demo script:
```bash
python demo.py --image path/to/image.jpg --prompt "How many dogs are in the image?"
```

---

## 🏋️ Training Pipeline

### 1. Data Preparation
The training annotations are provided in `data/vg_brutal_pairs.jsonl`. Download the corresponding Visual Genome images:
```bash
python scripts/download_vg_images.py \
  --names-file data/vg_brutal_image_names.txt \
  --out-dir data/visual_genome
```

### 2. Multi-GPU Training (DDP)
Launch 2-GPU (or multi-GPU) distributed training with Gated GRPO:
```bash
bash scripts/train_qwen35_9b_ddp.sh \
  --gpus 0,1 \
  --model Qwen/Qwen3.5-9B \
  --output-dir checkpoints/qwen35_9b_answer_ced \
  --steps 2000
```

### 3. Single-GPU Training
For single-card setups:
```bash
bash scripts/train_qwen35_9b_single.sh \
  --gpu 0 \
  --model Qwen/Qwen3.5-9B \
  --output-dir checkpoints/qwen35_9b_answer_ced_single \
  --steps 2000
```

### 4. Merging Checkpoint
After training completes, merge the trained LoRA adapter into a standalone Hugging Face directory:
```bash
python scripts/merge_checkpoint.py \
  --base Qwen/Qwen3.5-9B \
  --checkpoint checkpoints/qwen35_9b_answer_ced/checkpoints/step_002000.pt \
  --dest checkpoints/qwen35_9b_answer_ced/merged_hf
```

---

## 📂 Repository Structure

```text
evidence-rl/
├── README.md                      # Documentation
├── demo.py                        # Standalone inference demo
├── requirements.txt               # Dependencies
├── environment.yml                # Conda environment
├── data/                          # Training annotations & data list
│   ├── vg_brutal_pairs.jsonl      # Visual Genome brutal pairs
│   └── vg_brutal_image_names.txt  # Image filename list
├── src/                           # Core training engine
│   ├── train_main_experiment.py   # Training loop & GRPO optimization
│   ├── ced_core.py                # Counterfactual Evidence Disentanglement (CED) intervention operator
│   ├── mini_grpo_smoke.py         # Rollout sampling & LoRA model setup
│   ├── model_loader.py            # VLM architecture loading utilities
│   ├── visual_token_map.py        # 2x2 spatial merge & visual token indexing
│   ├── answer_format_utils.py     # Two-line format parsing
│   ├── anti_hacking.py            # Length & format penalty verifiers
│   └── reward/                    # CED reward components
│       ├── action_logprob_ate.py  # Counterfactual margin computation
│       ├── preference_gate.py     # Routed gate function g(m)
│       └── causal_margin.py       # Margin statistics & bounds
└── scripts/                       # Training & utility scripts
    ├── train_qwen35_9b_ddp.sh     # Multi-GPU training entry
    ├── train_qwen35_9b_single.sh  # Single-GPU training entry
    ├── merge_checkpoint.py        # LoRA to HF checkpoint merge script
    └── download_vg_images.py      # Visual Genome downloader
```

---

## 📖 Citation

If you find this work or code useful, please cite:

```bibtex
@misc{huang2026evidencerlevidenceintensivevisualreasoning,
      title={Evidence-RL: Towards Evidence-intensive Visual Reasoning}, 
      author={Haojie Huang and Xinlei Yu and Chengming Xu and Zhangquan Chen and Cheng Yang and Qingdong He and Yu Yang and Jiangning Zhang and Xiaobin Hu},
      year={2026},
      eprint={2608.08021},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2608.08021}, 
}
```

---

## 📄 License
This repository is licensed under the [Apache 2.0 License](LICENSE).
