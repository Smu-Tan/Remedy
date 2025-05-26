<p align="center">
  <img src="https://github.com/user-attachments/assets/dcc72523-c91c-4712-9478-28850e1151c6" alt="ReMedy Logo" width="440"/>
</p>
<h1 align="center">🚀 ReMedy: Machine Translation Evaluation via Reward Modeling</h1>
<p align="center"><b>Learning High-Quality Machine Translation Evaluation from Human Preferences with Reward Modeling</b></p>

<p align="center">
<a href="https://arxiv.org/abs/2504.13630" style="text-decoration: none; border: none;"><img src="https://img.shields.io/badge/arXiv-2405.12345-b31b1b" alt="arXiv" style="border: none;" /></a>  <a href="https://pypi.org/project/remedy-mt-eval/" style="text-decoration: none; border: none;"><img src="https://img.shields.io/pypi/v/remedy-mt-eval" alt="PyPI version" style="border: none;" /></a>  <a href="https://github.com/Smu-Tan/Remedy/stargazers" style="text-decoration: none; border: none;"><img src="https://img.shields.io/github/stars/Smu-Tan/Remedy" alt="GitHub Stars" style="border: none;" /></a>  <a href="./LICENSE" style="text-decoration: none; border: none;"><img src="https://img.shields.io/github/license/Smu-Tan/Remedy" alt="License" style="border: none;" /></a> 
</p>


---

## ✨ About ReMedy

**ReMedy** is a new state-of-the-art machine translation (MT) evaluation framework that reframes the task as **reward modeling** rather than direct regression. Instead of relying on noisy human scores, ReMedy learns from **pairwise human preferences**, leading to better alignment with human judgments.

- 📈 **State-of-the-art accuracy** on WMT22–24 (39 language pairs, 111 systems)  
- ⚖️ **Segment- and system-level** evaluation, outperforming GPT-4, PaLM-540B, Finetuned-PaLM2, MetricX-13B, and XCOMET  
- 🔍 **More robust** on low-quality and out-of-domain translations (ACES, MSLC benchmarks)  
- 🧠 Can be used as a **reward model** in RLHF pipelines to improve MT systems  

> ReMedy demonstrates that **reward modeling with pairwise preferences** offers a more reliable and human-aligned approach for MT evaluation.

---

## 📚 Contents

- [📦 Quick Installation](#-quick-installation)
- [⚙️ Requirements](#️-requirements)
- [🚀 Usage](#-usage)
  - [💾 Download Models](#-download-remedy-models)
  - [🔹 Basic Usage](#-basic-usage)
  - [🔹 Reference-Free Mode](#-reference-free-mode)
  - [📄 Output Files](#-output-files)
- [⚙️ Full Argument List](#️-full-argument-list)
- [🧠 Model Variants](#-model-variants)
- [🔁 Reproducing WMT Results](#-reproducing-wmt-results)
- [📚 Citation](#-citation)

---

## 📦 Quick Installation

> ReMedy requires **Python ≥ 3.12**, and leverages **[VLLM](https://github.com/vllm-project/vllm)** for fast inference.

### ✅ Recommended: Install via pip

```bash
pip install --upgrade pip
pip install remedy-mt-eval
```

### 🛠️ Install from Source

```bash
git clone https://github.com/Smu-Tan/Remedy
cd Remedy
pip install -e .
```

### 📜 Install via Poetry

```bash
git clone https://github.com/Smu-Tan/Remedy
cd Remedy
poetry install
```

---

## ⚙️ Requirements

- `Python` ≥ 3.12  
- `transformers` ≥ 4.51.1  
- `vllm` ≥ 0.8.5  
- `torch` ≥ 2.6.0  
- *(See `pyproject.toml` for full dependencies)*

---

## 🚀 Usage

### 💾 Download ReMedy Models

Before using, you can download the model from HuggingFace:

```bash
HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download ShaomuTan/ReMedy-9B-22 --local-dir Models/ReMedy-9B-22
```

You can replace `ReMedy-9B-22` with other variants like `ReMedy-9B-23`.

---

### 🔹 Basic Usage

```bash
remedy-score \
    --model ShaomuTan/ReMedy-9B-22 \
    --src_file testcase/en.src \
    --mt_file testcase/en-de.hyp \
    --ref_file testcase/de.ref \
    --src_lang en --tgt_lang de \
    --cache_dir Models \
    --save_dir testcase \
    --num_gpus 4 \
    --calibrate
```

### 🔹 Reference-Free Mode (Quality Estimation)

```bash
remedy-score \
    --model ShaomuTan/ReMedy-9B-22 \
    --src_file testcase/en.src \
    --mt_file testcase/en-de.hyp \
    --no_ref \
    --src_lang en --tgt_lang de \
    --cache_dir Models \
    --save_dir testcase/QE \
    --num_gpus 4 \
    --calibrate
```

---

## 📄 Output Files

- `src-tgt_raw_scores.txt`
- `src-tgt_sigmoid_scores.txt`
- `src-tgt_calibration_scores.txt`
- `src-tgt_detailed_results.tsv`
- `src-tgt_result.json`

Inspired by **SacreBLEU**, ReMedy provides JSON-style results to ensure transparency and comparability.

<details>
<summary>📘 Example JSON Output</summary>

```json
{
  "metric_name": "remedy-9B-22",
  "raw_score": 4.502863049214531,
  "sigmoid_score": 0.9613502018042875,
  "calibration_score": 0.9029647169507162,
  "calibration_temp": 1.7999999999999998,
  "signature": "metric_name:remedy-9B-22|lp:en-de|ref:yes|version:0.1.1",
  "language_pair": "en-de",
  "source_language": "en",
  "target_language": "de",
  "segments": 2037,
  "version": "0.1.1",
  "args": {
    "src_file": "testcase/en.src",
    "mt_file": "testcase/en-de.hyp",
    "src_lang": "en",
    "tgt_lang": "de",
    "model": "Models/remedy-9B-22",
    "cache_dir": "Models",
    "save_dir": "testcase",
    "ref_file": "testcase/de.ref",
    "no_ref": false,
    "calibrate": true,
    "num_gpus": 4,
    "num_seqs": 256,
    "max_length": 4096,
    "enable_truncate": false,
    "version": false,
    "list_languages": false
  }
}
```

</details>

---

## ⚙️ Full Argument List

<details>
<summary>📋 Show CLI Arguments</summary>

### 🔸 Required

```python
--src_file           # Path to source file
--mt_file            # Path to MT output file
--src_lang           # Source language code
--tgt_lang           # Target language code
--model              # Model path or HuggingFace ID
--save_dir           # Output directory
```

### 🔸 Optional

```python
--ref_file           # Reference file path
--no_ref             # Reference-free mode
--cache_dir          # Cache directory
--calibrate          # Enable calibration
--num_gpus           # Number of GPUs
--num_seqs           # Number of sequences (default: 256)
--max_length         # Max token length (default: 4096)
--enable_truncate    # Truncate sequences
--version            # Print version
--list_languages     # List supported languages
```

</details>

---

## 🧠 Model Variants

| Model         | Size | Base Model   | Ref/QE | Download |
|---------------|------|--------------|--------|----------|
| ReMedy-9B-22  | 9B   | Gemma-2-9B   | Both   | [🤗 HuggingFace](https://huggingface.co/ShaomuTan/ReMedy-9B-22) |
| ReMedy-9B-23  | 9B   | Gemma-2-9B   | Both   | [🤗 HuggingFace](https://huggingface.co/ShaomuTan/ReMedy-9B-23) |
| ReMedy-9B-24  | 9B   | Gemma-2-9B   | Both   | [🤗 HuggingFace](https://huggingface.co/ShaomuTan/ReMedy-9B-24) |

> More variants coming soon...

---

## 🔁 Reproducing WMT Results

<details>
<summary>Click to show instructions for reproducing WMT22–24 evaluation</summary>

### 1. Clone ReMedy repo
```bash
git clone https://github.com/Smu-Tan/Remedy
cd Remedy
```

### 2. Install `mt-metrics-eval`

```bash
# Install MTME and download WMT data
git clone https://github.com/google-research/mt-metrics-eval.git
cd mt-metrics-eval
pip install .
cd ..
python3 -m mt_metrics_eval.mtme --download
```

### 3. Run ReMedy on WMT data

```bash
sbatch wmt/wmt22.sh
sbatch wmt/wmt23.sh
sbatch wmt/wmt24.sh
```

> 📄 Results will be comparable with other metrics reported in WMT shared tasks.

</details>

---

## 📚 Citation

If you use **ReMedy**, please cite the following paper:

```bibtex
@article{tan2024remedy,
  title={ReMedy: Learning Machine Translation Evaluation from Human Preferences with Reward Modeling},
  author={Tan, Shaomu and Monz, Christof},
  journal={arXiv preprint},
  year={2024}
}
```

---
