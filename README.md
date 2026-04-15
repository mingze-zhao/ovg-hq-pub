# OVG-HQ: Online Video Grounding with Hybrid-modal Queries

Official implementation of **OVG-HQ** ([ICCV 2025 OpenAccess PDF](https://openaccess.thecvf.com/content/ICCV2025/papers/Zeng_OVG-HQ_Online_Video_Grounding_with_Hybrid-modal_Queries_ICCV_2025_paper.pdf)).

**OVG-HQ** extends video grounding to **online** streaming video with **hybrid-modal queries** (text, images, video segments, and combinations). The **OVG-HQ-Unify** framework uses a **Parametric Memory Block (PMB)** (Test-Time Training–style sequence modeling) to retain useful context under limited streaming history, and **cross-modal distillation** to reduce modality imbalance when training on mixed query types. The paper introduces **QVHighlights-Unify** and online metrics **oR@n**, **IoU=m**, and **omAP** for accuracy and timeliness.

**Authors:** Runhao Zeng*, Jiaqi Mao*, Minghao Lai, Minh Hieu Phan, Yanjie Dong, Wei Wang, Qi Chen†, Xiping Hu† (*equal contribution, †corresponding)

**Affiliations:** Artificial Intelligence Research Institute, Shenzhen MSU-BIT University; University of Adelaide

**Contact:** zengrh@smbu.edu.cn, maojiaqi2324@gmail.com, huxp@smbu.edu.cn

**Original authors’ repository (paper):** [https://github.com/maojiaqi2324/OVG-HQ/tree/main](https://github.com/maojiaqi2324/OVG-HQ/tree/main)

---

## Dataset download

Features and annotations for this project are shared via **hpccube** file hosting (browser download; extract code required):

| | |
|--|--|
| **Link** | [https://ksefile.hpccube.com:65241/efile/s/w/YWNsdjRiZWt4MA==_957306cfdd3e56f5](https://ksefile.hpccube.com:65241/efile/s/w/YWNsdjRiZWt4MA==_957306cfdd3e56f5) |
| **Extract code** | `GqHu` |
| **URL with code (if supported)** | [https://ksefile.hpccube.com:65241/efile/s/w/YWNsdjRiZWt4MA==_957306cfdd3e56f5?pwd=GqHu](https://ksefile.hpccube.com:65241/efile/s/w/YWNsdjRiZWt4MA==_957306cfdd3e56f5?pwd=GqHu) |

Open the link in a browser, enter **`GqHu`** when prompted, and download the archive. After extracting, place features and labels under the paths expected by your YAML (see **Data preparation**, e.g. `data/QVHighlight/features/...`).

---

## Repository layout

| Path | Role |
|------|------|
| [`training/train.py`](training/train.py) | Main training entry |
| [`training/evaluate.py`](training/evaluate.py) | Evaluation and submission-style metrics |
| [`training/config.py`](training/config.py) | YAML config loading (`configs/base.yml` + experiment YAML) |
| [`lighthouse/`](lighthouse/) | Model implementations (e.g., online QD-DETR, baselines, SlowFast integration) |
| [`configs/`](configs/) | Dataset- and task-specific YAML (QVHighlight-Unify, offline baselines, etc.) |
| [`annotation/`](annotation/) | JSONL annotations (including QVHighlight / unify / ICQ splits) |
| [`scripts/`](scripts/) | Shell examples for experts / unified training; [`multi_query_evaluate.py`](scripts/multi_query_evaluate.py) for multi-query evaluation |

---

## Environment

- **Python:** 3.x with **PyTorch** and **CUDA** (configs default to `device: cuda`; the paper reports experiments on a single **RTX 4090**).
- **Core libraries** (non-exhaustive): `torch`, `torchvision`, `numpy`, `pandas`, `tqdm`, `PyYAML`, `easydict`, `h5py`, `wandb`, `tensorboardX`, `einops`, `scipy`, `scikit-learn`, `matplotlib`, `pyinstrument`, and optional **`lighthouse/slowfast`** for CLIP+SlowFast setups.

There is no checked-in `requirements.txt`; install PyTorch for your CUDA version first, then add packages until `training/train.py` runs.

---

## Data preparation

0. **Download packaged data (optional)**  
   Use [Dataset download](#dataset-download) (hpccube link + extract code `GqHu`).

1. **Videos / features**  
   Place precomputed features under paths referenced in your YAML (see [`configs/base.yml`](configs/base.yml)). For QVHighlight-style runs, examples include:
   - `data/QVHighlight/features/clip_video`
   - `data/QVHighlight/features/clip_text` (and multimodal paths such as `clip_text_c`, `clip_image_g`, `clip_image_r`, `clip_image_c`, `clip_segment_g` when using [`configs/qvhighlight_clip/clip_unify.yml`](configs/qvhighlight_clip/clip_unify.yml)).

2. **Annotations**  
   Point `train_path` / `eval_path` in the YAML to JSONL files under [`annotation/`](annotation/) (e.g. `annotation/qvhighlight/…`, `annotation/qvhighlight_icq/…`, unify splits).

3. **Anchor labels (HDF5)**  
   `label_file_path` in `configs/base.yml` expects proposal label HDF5 files (see `label_output/…` pattern). Generate or obtain these to match your `segment_size`, `anchor_windows`, and downsampling settings.

Create local directories for features, checkpoints, and logs as required by your YAML paths.

---

## Training

From the **repository root** (so `configs/` resolves correctly):

```bash
python training/train.py --config <PATH_TO_YAML> [--pretrained_model_path <CKPT>] [--domain <DOMAIN>] [--savecode]
```

**Examples (see [`scripts/`](scripts/)):**

- **Text expert (QVHighlight CLIP):**

  ```bash
  python training/train.py --config configs/qvhighlight_clip/clip_text.yml --savecode
  ```

- **Unified hybrid-modal model with distillation** (requires a trained teacher; set `teacher_model_path` in [`configs/qvhighlight_clip/clip_unify.yml`](configs/qvhighlight_clip/clip_unify.yml)):

  ```bash
  python training/train.py --config configs/qvhighlight_clip/clip_unify.yml --savecode
  ```

`--savecode` snapshots training scripts into the run directory for reproducibility.

**Weights & Biases:** [`configs/base.yml`](configs/base.yml) contains `wandb_*` keys; `training/train.py` sets `WANDB_MODE=offline` in code—adjust if you want online logging.

---

## Evaluation

```bash
python training/evaluate.py \
  --config <YAML_USED_OR_SAVED_RUN_CONFIG> \
  --model_path <PATH_TO_BEST_CKPT> \
  --eval_split_name val \
  --eval_path <PATH_TO_EVAL_JSONL> \
  [--results_dir <DIR>] \
  [--eval_query_type <TYPE>]
```

See [`scripts/evaluation.sh`](scripts/evaluation.sh) for a concrete QVHighlight val example. Metrics and submissions are written under `results_dir` (JSONL + `_metrics.json` when applicable), using the standalone eval utilities under [`training/standalone_eval/`](training/standalone_eval/).

---

## License and third-party code

Training and model files retain notices from **Moment-DETR** ([moment_detr](https://github.com/jayleicn/moment_detr)) and **LY Corporation** (Apache-2.0 headers in several `training/` and `lighthouse/` files). Respect those licenses when reusing or redistributing.

---

## Citation

If you use this code or the QVHighlights-Unify / OVG-HQ setting, please cite the ICCV 2025 paper:

```bibtex
@inproceedings{zeng2025ovg,
  title={OVG-HQ: Online Video Grounding with Hybrid-modal Queries},
  author={Zeng, Runhao and Mao, Jiaqi and Lai, Minghao and Phan, Minh Hieu and Dong, Yanjie and Wang, Wei and Chen, Qi and Hu, Xiping},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={21085--21096},
  year={2025}
}
```

---

## Paper

ICCV 2025 (official OpenAccess PDF): [https://openaccess.thecvf.com/content/ICCV2025/papers/Zeng_OVG-HQ_Online_Video_Grounding_with_Hybrid-modal_Queries_ICCV_2025_paper.pdf](https://openaccess.thecvf.com/content/ICCV2025/papers/Zeng_OVG-HQ_Online_Video_Grounding_with_Hybrid-modal_Queries_ICCV_2025_paper.pdf)
