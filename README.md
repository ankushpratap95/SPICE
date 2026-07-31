# SPICE: Synergy and Partial Information Based Curriculum Evolution

Official PyTorch implementation of **SPICE**, a PID-guided progressive curriculum framework for multimodal interaction learning.

> **SPICE: Synergy and Partial Information Based Curriculum Evolution**
> Ankush Pratap Singh, Houwei Cao, Yong Liu
> *ACM International Conference on Multimodal Interaction (ICMI '26), Napoli, Italy*
> [[Paper / DOI]](https://doi.org/10.1145/3776574.3831187)

---

## Overview

Existing multimodal curriculum learning strategies usually assume that the relative difficulty of samples is fixed throughout training. SPICE instead treats sample difficulty as a **dynamic, model-dependent quantity** and reorders samples in real time as the model matures.

Guided by **Partial Information Decomposition (PID)** theory, SPICE decomposes each sample's multimodal information into three interpretable components estimated directly from the model's own unimodal and multimodal predictions:

- **Redundant (R)** — shared, mutually consistent cues that all modalities agree on and that align with the label. *Easy.*
- **Unique (U)** — discriminative evidence carried by a single dominant modality. *Specialized.*
- **Synergistic (S)** — higher-order information that only emerges after fusion. *Complex.*

The curriculum progresses from easy to complex:

```
redundant (easy) → unique (specialized) → synergistic (complex)
```

PID scores are recomputed periodically during training (every `k = 5` epochs by default), so the ordering continuously adapts to the model's evolving state.

We provide two sample-allocation strategies:

| Variant | Script | Strategy |
|---------|--------|----------|
| **SPICE-E** | `train-PID.py` | Uses the **entire dataset** at every stage; reorders samples via stage-specific PID sampling probabilities. Best overall accuracy. |
| **SPICE-S** | `train-PID-stagewise.py` | Bins samples by dominant PID component and adds bins stage by stage (`R → R∪U → R∪U∪S`). More gradient-efficient in early stages. |

---

## Repository Structure

```
SPICE/
├── CREMA-D/            # Audio–Visual emotion recognition (2 modalities)
├── KS/                 # Kinetics-Sounds audiovisual action recognition (2 modalities)
├── NVGesture/          # Trimodal gesture recognition: RGB / Optical-Flow / Depth (3 modalities)
├── VGGSound/           # Large-scale audiovisual benchmark (2 modalities)
└── requirements.txt

# Each dataset folder contains:
<dataset>/
├── train-PID.py             # SPICE-E (entire-dataset ordering)
├── train-PID-stagewise.py   # SPICE-S (stage-wise binning)
├── models/                  # ResNet18 encoders, fusion module, full model
├── dataset/                 # Dataset loader
└── utils/                   # Seeding, weight init, helpers
```

---

## Installation

```bash
git clone https://github.com/ankushpratap95/SPICE.git
cd SPICE
pip install -r requirements.txt
```

The code was developed and tested with:

```
torch==1.7.1        torchvision==0.8.2
numpy==1.21.2       scipy==1.7.1
opencv-python==4.5.3.56   Pillow==8.3.2
h5py==2.10.0        transformers==4.19.0
matplotlib==3.4.3   tqdm==4.62.2
```

A CUDA-capable GPU is strongly recommended.

---

## Data Preparation

SPICE follows the same dataset preprocessing protocol as **GeWU-Lab's OGM-GE** ([gewu-lab/ogm-ge_cvpr2022](https://github.com/gewu-lab/ogm-ge_cvpr2022)). Please refer to those repositories for downloading raw data and extracting frames/audio. After preprocessing, arrange the data so the loaders can find it, as described below.

### CREMA-D
Expects (relative to `CREMA-D/`):
```
data/CREMAD/Crema_D/
├── annotations/train.csv, test.csv
├── AudioWAV/*.wav                 # --audio_path
└── Image-01-FPS/<clip_id>/*.jpg   # extracted frames, under --visual_path
```

### Kinetics-Sounds (KS)
Expects (under `--data_root`):
```
<data_root>/
├── annotations/train.csv, test.csv
├── train_img/Image-01-FPS/<clip>/ , test_img/Image-01-FPS/<clip>/
└── train_wav/<clip>.wav , test_wav/<clip>.wav
```

### NVGesture
Expects RGB, optical-flow, and depth modalities under `--data_root` (default `./data/nvGesture/nvGesture_v1/`), following the standard NVGesture split file layout.

### VGGSound
Paths are configured inside `VGGSound/dataset/VGGSoundDataset.py`. It expects `./data/vggsound.csv` plus extracted frames under `./data/video/{train,test}-set-img/Image-01-FPS/` and audio under `./data/video/{train,test}-audios/`. Edit these paths to match your local storage.

> Frame extraction uses `Image-XX-FPS` folders where `XX` matches the `--fps` argument (default `1`). Make sure your extracted-frame directory name matches the `--fps` you train with.

---

## Training

Run from inside the relevant dataset folder. Below are the defaults used in the paper; adjust paths as needed.

### CREMA-D
```bash
cd CREMA-D
# SPICE-E (recommended)
python train-PID.py \
    --audio_path ./data/CREMAD/Crema_D/AudioWAV \
    --visual_path ./data/CREMAD/Crema_D/ \
    --batch_size 32 --learning_rate 0.01 --epochs 150 --warm_up 30

# SPICE-S
python train-PID-stagewise.py --audio_path ... --visual_path ...
```

### Kinetics-Sounds
```bash
cd KS
python train-PID.py --data_root ./kinetics-dataset/k400 \
    --batch_size 32 --learning_rate 0.1 --epochs 150 --warm_up 30
```

### NVGesture (trimodal)
```bash
cd NVGesture
python train-PID.py --data_root ./data/nvGesture/nvGesture_v1/ \
    --batch_size 4 --learning_rate 0.01 --epochs 150 --warm_up 30
```

### VGGSound (large-scale)
```bash
cd VGGSound
python train-PID.py --batch_size 128 --learning_rate 0.01 --epochs 150 --warm_up 30
```

### Key arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--epochs` | Total training epochs (split into 3 curriculum stages) | 150 |
| `--warm_up` | Warm-up epochs before curriculum begins (stabilizes early predictions) | 30 |
| `--batch_size` | Batch size | 32 / 4 (NVG) / 128 (VGGSound) |
| `--learning_rate` | Initial learning rate | 0.01 / 0.1 (KS) |
| `--optimizer` | `sgd` or `adam` | `sgd` (momentum 0.9, weight decay 1e-4) |
| `--random_seed` | Random seed | 1751780633 |

PID scores are recomputed every 5 epochs (`epoch % 5 == 0`) and the three curriculum stages each span one third of the post-warm-up epochs.

---

## Method at a Glance

For a bimodal sample with unimodal confidences $c^{(m)}$ (true-class softmax) and multimodal confidence $c^{multi}$, and pairwise/joint KL divergences $D$:

- **Redundancy:** $R_i = \left(\prod_m c^{(m)}_i\right)\cdot \exp(-\bar{D}_i)$ — all modalities confident *and* agree.
- **Unique:** $U^{(m)}_i = c^{(m)}_i\prod_{n\neq m}(1-c^{(n)}_i)\cdot\left(1-\exp(-D^{(m)}_i)\right)$ — one modality confident, others uncertain and divergent.
- **Synergy:** $S_i = \left(c^{multi}_i - \max_m c^{(m)}_i\right)\cdot D^{syn}_i$ — fusion improves over the best single modality.

Scores are min-max normalized to $[0,1]$. In **SPICE-E**, the Redundant stage samples proportionally to $P_R$ (easy-first), while the Unique and Synergistic stages sample proportionally to $1-P_U$ and $1-P_S$ (easy-to-hard). In **SPICE-S**, samples are hard-assigned to bins via $\arg\max(\hat R,\hat U,\hat S)$ and bins are introduced stage by stage. The trimodal (NVGesture) formulation generalizes these to three unimodal branches.

At inference, SPICE uses standard late fusion, summing the multimodal and unimodal logits (following BSS, MLA, and DI-MML).

---

## Citation

Please cite the work:

```bibtex
@misc{singh2026spicesynergypartialinformation,
      title={SPICE: Synergy and Partial Information Based Curriculum Evolution}, 
      author={Ankush Pratap Singh and Houwei Cao and Yong Liu},
      year={2026},
      eprint={2606.16639},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2606.16639}, 
}
```

<!-- Once the ICMI '26 proceedings are published, replace the entry above with:
@inproceedings{singh2026spice,
  title     = {SPICE: Synergy and Partial Information Based Curriculum Evolution},
  author    = {Singh, Ankush Pratap and Cao, Houwei and Liu, Yong},
  booktitle = {Proceedings of the International Conference on Multimodal Interaction (ICMI '26)},
  year      = {2026},
  address   = {Napoli, Italy},
  publisher = {ACM},
  doi       = {10.1145/3776574.3831187}
}
-->

---

## Acknowledgments

Dataset preprocessing and evaluation protocols follow the **BSS**([njustkmg/IJCAI25-BSS](https://github.com/njustkmg/IJCAI25-BSS)) framework and **GeWU-Lab's OGM-GE** ([gewu-lab/ogm-ge_cvpr2022](https://github.com/gewu-lab/ogm-ge_cvpr2022)). We thank the authors of these works, as well as the maintainers of the CREMA-D, Kinetics-Sounds, NVGesture, and VGGSound datasets.

## License

This code is released under the [MIT License](LICENSE) for academic and research use. Dataset usage is subject to the respective dataset licenses (CREMA-D, Kinetics-Sounds, NVGesture, VGGSound), and portions of the data-preprocessing pipeline derive from the BSS and OGM-GE repositories — please also respect their licenses.
