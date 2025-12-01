
# WMCopier: Forging Invisible Image Watermarks on Arbitrary Images

This repository explores attacks against invisible watermarking schemes using diffusion models, providing practical tools for watermark forgery and evaluation.

<p align="center">
  <img src="framework.png" alt="Framework of WMCopier" width="95%">
</p>

---

## 🚀 Getting Started

### 0. Install Requirements

First, create and activate a conda environment with **Python 3.9+**:

```bash
conda create -n wmcopier python=3.9
conda activate wmcopier
```
Then install all required dependencies:
```bash
pip install -r requirements.txt
```

### 1. Generate Watermarked Auxiliary Dataset

You can easily generate watermarked images using our toolkit:

👉 **https://github.com/holdrain/WMSuite.git**

WMSuite currently supports **six watermarking algorithms**:

- **HiDDeN**
- **DwTDCT**
- **RivaGAN**
- **StegaStamp**
- **VINE**
- **Stable Signature**


### 2. Train a diffusion attacker

Before training, update the `train_data_path` variable in `setting.py` with the path to your auxiliary dataset.

Then train the watermarked unconditional diffusion model:
```bash
bash scripts/train.sh
```

Alternatively, you can use our pretrained model (trained on 5,000 watermarked images). Download the checkpoint package from [Google Drive](https://drive.google.com/file/d/1uROeoV2l3dcyGCGS-vB3_pv_UXCKymEM/view?usp=sharing).

After downloading, unzip the file and place the extracted folders so that your project structure looks like:


```
-WMCopier
  -checkpoints
  -configs
  ...
```

### 3. Perform the Forgery Attack

You can try our forgery attack with a simple example on RivaGAN watermark by running the notebook: **`demo/forge.ipynb`**

For large-scale experiments on an entire image folder, run:

```bash
bash scripts/attack.sh
```

---
### 4. Attack on Real-World Watermark

We do not publish the checkpoints here, following discussions with Amazon’s AGI Team. For more details, please refer to the "Broad Impact" section in our paper.







## 🔗 Reference

- Training scripts for the diffusion model are based on the [HuggingFace Diffusers tutorial](https://huggingface.co/docs/diffusers/tutorials/basic_training).
- Distortion implementations are adapted from [Tree-ring-watermark](https://github.com/YuxinWenRick/tree-ring-watermark).