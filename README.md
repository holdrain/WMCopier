
# WMCopier: Forging Invisible Image Watermarks on Arbitrary Images

This repository explores attacks against invisible watermarking schemes using diffusion models, providing practical tools for watermark forgery and evaluation.

<p align="center">
  <img src="framework.png" alt="Framework of WMCopier" width="95%">
</p>

---

## 🚀 Getting Started

### 0. Install Requirements

It is recommended to use **Python 3.9+** and a virtual environment.  
To install all required dependencies, simply run:

```bash
pip install -r requirements.txt
```

### 1. Generate Watermarked Auxiliary Dataset

To create the auxiliary dataset, run:
```bash
bash scripts/create_dataset.sh
```

### 2. Train a Diffusion Model on the Auxiliary Dataset

Before training, update the `train_data_path` variable in `setting.py` with the path to your auxiliary dataset.

Then train the watermarked unconditional diffusion model:
```bash
bash scripts/train.sh
```

Alternatively, you can use our pretrained model (trained on 5,000 watermarked images with RivaGAN) by downloading the checkpoints from [Google Drive](https://drive.google.com/file/d/1ymPsx4VAY-jtZuljX9S0v9FuiiZRNmNe/view?usp=drive_link).

After downloading, place the checkpoint files in:
```
experiments/ddim_rivagan_no/01-05_10:12/ckp
```

### 3. Run the Forgery Attack

To forge watermarks on arbitrary images, run:
```bash
bash scripts/attack.sh
```

---
### 4. Attack on Real-World Watermark
We do not publish the checkpoints here, following discussions with Amazon’s AGI Team. For more details, please refer to the "Broad Impact" section in our paper.

---
## 🔗 Reference

- Training scripts for the diffusion model are based on the [HuggingFace Diffusers tutorial](https://huggingface.co/docs/diffusers/tutorials/basic_training).
- Distortion implementations are adapted from [Tree-ring-watermark](https://github.com/YuxinWenRick/tree-ring-watermark).