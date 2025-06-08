# WMCopier: Forging Invisible Image Watermarks on Arbitrary Images

Official implementation for the paper:  
**WMCopier: Forging Invisible Image Watermarks on Arbitrary Images**

<!-- > (Paper link: [arXiv/xxxx.xxxxx](https://arxiv.org/abs/xxxx.xxxxx)) -->

The project explores attacks against invisible watermarking schemes using diffusion models, and provides tools for watermark forgery.

---

## Getting Started

#### Generate Watermarked Auxiliary Dataset
Create the auxiliary dataset by running the script:
```bash
bash  scripts/create_dataset.sh
```

#### Train a diffusion model on Auxiliary Dataset
Before training, please update the train_data_path variable in setting.py with the path to your auxiliary dataset.

And then train a watermarked unconditional diffusion model by running the script:
```bash
bash  scripts/train.sh
```
Alternatively, you may use our pretrained model(trained d on 5000 watermarked images(RivaGan)) by downloading the checkpoints provided [here](https://drive.google.com/file/d/1Hq4bLlxyIIZcVllrD4TcwTbhOD2iczOY/view?usp=drive_link).



#### Perform our attack







---

## Reference

The training scripts for the diffusion model are based on the [HuggingFace tutorial](https://huggingface.co/docs/diffusers/tutorials/basic_training).

The implement of distortions are adapted from the [Tree-ring-watermark](https://github.com/YuxinWenRick/tree-ring-watermark).

