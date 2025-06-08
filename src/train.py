import argparse
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from setting import model_choices
from trainer import Trainer
from utils.helpers import *


def Options():
    parser = argparse.ArgumentParser(description="Train a diffusion model")

    # General settings
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to the configuration file")
    parser.add_argument("--seed", type=int, default=2025, help="Random seed")
    parser.add_argument("--results_folder", type=str, help="Path to the output directory")
    parser.add_argument('--exp_name', type=str, default='ddim_hidden', help="Name of the experiment")

    # Image and batch size settings
    parser.add_argument('--data_path', type=str, help="Path to the data directory")
    parser.add_argument("--train_batchsize", type=int, default=32, help="Training batch size")
    parser.add_argument("--num_samples", type=int, default=32, help="Evaluation batch size")

    # Model
    parser.add_argument("--image_size", type=int, required=True, help="Size of the input images")
    parser.add_argument("--backbone", type=str, default="unet", help="Model architecture name (default: 'unet')")
    
    # Learning rate and optimization settings
    parser.add_argument("--learning_rate", type=float, help="Learning rate for optimizer")
    parser.add_argument("--lr_scheduler", type=str, default="no", help="Learning rate scheduler (default: 'no')")
    parser.add_argument('--train_num_steps', type=int, default=700000, help="Number of total training steps")
    parser.add_argument('--gradient_accumulate_every', type=int, default=2, help="Number of steps to accumulate gradients before updating weights")
    parser.add_argument('--ema_decay', type=float, default=0.995, help="Exponential moving average decay rate")
    parser.add_argument('--amp', action='store_true', help="Whether to enable mixed precision training")

    # Fine-tuning and EMA settings
    parser.add_argument("--use_ema",action="store_true", help="Enable Exponential Moving Average (EMA) of model weights")

    # Training continuation settings
    parser.add_argument('--pretrained_ckp',default=None, help="Path to the pretrained checkpoint")
    parser.add_argument("--resume_from", type=int, default=None, help="resume from which step, e.g. if equals N, then `model-N.pt` will be loaded from the experiment result directory")
    parser.add_argument('--train_num', type=int, default=None, help="Number of wmimages")
    parser.add_argument('--save_and_sample_every', type=int, help="Number of steps to save and sample the model")
    parser.add_argument('--beta_schedule',default='linear')
    return parser.parse_args()


if __name__ == "__main__":
    # config
    args = Options()
    Tconfig = parse_yml(args.config)
    print(Tconfig.items())
    Tconfig = combine(args, Tconfig)
    set_seeds(Tconfig.seed)
    now = datetime.now()
    timestamp = now.strftime("%m-%d_%H:%M")
    if not Tconfig.resume_from:
        Tconfig.results_folder = new_dir(os.path.join(Tconfig.results_folder, Tconfig.exp_name, timestamp))
    save_config_to_json(Tconfig.results_folder + "/Tconfig.json", Tconfig)
    

    # Model,noise_scheduler,optmizer,lr
    model = model_choices[Tconfig.backbone](sample_size=Tconfig.image_size)

    # train
    trainer = Trainer(
        diffusion_model=model,
        folder=Tconfig.data_path,
        image_size = Tconfig.image_size,
        train_batch_size=Tconfig.train_batchsize,
        train_lr=Tconfig.learning_rate,
        lr_scheduler=Tconfig.lr_scheduler,
        gradient_accumulate_every= Tconfig.gradient_accumulate_every,
        use_ema = Tconfig.use_ema,
        train_num_steps = Tconfig.train_num_steps,
        ema_update_every = Tconfig.ema_update_every,
        ema_decay = Tconfig.ema_decay,
        num_samples = Tconfig.num_samples,
        save_and_sample_every = Tconfig.save_and_sample_every,
        results_folder = Tconfig.results_folder,
        amp = Tconfig.amp,
        resume_from = Tconfig.resume_from,
        pretrained_ckp = Tconfig.pretrained_ckp,  # for finetune mode
        train_num = Tconfig.train_num,
        beta_schedule = Tconfig.beta_schedule,
    )
    trainer.train()