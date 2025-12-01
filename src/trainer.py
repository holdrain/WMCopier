# trainer class

import random
from itertools import cycle
import torch
from accelerate import Accelerator
from diffusers import DDIMPipeline, DDIMScheduler
from ema_pytorch import EMA
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ExponentialLR, StepLR
from torch.utils.data import DataLoader
from torchvision import transforms, utils
from tqdm.auto import tqdm

from src.dataset import CustomImageFolder
from src.utils.helpers import *
from src.setting import beta_schedule_choices

def get_lr_scheduler(optimizer, scheduler_type, **kwargs):
    """
    Returns different types of learning rate schedulers (not based on validation set).

    Arguments:
    - optimizer: The optimizer used for training (e.g., Adam, SGD, etc.)
    - scheduler_type: Type of the learning rate scheduler to use
    - kwargs: Additional hyperparameters, passed based on the scheduler type

    Returns:
    - scheduler: The learning rate scheduler
    """
    if scheduler_type == "no":
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda step: 1.0
        )
    elif scheduler_type == "step":
        # StepLR: Reduces the learning rate every 'step_size' steps by a factor of 'gamma'
        step_size = kwargs.get(
            "step_size", 100000
        )  # The number of steps after which the learning rate is reduced
        gamma = kwargs.get(
            "gamma", 0.1
        )  # The factor by which the learning rate is reduced
        scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)

    elif scheduler_type == "exponential":
        # ExponentialLR: Decays the learning rate exponentially by the factor 'gamma' after each step
        gamma = kwargs.get("gamma", 0.999)  # The decay factor
        scheduler = ExponentialLR(optimizer, gamma=gamma)

    elif scheduler_type == "cosine":
        # CosineAnnealingLR: Adjusts the learning rate using cosine annealing
        T_max = kwargs.get(
            "T_max", 500000
        )  # The maximum number of iterations for one cycle
        eta_min = kwargs.get("eta_min", 1e-7)  # The minimum learning rate value
        scheduler = CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)

    else:
        raise ValueError(f"Unsupported scheduler type: {scheduler_type}")

    return scheduler


def exists(x):
    return x is not None


def load_training_state(
    model, ema, accelerator, opt, lr_scheduler, milestone, results_folder="./results"
):
    device = accelerator.device
    data = torch.load(
        os.path.join(results_folder, f"model-{milestone}.pt"), map_location=device
    )
    model = accelerator.unwrap_model(model)
    model.load_state_dict(data["model"])
    step = data["step"]
    opt.load_state_dict(data["opt"])
    lr_scheduler.load_state_dict(data["lr_scheduler"])

    if ema is not None:
        ema.load_state_dict(data["ema"])
    if exists(accelerator.scaler) and exists(data["scaler"]):
        accelerator.scaler.load_state_dict(data["scaler"])

    return step, model, ema, opt, lr_scheduler


def load_pretrained_model(model, accelerator, pretrained_ckp):
    data = torch.load(pretrained_ckp, map_location="cpu")
    if accelerator is not None:
        model = accelerator.unwrap_model(model)
        if exists(accelerator.scaler) and exists(data["scaler"]):
            accelerator.scaler.load_state_dict(data["scaler"])
    model.load_state_dict(data["model"])
    return model


def divisible_by(numer, denom):
    return (numer % denom) == 0


def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


class Trainer:
    def __init__(
        self,
        diffusion_model,
        folder,
        image_size,
        *,
        train_batch_size=16,
        gradient_accumulate_every=1,
        train_lr=1e-4,
        lr_scheduler="no",
        use_ema=False,
        train_num_steps=100000,
        ema_update_every=10,
        ema_decay=0.995,
        adam_betas=(0.9, 0.99),
        num_samples=4,
        save_and_sample_every=5,
        results_folder="./results",
        amp=False,
        mixed_precision_type="fp16",
        split_batches=True,
        max_grad_norm=1.0,
        resume_from=0,
        pretrained_ckp=None,  # for finetune mode
        train_num=None,
        beta_schedule='linear',
        input_type = 'image',
    ):
        super().__init__()

        # accelerator

        self.accelerator = Accelerator(
            split_batches=split_batches,
            # dispatch_batches=True,
            mixed_precision=mixed_precision_type if amp else "no",
            log_with="tensorboard",
            project_dir=results_folder,
            step_scheduler_with_optimizer=False,  # control the lr decay seperately
        )
        if self.accelerator.is_main_process:
            self.accelerator.init_trackers("logs")

        # model
        self.model = diffusion_model
        self.criterion = torch.nn.MSELoss()
        # beta schedule
        betas = beta_schedule_choices[beta_schedule]()
        self.noise_scheduler = DDIMScheduler(num_train_timesteps=1000,trained_betas=betas)

        # sampling and training hyperparameters
        assert has_int_squareroot(
            num_samples
        ), "number of samples must have an integer square root"
        self.num_samples = num_samples
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every

        assert (
            train_batch_size * gradient_accumulate_every
        ) >= 16, f"your effective batch size (train_batch_size x gradient_accumulate_every) should be at least 16 or above"

        self.train_num_steps = train_num_steps
        self.image_size = diffusion_model.sample_size
        self.max_grad_norm = max_grad_norm

        # dataset and dataloader
        transform = transforms.Compose(
            [
                transforms.Resize((image_size,image_size)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        self.ds = CustomImageFolder(folder, transform=transform, num=train_num, random_sample=True)
        if self.accelerator.is_main_process: 
            print(f"train on {len(self.ds)} images~")

        dl = DataLoader(
            self.ds,
            batch_size=train_batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=0,
        )

        dl = self.accelerator.prepare(dl)
        self.dl = cycle(dl)

        # optimizer and lr_scheduler
        self.opt = AdamW(diffusion_model.parameters(), lr=train_lr, betas=adam_betas)
        self.lr_scheduler = get_lr_scheduler(self.opt, lr_scheduler)
        # for logging results in a folder periodically

        self.ema = None
        if self.accelerator.is_main_process:
            if use_ema:
                self.ema = EMA(
                    diffusion_model, beta=ema_decay, update_every=ema_update_every
                )
                self.ema.to(self.device)
        # results folder
        self.results_folder = new_dir(results_folder)
        self.samples_results_folder = new_dir(os.path.join(results_folder, "samples"))
        self.ckp_results_folder = new_dir(os.path.join(results_folder, "ckp"))

        # step counter state
        self.step = 0

        # prepare model, dataloader, optimizer, loss with accelerator
        if resume_from > 0:
            # resume mode
            self.step, self.model, self.ema, self.opt, self.lr_scheduler = (
                load_training_state(
                    self.model,
                    self.ema,
                    self.accelerator,
                    self.opt,
                    self.lr_scheduler,
                    resume_from,
                    self.ckp_results_folder,
                )
            )
            self.accelerator.print(
                f"Resume training from checkpoint: model-{resume_from}.pt"
            )
        if pretrained_ckp is not None:
            # finetune mode
            self.model = load_pretrained_model(self.model,self.accelerator, pretrained_ckp)
            self.accelerator.print(f"Load pretrained checkpoint!")

        self.model, self.opt = self.accelerator.prepare(self.model, self.opt)

        self.input_type = input_type


    @property
    def device(self):
        return self.accelerator.device

    def save(self, milestone):
        if not self.accelerator.is_local_main_process:
            return

        data = {
            "step": self.step,
            "model": self.accelerator.get_state_dict(self.model),
            "opt": self.opt.state_dict(),
            "ema": self.ema.state_dict() if self.ema is not None else None,
            "lr_scheduler": self.lr_scheduler.state_dict(),
            "scaler": (
                self.accelerator.scaler.state_dict()
                if exists(self.accelerator.scaler)
                else None
            ),
        }

        torch.save(data, os.path.join(self.ckp_results_folder, f"model-{milestone}.pt"))

    def train(self):
        accelerator = self.accelerator
        device = accelerator.device

        with tqdm(
            initial=self.step,
            total=self.train_num_steps,
            disable=not accelerator.is_main_process,
        ) as pbar:

            while self.step < self.train_num_steps:
                self.model.train()

                total_loss = 0.0
                for _ in range(self.gradient_accumulate_every):
                    data = next(self.dl).to(device)
                    noise = torch.randn_like(data).to(device)
                    with self.accelerator.autocast():
                        timesteps = torch.randint(0,self.noise_scheduler.num_train_timesteps,(data.shape[0],),
                            device=data.device,
                        ).long()
                        noisy_images = self.noise_scheduler.add_noise(data, noise, timesteps)
                        noise_pred = self.model(noisy_images, timesteps, return_dict=True).sample

                        loss = self.criterion(noise_pred, noise)
                        loss = loss / self.gradient_accumulate_every
                        total_loss += loss.item()

                    self.accelerator.backward(loss)

                pbar.set_description(f"loss: {loss:.4f}")
                self.accelerator.log({"loss": total_loss,},step=self.step)

                accelerator.wait_for_everyone()
                accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                self.opt.step()
                self.opt.zero_grad()

                accelerator.wait_for_everyone()

                self.step += 1
                if accelerator.is_main_process:
                    if self.ema is not None:
                        self.ema.update()

                    # save sample images and model
                    if self.step != 0 and divisible_by(self.step, self.save_and_sample_every):
                        eval_model = (
                            self.ema.ema_model.eval()
                            if self.ema is not None
                            else self.model.eval()
                        )
                        pipeline = DDIMPipeline(
                            unet=self.accelerator.unwrap_model(eval_model),
                            scheduler=self.noise_scheduler,
                        )
                        with torch.inference_mode():
                            milestone = self.step // self.save_and_sample_every
                            batches = num_to_groups(self.num_samples, self.batch_size)
                            all_images_list = list(
                                map(
                                    lambda n: torch.from_numpy(pipeline(n,generator=torch.manual_seed(random.randint(0, 1000000)),output_type="np",).images),
                                    batches,
                                )
                            )

                        utils.save_image(torch.cat(all_images_list).permute(0,3,1,2),os.path.join(self.samples_results_folder, f"sample-{milestone}.png"),nrow=4),
                        self.save(milestone)

                pbar.update(1)

        accelerator.print("training complete")
