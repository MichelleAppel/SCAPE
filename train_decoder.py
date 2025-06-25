import os
import torch
import wandb

import matplotlib.pyplot as plt
from omegaconf import OmegaConf

from torch.utils.data import DataLoader
from models import get_decoder, build_modulation_layer
from loss import get_loss
from simulator import build_simulator, compute_stim_weights
from utils import generate_phosphenes, visualize_training_sample
from local_datasets import get_dataset

# Disable CuDNN optimizations for full determinism if needed
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled = False

# Set seed for reproducibility
torch.manual_seed(0)
torch.cuda.manual_seed(0)

def train(cfg):
    # 1. W&B setup
    wandb_kwargs = {}
    wandb_dir = cfg['general'].get('wandb_dir')
    if wandb_dir:
        os.makedirs(wandb_dir, exist_ok=True)
        wandb_kwargs['dir'] = wandb_dir

    wandb.init(
        project=cfg['general']['project_name'],
        entity=cfg['general']['entity'],
        name=cfg['general']['run_name'],
        config=cfg,
        **wandb_kwargs
    )
    
    # 2. Data loading: dispatch based on config
    train_ds, val_ds = get_dataset(cfg, split='train')
    dataloader = DataLoader(
        train_ds,
        batch_size=cfg['train']['batch_size'],
        shuffle=True,
        num_workers=cfg['train']['num_workers'],
        drop_last=True
    )
    val_dataloader = DataLoader(
        val_ds,
        batch_size=cfg['train']['batch_size'],
        shuffle=False,
        num_workers=cfg['train']['num_workers'],
        drop_last=True
    )
    val_iter = iter(val_dataloader)
    
    # 3. Model / Loss / Optimizer
    device = torch.device(cfg['general']['device'])
    decoder = get_decoder(cfg).to(device)
    wandb.watch(decoder, log="all", log_freq=cfg['general']['model_log_freq'])

    criterion = get_loss(cfg).to(device)
    optimizer = torch.optim.Adam(
        decoder.parameters(),
        lr=float(cfg['train']['lr']),
        weight_decay=float(cfg['train']['weight_decay'])
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg['train']['epochs'],   # one cycle over your full training run
        eta_min=5e-7                    # final minimal LR
    )
    
    # 4. Simulator & stimulus weights
    simulator = build_simulator(cfg)
    stim_weights = compute_stim_weights(simulator, cfg)

    # 5. Optional modulation layer for DoG
    if cfg['dataset']['processing'] == 'DoG':
        modulation_layer = build_modulation_layer(cfg, simulator)
    else:
        modulation_layer = None

    torch.use_deterministic_algorithms(False)
    
    epochs = cfg['train']['epochs']
    save_every = cfg['general'].get('save_every', 1)
    out_dir = cfg['general']['save_path']
    os.makedirs(out_dir, exist_ok=True)

    global_step = 0
    for epoch in range(1, epochs + 1):
        decoder.train()
        epoch_loss = 0.0
        
        for batch_idx, batch in enumerate(dataloader, start=1):
            targets = batch['image'].to(device)

            # Generate phosphene inputs
            _, phos = generate_phosphenes(
                batch, simulator, stim_weights, cfg, modulation_layer
            )
            phos = phos.to(device)

            # Forward + loss
            outputs = decoder(phos)
            loss = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            global_step += 1

            # Log train loss + learning rate each step
            current_lr = optimizer.param_groups[0]['lr']
            wandb.log({
                'train/loss': loss.item(),
                'train/lr': current_lr,
                'epoch': epoch,
                'step': global_step
            }, step=global_step)

            # Periodic validation logging
            if global_step % cfg['general']['model_log_freq'] == 0:
                try:
                    val_batch = next(val_iter)
                except StopIteration:
                    val_iter = iter(val_dataloader)
                    val_batch = next(val_iter)

                with torch.no_grad():
                    val_images = val_batch['image'].to(device)
                    val_stimulus, val_phos = generate_phosphenes(
                        val_batch, simulator, stim_weights, cfg, modulation_layer
                    )
                    val_phos = val_phos.to(device)
                    val_outputs = decoder(val_phos)
                    val_targets = val_images.mean(dim=1, keepdim=True)
                    val_loss = criterion(val_outputs, val_targets)

                wandb.log({
                    'val/loss': val_loss.item(),
                    'epoch': epoch,
                    'step': global_step
                }, step=global_step)

                viz = visualize_training_sample(
                    batch=val_batch,
                    stimulus=val_stimulus,
                    phosphene_inputs=val_phos,
                    reconstructions=val_outputs,
                    epoch=epoch,
                    step=batch_idx
                )
                wandb.log({'val/sample': wandb.Image(viz)}, step=global_step)
                plt.close(viz)

        avg_loss = epoch_loss / len(dataloader)
        wandb.log({'train/epoch_loss': avg_loss}, step=global_step)
        print(f"[Epoch {epoch}/{epochs}] avg_loss: {avg_loss:.4f}")

        decoder.eval()
        val_loss_sum = 0.0
        with torch.no_grad():
            for vb in val_dataloader:
                imgs = vb['image'].to(device)
                _, phs = generate_phosphenes(vb, simulator, stim_weights, cfg, modulation_layer)
                phs = phs.to(device)
                outs = decoder(phs)
                tgts = imgs.mean(dim=1, keepdim=True)
                val_loss_sum += criterion(outs, tgts).item()

        avg_val_loss = val_loss_sum / len(val_dataloader)
        wandb.log({'val/epoch_loss': avg_val_loss}, step=global_step)
        print(f"[Epoch {epoch}/{epochs}] val_loss:   {avg_val_loss:.4f}")

        if epoch % save_every == 0 or epoch == epochs:
            ckpt_path = os.path.join(out_dir, f"decoder_epoch{epoch:02d}.pt")
            torch.save(decoder.state_dict(), ckpt_path)
            wandb.save(ckpt_path)
        scheduler.step()
    
    wandb.finish()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Train the phosphene decoder with a YAML config'
    )
    parser.add_argument(
        'config',
        nargs='?',
        help='Path to experiment YAML overrides'
    )
    args = parser.parse_args()

    # Load experiment overrides and base config from general.base_path
    exp_cfg = OmegaConf.load(args.config)
    base_path = exp_cfg['general']['base_path']
    base_cfg = OmegaConf.load(base_path)
    cfg = OmegaConf.merge(base_cfg, exp_cfg)
    cfg = OmegaConf.to_container(cfg, resolve=True)

    train(cfg)
