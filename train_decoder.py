import os
import torch
import wandb
from torch.utils.data import DataLoader
from models       import get_decoder, build_modulation_layer
from loss         import HybridLoss
from simulator    import build_simulator, compute_stim_weights
from utils        import generate_phosphenes, visualize_training_sample
from data.local_datasets import LaPaDataset
import matplotlib.pyplot as plt

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled   = False

def train(cfg):
    # 1. W&B setup
    wandb.init(
        project=cfg['general']['project_name'],
        entity= cfg['general']['entity'],
        name=   cfg['general']['run_name'],
        config=cfg
    )
    
    # 2. Data
    dataset    = LaPaDataset(cfg)
    dataloader = DataLoader(
        dataset,
        batch_size=cfg['train']['batch_size'],
        shuffle=True,
        num_workers=cfg['train']['num_workers'],
        drop_last=True
    )
    
    # 3. Model / Loss / Optimizer
    device  = torch.device(cfg['general']['device'])
    decoder = get_decoder(cfg).to(device)
    wandb.watch(decoder, log="all", log_freq=cfg['general']['model_log_freq'])
    
    criterion  = HybridLoss(
        alpha=cfg['loss']['alpha'],
        beta= cfg['loss']['beta']
    )
    optimizer  = torch.optim.Adam(
        decoder.parameters(),
        # convert to float
        lr=float(cfg['train']['lr']),
        weight_decay=float(cfg['train']['weight_decay'])
    )
    
    # 4. (Optional) stimulator and weights here...
    simulator = build_simulator(cfg)
    stim_weights = compute_stim_weights(simulator, cfg)

    if cfg['dataset']['processing'] == 'LoG':
        modulation_layer = build_modulation_layer(cfg, simulator)
    else:
        modulation_layer = None

    torch.use_deterministic_algorithms(False)
    
    epochs        = cfg['train']['epochs']
    save_every    = cfg['general'].get('save_every', 1)  # epochs between saves
    out_dir       = cfg['general']['save_path']
    os.makedirs(out_dir, exist_ok=True)
    
    global_step = 0
    for epoch in range(1, epochs+1):
        decoder.train()
        epoch_loss = 0.0
        
        for batch_idx, batch in enumerate(dataloader, start=1):
            images = batch['image'].to(device)
            
            # 5. Preprocess → phosphene inputs
            stimulus, phos = generate_phosphenes(
                batch, simulator, stim_weights, cfg, modulation_layer
            )
            phos = phos.to(device)
            
            # 6. Forward + loss
            outputs = decoder(phos)
            targets = images.mean(dim=1, keepdim=True)  # grayscale
            loss    = criterion(outputs, targets.to(device))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss   += loss.item()
            global_step  += 1
            
            # 7. Log step metrics
            wandb.log({
                "train/loss": loss.item(),
                "epoch":      epoch,
                "step":       global_step
            }, step=global_step)
            
            # 8. (Optional) visualize occasionally
            if global_step % cfg['general']['model_log_freq'] == 0:
                viz = visualize_training_sample(
                    batch=batch,
                    stimulus=stimulus,
                    phosphene_inputs=phos,
                    reconstructions=outputs,
                    losses=None,
                    epoch=epoch,
                    step=batch_idx
                )
                wandb.log({"train/sample": wandb.Image(viz)}, step=global_step)
                plt.close(viz)
        
        avg_loss = epoch_loss / len(dataloader)
        wandb.log({"train/epoch_loss": avg_loss}, step=global_step)
        print(f"[Epoch {epoch}/{epochs}] avg_loss: {avg_loss:.4f}")
        
        # 9. Save checkpoint
        if epoch % save_every == 0 or epoch == epochs:
            ckpt_path = os.path.join(out_dir, f"decoder_epoch{epoch:02d}.pt")
            torch.save(decoder.state_dict(), ckpt_path)
            # Upload to W&B
            wandb.save(ckpt_path)
    
    wandb.finish()

if __name__ == "__main__":
    import yaml
    import argparse

    parser = argparse.ArgumentParser(
        description="Train the phosphene‐decoder with a given YAML config"
    )
    parser.add_argument(
        "config",
        nargs="?",
        default="config/decoder_exp1.yaml",
        help="Path to YAML config file (default: config/decoder_exp1.yaml)"
    )
    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as fp:
        cfg = yaml.safe_load(fp)

    # Kick off training
    train(cfg)

