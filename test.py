import os
import time
import argparse
import pandas as pd
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from local_datasets import get_dataset
from models import get_decoder, build_modulation_layer
from simulator import build_simulator, compute_stim_weights
from utils import generate_phosphenes
from pytorch_msssim import ssim as ssim_metric
import piq


def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate a trained phosphene decoder on the test set'
    )
    parser.add_argument(
        'config',
        help='Path to the YAML config file (including evaluate section)'
    )
    return parser.parse_args()


def build_metrics(metric_names, device='cuda'):
    """
    Instantiate metric functions based on names.
    Returns a dict of callables: metric -> fn(pred, target) -> float
    """
    metrics = {}
    # MSE
    if 'mse' in metric_names:
        metrics['mse'] = lambda x, y: F.mse_loss(x, y, reduction='mean').item()
    # SSIM
    if 'ssim' in metric_names:
        # data_range=1.0 for normalized [0,1]
        metrics['ssim'] = lambda x, y: ssim_metric(x, y, data_range=1.0, size_average=True).item()
    # VGG perceptual
    if 'vgg_perceptual' in metric_names:
        vgg = piq.VGGLoss(reduction='mean', data_range=1.0).to(device)
        metrics['vgg_perceptual'] = lambda x, y: vgg(x, y).item()
    # LPIPS
    if 'lpips' in metric_names:
        lpips_fn = piq.LPIPS(reduction='mean', data_range=1.0).to(device)
        metrics['lpips'] = lambda x, y: lpips_fn(x, y).item()
    return metrics


def main():
    args = parse_args()
    # 1. Load and merge config
    exp_cfg = OmegaConf.load(args.config)
    base_cfg = OmegaConf.load(exp_cfg['general']['base_path'])
    cfg = OmegaConf.merge(base_cfg, exp_cfg)
    # Convert to dict for simple indexing
    cfg = OmegaConf.to_container(cfg, resolve=True)

    # 2. Setup device and seeds
    device = torch.device(cfg['general']['device'])
    torch.manual_seed(0)

    # 3. Prepare test dataset & loader
    test_ds = get_dataset(cfg, split=cfg['evaluate']['split'])
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg['evaluate']['batch_size'],
        shuffle=False,
        num_workers=cfg['evaluate']['num_workers'],
        drop_last=False
    )

    # 4. Build model and load checkpoint
    decoder = get_decoder(cfg).to(device)
    ckpt_path = cfg['evaluate']['model_ckpt']
    state = torch.load(ckpt_path, map_location=device)
    decoder.load_state_dict(state)
    decoder.eval()

    # 5. Build simulator and modulation (for LoG)
    simulator = build_simulator(cfg)
    stim_weights = compute_stim_weights(simulator, cfg)
    if cfg['dataset']['processing'] == 'LoG':
        modulation_layer = build_modulation_layer(cfg, simulator)
    else:
        modulation_layer = None

    # 6. Instantiate metrics
    metric_names = cfg['evaluate']['metrics']
    metrics_fns = build_metrics(metric_names)

    # 7. Evaluation loop
    records = []
    total_time = 0.0
    total_images = 0

    with torch.no_grad():
        for batch in test_loader:
            imgs = batch['image'].to(device)
            # generate phosphenes
            _, phos = generate_phosphenes(batch, simulator, stim_weights, cfg, modulation_layer)
            phos = phos.to(device)

            # time inference
            start = time.perf_counter()
            outs = decoder(phos)
            elapsed = time.perf_counter() - start

            bs = imgs.size(0)
            total_time += elapsed
            total_images += bs

            # compute targets (grayscale mean)
            targets = imgs.mean(dim=1, keepdim=True)

            # per-sample metrics
            for i in range(bs):
                rec = {'idx': total_images - bs + i}
                pred_i = outs[i:i+1]
                tgt_i = targets[i:i+1]
                for name, fn in metrics_fns.items():
                    rec[name] = fn(pred_i, tgt_i)
                records.append(rec)

    # 8. Save per-image CSV
    out_dir = cfg['evaluate']['output']['results_dir']
    os.makedirs(out_dir, exist_ok=True)
    per_csv = cfg['evaluate']['output']['per_image_csv']
    df = pd.DataFrame.from_records(records)
    df.to_csv(os.path.join(out_dir, per_csv), index=False)

    # 9. Compute summary (mean + std)
    summary = df[metric_names].agg(['mean', 'std']).transpose().reset_index()
    summary.columns = ['metric', 'mean', 'std']
    summary_csv = cfg['evaluate']['output']['summary_csv']
    summary.to_csv(os.path.join(out_dir, summary_csv), index=False)

    # 10. Print overall speed
    if cfg['evaluate']['timing']['measure_inference']:
        imgs_per_s = total_images / total_time
        print(f"Inference speed: {imgs_per_s:.1f} img/s over {total_images} images")

    print(f"Per-image metrics saved to: {os.path.join(out_dir, per_csv)}")
    print(f"Summary metrics saved to:   {os.path.join(out_dir, summary_csv)}")


if __name__ == '__main__':
    main()
