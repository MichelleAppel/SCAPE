import os
import time
import argparse
import copy
import pandas as pd
import torch
from tqdm import tqdm
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt

from local_datasets import get_dataset
from models import get_decoder, build_modulation_layer
from simulator import build_simulator, compute_stim_weights
from utils import generate_phosphenes, visualize_training_sample
from loss import get_loss


def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate a trained phosphene decoder on the test set using loss modules as metrics'
    )
    parser.add_argument(
        'config',
        help='Path to the YAML config file (including evaluate section)'
    )
    return parser.parse_args()


def build_metric_modules(cfg, metric_names, device):
    """
    Instantiate loss modules for each metric name by overriding cfg['loss']['loss_function']
    Returns a dict mapping metric_name -> loss_module
    """
    modules = {}
    name_map = {
        'mse': 'MSE',
        'ssim': 'SSIMLoss',
        'lpips': 'LPIPSLoss',
        'vgg_perceptual': 'VGGPerceptualLoss'
    }
    for m in metric_names:
        loss_name = name_map.get(m, m)
        cfg_m = copy.deepcopy(cfg)
        cfg_m.setdefault('loss', {})
        cfg_m['loss']['loss_function'] = loss_name
        module = get_loss(cfg_m).to(device)
        module.eval()
        modules[m] = module
    return modules


def main():
    args = parse_args()
    # Load and merge config
    exp_cfg = OmegaConf.load(args.config)
    base_cfg = OmegaConf.load(exp_cfg['general']['base_path'])
    cfg = OmegaConf.merge(base_cfg, exp_cfg)
    cfg = OmegaConf.to_container(cfg, resolve=True)

    # Device and seed
    device = torch.device(cfg['general']['device'])
    torch.manual_seed(0)

    # Dataset and subset for evaluation count
    full_ds = get_dataset(cfg, split=cfg['evaluate']['split'])
    num_eval = cfg['evaluate'].get('num_images_eval', len(full_ds))
    num_eval = min(num_eval, len(full_ds))
    eval_ds = Subset(full_ds, range(num_eval))
    test_loader = DataLoader(
        eval_ds,
        batch_size=cfg['evaluate']['batch_size'],
        shuffle=False,
        num_workers=cfg['evaluate']['num_workers'],
        drop_last=False
    )

    # Load model
    decoder = get_decoder(cfg).to(device)
    ckpt = cfg['evaluate']['model_ckpt']
    decoder.load_state_dict(torch.load(ckpt, map_location=device))
    decoder.eval()

    # Simulator & modulation
    simulator = build_simulator(cfg)
    torch.use_deterministic_algorithms(False)
    stim_weights = compute_stim_weights(simulator, cfg)
    modulation_layer = (build_modulation_layer(cfg, simulator)
                        if cfg['dataset']['processing'] == 'DoG' else None)

    # Metrics
    metric_names = cfg['evaluate']['metrics']
    metric_modules = build_metric_modules(cfg, metric_names, device)

    # Prepare output dirs
    out_dir = cfg['evaluate']['output']['results_dir']
    img_dir = os.path.join(out_dir, 'images')
    os.makedirs(img_dir, exist_ok=True)

    records = []
    total_time = 0.0
    total_images = 0
    plot_limit = cfg['evaluate'].get('num_images_plot', 0)
    plot_count = 0

    # Evaluation loop with progress bar
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Evaluating', unit='batch'):
            imgs = batch['image'].to(device)
            start = time.perf_counter()
            stimulus, phos = generate_phosphenes(
                batch, simulator, stim_weights, cfg, modulation_layer)
            phos = phos.to(device)

            # Inference timing
            outs = decoder(phos)
            elapsed = time.perf_counter() - start

            bs = imgs.size(0)
            total_time += elapsed
            total_images += bs

            targets = imgs.mean(dim=1, keepdim=True)

            # Per-sample handling
            for i in range(bs):
                idx = total_images - bs + i
                rec = {'idx': idx}

                # Save image if under plot limit
                if plot_count < plot_limit:
                    fig = visualize_training_sample(
                        batch={k: v[i:i+1] for k, v in batch.items()},
                        stimulus=stimulus[i:i+1],
                        phosphene_inputs=phos[i:i+1],
                        reconstructions=outs[i:i+1],
                        epoch=0,
                        step=idx
                    )
                    fig.savefig(os.path.join(img_dir, f'sample_{idx:04d}.png'))
                    plt.close(fig)
                    plot_count += 1

                # Compute metrics
                pred_i = outs[i:i+1]
                tgt_i = targets[i:i+1]
                for m, module in metric_modules.items():
                    rec[m] = module(pred_i, tgt_i).item()
                # Inference time per image
                if cfg['evaluate']['timing']['measure_inference']:
                    rec['infer_time_s'] = elapsed / bs

                records.append(rec)

    # Save CSVs
    df = pd.DataFrame(records)
    df.to_csv(os.path.join(out_dir, cfg['evaluate']['output']['per_image_csv']),
              index=False)

    summary_metrics = list(metric_names)
    if cfg['evaluate']['timing']['measure_inference']:
        summary_metrics.append('infer_time_s')
    summary = df[summary_metrics].agg(['mean', 'std']).transpose().reset_index()
    summary.columns = ['metric', 'mean', 'std']
    summary.to_csv(os.path.join(out_dir,
                                 cfg['evaluate']['output']['summary_csv']),
                   index=False)

    # Final report
    if cfg['evaluate']['timing']['measure_inference']:
        speed = total_images / total_time if total_time > 0 else float('nan')
        print(f"Inference speed: {speed:.1f} img/s over {total_images} imgs")
    print(f"Evaluated {total_images} images, plotted {plot_count} samples.")
    print(f"Results saved in: {out_dir}")

if __name__ == '__main__':
    main()