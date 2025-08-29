#!/usr/bin/env python3
import os, sys, json, time, math, argparse, copy, hashlib
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as T
from torchvision.datasets import ImageFolder

import seaborn as sns
import matplotlib.pyplot as plt
import torchvision as tv

import pickle
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# optional COCO support, guarded
try:
    from torchvision.datasets import CocoDetection
    HAS_COCO = True
except Exception:
    HAS_COCO = False

import matplotlib
matplotlib.use("Agg")  # headless

# # local imports from your repo layout
# sys.path.append('./..')
# sys.path.append('./../..')

from dynaphos import utils as dutils
from dynaphos.simulator import GaussianSimulator as PhospheneSimulator
from phosphene.uniformity import DynamicAmplitudeNormalizer
from phosphene.density import VisualFieldMapper
from components.SeparableModulated2d import SeparableModulatedConv2d
from utils import robust_percentile_normalization, shift_stimulus_to_phosphene_centroid
sys.path.append('./notebook')

# ---------------- args and config ----------------
def parse_args():
    ap = argparse.ArgumentParser(description="RDM and RSA evaluation for SCAPE and baselines")
    ap.add_argument("--config", required=True, help="Path to JSON or YAML config file")
    return ap.parse_args()

def load_cfg(path):
    if path.endswith(".json"):
        import json, yaml as _yaml
        with open(path, "r") as f:
            data = json.load(f)
        return data
    else:
        import yaml
        with open(path, "r") as f:
            return yaml.safe_load(f)

# ---------------- utils ----------------
def seed_all(seed):
    torch.manual_seed(seed)
    np.random.seed(seed % (2**32 - 1))

def normalize01(t):
    tmin = float(t.min())
    tmax = float(t.max())
    if tmax - tmin < 1e-8:
        return torch.zeros_like(t)
    return (t - tmin) / (tmax - tmin)

def dilation3x3(x):
    return F.max_pool2d(x, kernel_size=3, stride=1, padding=1)

def dilation5x5(x):
    return F.max_pool2d(x, kernel_size=5, stride=1, padding=2)

def canny_edges_from_numpy(np_gray, low=100, high=200, dilate=True):
    import cv2
    u8 = (np_gray * 255.0).astype(np.uint8)
    ce = cv2.Canny(u8, low, high).astype(np.float32) / 255.0
    t = torch.from_numpy(ce).view(1,1,ce.shape[0], ce.shape[1])
    if dilate:
        t = dilation3x3(t)
    return t.squeeze(0).squeeze(0).numpy().astype(np.float32)

def perlin_like(H, W, tiles_x=8, tiles_y=8, seed=0):
    rng = np.random.default_rng(seed)
    # simple smooth noise, cheap stand-in
    grid_x = np.linspace(0, 2*math.pi*tiles_x, W)
    grid_y = np.linspace(0, 2*math.pi*tiles_y, H)
    gx, gy = np.meshgrid(grid_x, grid_y)
    phase = rng.uniform(0, 2*math.pi, size=(3,))
    arr = 0.5*np.sin(gx + phase[0]) + 0.3*np.sin(gy + phase[1]) + 0.2*np.sin(gx+gy+phase[2])
    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
    return arr.astype(np.float32)

def avgpool_numpy_imgs(X_np_flat, H, W, k):
    X = torch.from_numpy(X_np_flat.astype(np.float32))
    X = X.view(X.shape[0], 1, H, W)
    Xp = F.avg_pool2d(X, kernel_size=k, stride=k)
    N, _, Hp, Wp = Xp.shape
    return Xp.squeeze(1).contiguous().view(N, Hp*Wp).numpy(), Hp, Wp

def random_projection(X_np, out_dim=1024, seed=123):
    rng = np.random.default_rng(seed)
    N, D = X_np.shape
    W = rng.standard_normal((D, out_dim)).astype(np.float32) / np.sqrt(D)
    return X_np.astype(np.float32) @ W

@torch.no_grad()
def corr_rdm_gpu(X_np, device="cuda"):
    X = torch.from_numpy(X_np.astype(np.float32)).to(device)
    mu = X.mean(dim=1, keepdim=True)
    sd = X.std(dim=1, keepdim=True)
    sd = torch.where(sd < 1e-8, torch.ones_like(sd), sd)
    Z = (X - mu) / sd
    denom = max(Z.shape[1] - 1, 1)
    C = (Z @ Z.T) / denom
    C = torch.clamp(C, -1.0, 1.0)
    RDM = (1.0 - C).float().cpu().numpy()
    return RDM

from scipy.stats import spearmanr

def bootstrap_rsa(rdm_ref, rdm_m, B=5000, seed=1234):
    rng = np.random.default_rng(seed)
    N = rdm_ref.shape[0]
    out = np.empty(B, np.float32)
    for b in range(B):
        sub = rng.integers(0, N, size=N)
        rr = rdm_ref[np.ix_(sub, sub)]
        rm = rdm_m[np.ix_(sub, sub)]
        tri = np.triu_indices(len(sub), 1)
        out[b] = spearmanr(rr[tri], rm[tri]).statistic
    return out

def permute_rsa_p(rdm_ref, rdm_m, n_perm=10000, seed=1234):
    rng = np.random.default_rng(seed)
    N = rdm_ref.shape[0]
    tri = np.triu_indices(N, 1)
    obs = spearmanr(rdm_ref[tri], rdm_m[tri]).statistic
    idx = np.arange(N)
    null = np.empty(n_perm, np.float32)
    for k in range(n_perm):
        p = rng.permutation(idx)
        rm = rdm_m[p][:, p]
        null[k] = spearmanr(rdm_ref[tri], rm[tri]).statistic
    pval = (np.sum(null >= obs) + 1) / (n_perm + 1)
    return obs, pval

def holm_correction(pvals):
    pvals = np.asarray(pvals, float)
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty_like(pvals)
    prev = 0.0
    for k, i in enumerate(order, start=1):
        adj[i] = max(prev, (m - k + 1) * pvals[i])
        prev = adj[i]
    return np.minimum(adj, 1.0)

def build_vgg_until(layer_name="conv3_3", device="cuda"):
    idx_map = {"conv1_2": 4, "conv2_2": 9, "conv3_3": 16, "conv4_3": 23, "conv5_3": 30}
    idx = idx_map[layer_name]
    vgg = tv.models.vgg16(weights=tv.models.VGG16_Weights.IMAGENET1K_FEATURES).features[:idx]
    vgg.eval().to(device)
    for p in vgg.parameters():
        p.requires_grad = False
    return vgg

@torch.no_grad()
def vgg_features_from_grayscale(x_hw, vgg, batch=64, device="cuda"):
    """
    x_hw: torch [N,H,W] in [0,1]
    returns: numpy [N,C] after global average pooling
    """
    N, H, W = x_hw.shape

    # build mean/std on the same device we will use
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1,3,1,1)
    std  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1,3,1,1)

    feats = []
    for s in range(0, N, batch):
        e = min(s + batch, N)
        # to 3-channel and move to device BEFORE normalization
        x = x_hw[s:e].unsqueeze(1).expand(-1, 3, -1, -1).to(device)
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        x = (x - mean) / std
        y = vgg(x)                              # [b,C,Hf,Wf]
        y = F.adaptive_avg_pool2d(y, 1).squeeze(-1).squeeze(-1)  # [b,C]
        feats.append(y.detach().cpu())
    return torch.cat(feats, dim=0).numpy().astype(np.float32)


@torch.no_grad()
def corr_rdm_gpu_from_vectors(X_np, device="cuda"):
    X = torch.from_numpy(X_np.astype(np.float32)).to(device)
    mu = X.mean(dim=1, keepdim=True)
    sd = X.std(dim=1, keepdim=True)
    sd = torch.where(sd < 1e-8, torch.ones_like(sd), sd)
    Z = (X - mu) / sd
    denom = max(Z.shape[1] - 1, 1)
    C = (Z @ Z.T) / denom
    C = torch.clamp(C, -1.0, 1.0)
    return (1.0 - C).float().cpu().numpy()

# --- VGG multi-layer helpers ---

def get_vgg_slice(layer_name: str, device="cuda"):
    idx_map = {"conv1_2": 4, "conv2_2": 9, "conv3_3": 16, "conv4_3": 23, "conv5_3": 30}
    if layer_name not in idx_map:
        raise ValueError(f"Unknown VGG layer {layer_name}")
    vgg = tv.models.vgg16(weights=tv.models.VGG16_Weights.IMAGENET1K_FEATURES).features[:idx_map[layer_name]]
    vgg.eval().to(device)
    for p in vgg.parameters():
        p.requires_grad = False
    return vgg

@torch.no_grad()
def vgg_multi_features(x_hw: torch.Tensor, layers, batch=64, device="cuda"):
    """
    x_hw: torch [N,H,W] in [0,1]
    returns: dict layer -> numpy [N,C_layer] (GAP pooled)
    """
    feats = {}
    for ln in layers:
        vgg = get_vgg_slice(ln, device=device)
        feats[ln] = vgg_features_from_grayscale(x_hw, vgg, batch=batch, device=device)
    return feats

def combine_rdms(rdm_dict: dict, weights=None, mode="mean"):
    """
    rdm_dict: {layer_name: RDM [N,N]}
    weights: list same length as rdm_dict values. If None, uniform.
    mode: "mean" only here. (Feature concat handled in main.)
    """
    names = list(rdm_dict.keys())
    L = len(names)
    if weights is None:
        w = np.ones(L, np.float32) / L
    else:
        w = np.array(weights, np.float32)
        w = w / np.sum(w)
    if mode != "mean":
        raise ValueError("combine_rdms supports mode='mean' only")
    acc = None
    for i, ln in enumerate(names):
        R = rdm_dict[ln].astype(np.float32)
        acc = (w[i] * R) if acc is None else acc + w[i] * R
    return acc



# ---------------- data ----------------
def make_loader(cfg):
    t = T.Compose([
        T.Resize((256, 256)),
        T.ToTensor(),
        T.Lambda(lambda img: (img * torch.tensor([0.2126,0.7152,0.0722]).view(3,1,1)).sum(dim=0, keepdim=True))
    ])
    dtype = cfg["data"]["type"]
    if dtype == "ImageFolder":
        ds = ImageFolder(cfg["data"]["root"], transform=t)
        n = len(ds)
        if "num_eval" in cfg["data"] and cfg["data"]["num_eval"] > 0:
            n = min(n, int(cfg["data"]["num_eval"]))
            ds = Subset(ds, range(n))
        return DataLoader(ds, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)
    elif dtype == "COCO":
        assert HAS_COCO, "torchvision COCO not available"
        img_dir = os.path.join(cfg["data"]["root"], "val2017")
        ann_file = os.path.join(cfg["data"]["root"], "annotations", "instances_val2017.json")
        ds = CocoDetection(root=img_dir, annFile=ann_file, transform=t)
        n = len(ds)
        if "num_eval" in cfg["data"] and cfg["data"]["num_eval"] > 0:
            n = min(n, int(cfg["data"]["num_eval"]))
            ds = Subset(ds, range(n))
        return DataLoader(ds, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)
    else:
        raise ValueError(f"Unknown data.type {dtype}")

# ---------------- simulator and maps ----------------
def load_simulator_and_weights(cfg):
    params = dutils.load_params(cfg["sim"]["params_path"])
    with open(cfg["sim"]["electrodes_path"], "rb") as f:
        coords = pickle.load(f)  # noqa

    view_angle_dict = {1024: 16,
                       4224: 25,
                       94: 0.4,
                       320: 16} 
    n_phosphenes = len(coords)
    print('n_phosphenes sim:', n_phosphenes)
    params["run"]["view_angle"] = view_angle_dict.get(n_phosphenes, 16)
    cfg["sim"]["view_angle"] = params["run"]["view_angle"]

    print('view angle:', params["run"]["view_angle"])
    simulator = PhospheneSimulator(params, coords)
    amplitude = params["sampling"]["stimulus_scale"]

    stim_init = amplitude * torch.ones(simulator.num_phosphenes, device="cuda")
    cache_tag = hashlib.md5((cfg["sim"]["electrodes_path"] + str(cfg["sim"]["view_angle"])).encode()).hexdigest()[:12]
    cache_path = os.path.join(cfg["out_dir"], f"stim_weights_{cache_tag}.pt")
    # if os.path.exists(cache_path):
    #     weights = torch.load(cache_path, map_location="cuda")
    # else:
    #     normalizer = DynamicAmplitudeNormalizer(
    #         simulator=simulator,
    #         base_size=3,
    #         scale=0.1,
    #         A_min=0,
    #         A_max=amplitude,
    #         learning_rate=0.005,
    #         steps=1000,
    #         target=None
    #     )
    #     _ = normalizer.run(stim_init, verbose=False)
    #     weights = normalizer.weights.detach()
    weights = torch.ones_like(stim_init)
        # os.makedirs(cfg["out_dir"], exist_ok=True)
        # torch.save(weights, cache_path)
    return simulator, weights

def build_sigma_maps(simulator, cfg):
    mapper = VisualFieldMapper(simulator=simulator)
    n = simulator.num_phosphenes
    print('n phosphenes: ', n)
    density_kde = mapper.build_density_map_kde(k=6, alpha=1.0, total_phosphenes=n)
    sigma_kde_pix = mapper.build_sigma_map_from_density(density_kde, space="pixel")
    sigma_map_tensor = torch.tensor(sigma_kde_pix).float().cuda().detach()
    mean_sigma = sigma_map_tensor.mean()
    # sigma_mean_map = torch.ones_like(sigma_map_tensor) * mean_sigma
    sigma_fixed_small = torch.ones_like(sigma_map_tensor) * 3.0
    # centroid for shift
    x_center, y_center = mapper.centroid_of_density(density_kde)
    return {
        "adaptive": sigma_map_tensor,
        # "mean": sigma_mean_map,
        "small": sigma_fixed_small
    }, (x_center, y_center)

# ---------------- stimulation and simulation ----------------
def make_stim_phos(simulator, stim_img, amplitude, threshold, stim_weights, normalization=True):
    simulator.reset()
    elec = simulator.sample_stimulus(stim_img, rescale=True)
    if normalization:
        elec = robust_percentile_normalization(
            elec, amplitude, threshold, low_perc=5, high_perc=90, gamma=1/3
        )
    else:
        elec = elec * amplitude
    elec = elec * stim_weights
    phos = simulator(elec)
    if phos.dim() == 2:
        phos = phos.unsqueeze(0).unsqueeze(0)
    elif phos.dim() == 3:
        phos = phos.unsqueeze(0)
    return phos

# ---------------- main job ----------------
import pickle
def main():
    args = parse_args()
    cfg = load_cfg(args.config)

    if cfg.get("save_config_copy", False):
        import json
        os.makedirs(cfg["out_dir"], exist_ok=True)
        with open(os.path.join(cfg["out_dir"], "config_used.json"), "w") as f:
            json.dump(cfg, f, indent=2)


    os.makedirs(cfg["out_dir"], exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed_all(cfg.get("seed", 1234))
    torch.use_deterministic_algorithms(False)

    # simulator and weights
    simulator, stim_weights = load_simulator_and_weights(cfg)
    amplitude = simulator.params["sampling"]["stimulus_scale"]
    threshold = simulator.params["thresholding"]["rheobase"]

    # sigma maps and centroid
    sigma_maps, centroid_deg = build_sigma_maps(simulator, cfg)

    # modulated filters
    mod_adapt = SeparableModulatedConv2d(in_channels=1, sigma_map=sigma_maps["adaptive"]).cuda().eval()
    # mod_mean  = SeparableModulatedConv2d(in_channels=1, sigma_map=sigma_maps["mean"]).cuda().eval()
    mod_small = SeparableModulatedConv2d(in_channels=1, sigma_map=sigma_maps["small"]).cuda().eval()

    # data
    loader = make_loader(cfg)
    torch.use_deterministic_algorithms(False)
    fov_deg = simulator.params["run"]["view_angle"]

    # feature collectors
    methods = cfg["methods"]
    pooled_orig = []
    pooled = {m: [] for m in methods}

    # loop
    for idx, batch in enumerate(loader):
        img = batch[0].cuda()  # [1,1,H,W] due to transform
        # shift to centroid
        np_img = img[0,0].cpu().numpy()
        # shifted = shift_stimulus_to_phosphene_centroid(np_img, centroid_deg, fov=fov_deg, mode="constant", cval=0.0)
        shifted = np_img  # do not shift for now
        img = torch.tensor(shifted, dtype=torch.float32).unsqueeze(0).unsqueeze(0).cuda()

        # construct stimuli per method
        stim_dict = {}
        if "grayscale" in methods:
            stim_dict["grayscale"] = img

        if "DoG_adaptive" in methods:
            dog = mod_adapt(img).detach().clamp_min(0.0)
            stim_dict["DoG_adaptive"] = normalize01(dog)

        # if "DoG_fixed_mean" in methods:
        #     dogm = mod_mean(img).detach().clamp_min(0.0)
        #     stim_dict["DoG_fixed_mean"] = normalize01(dogm)

        if "DoG_fixed_small" in methods:
            dogs = mod_small(img).detach().clamp_min(0.0)
            stim_dict["DoG_fixed_small"] = normalize01(dogs)

        if "canny_edge" in methods:
            ce_np = canny_edges_from_numpy(img[0,0].cpu().numpy(), low=100, high=200, dilate=True)
            stim_dict["canny_edge"] = torch.from_numpy(ce_np).unsqueeze(0).unsqueeze(0).cuda()

        if "random" in methods:
            H, W = img.shape[-2:]
            rp = perlin_like(H, W, tiles_x=8, tiles_y=8, seed=idx)
            stim_dict["random"] = torch.from_numpy(rp).unsqueeze(0).unsqueeze(0).cuda()

        # simulate
        phos_dict = {}
        for m in methods:
            phos = make_stim_phos(simulator, stim_dict[m], amplitude, threshold, stim_weights, normalization=True)
            phos_dict[m] = phos

        # collect pooled vectors for RDM
        H, W = img.shape[-2:]
        # original is the raw input image
        orig_vec = img[0,0].detach().cpu().numpy().astype(np.float32).ravel()
        pooled_orig.append(orig_vec)

        for m in methods:
            ph = phos_dict[m][0,0].detach().cpu().numpy().astype(np.float32).ravel()
            pooled[m].append(ph)

    # to arrays
    pooled_orig = np.stack(pooled_orig, 0)
    for m in methods:
        pooled[m] = np.stack(pooled[m], 0)

    # pool factor and random projection
    H, W = cfg["image_hw"]
    orig_p, Hp, Wp = avgpool_numpy_imgs(pooled_orig, H, W, k=cfg["rsa"]["pool_k"])
    meth_p = {m: avgpool_numpy_imgs(pooled[m], H, W, k=cfg["rsa"]["pool_k"])[0] for m in methods}

    if cfg["rsa"]["use_rp"]:
        orig_p = random_projection(orig_p, out_dim=cfg["rsa"]["rp_dim"], seed=cfg["seed"])
        meth_p = {m: random_projection(meth_p[m], out_dim=cfg["rsa"]["rp_dim"], seed=cfg["seed"]) for m in methods}

    # save features cache for reuse
    np.savez_compressed(os.path.join(cfg["out_dir"], "features.npz"),
                        orig=orig_p, **{f"m_{k}": v for k,v in meth_p.items()},
                        Hp=Hp, Wp=Wp, methods=np.array(methods, dtype=object))

    # RDMs
    rdm_orig = corr_rdm_gpu(orig_p, device=device)
    rdm_meth = {m: corr_rdm_gpu(meth_p[m], device=device) for m in methods}

    # RSA ranking
    tri = np.triu_indices(rdm_orig.shape[0], 1)
    rows = []
    for m in methods:
        rho = spearmanr(rdm_orig[tri], rdm_meth[m][tri]).statistic
        boot = bootstrap_rsa(rdm_orig, rdm_meth[m], B=cfg["rsa"]["bootstraps"], seed=cfg["seed"])
        lo, hi = np.percentile(boot, [2.5, 97.5])
        _, pval = permute_rsa_p(rdm_orig, rdm_meth[m], n_perm=cfg["rsa"]["permutations"], seed=cfg["seed"])
        rows.append({"method": m, "rho": float(rho), "lo": float(lo), "hi": float(hi), "p_perm": float(pval)})
    import pandas as pd
    df = pd.DataFrame(rows).sort_values("rho", ascending=False).reset_index(drop=True)
    df["p_holm"] = holm_correction(df["p_perm"].values)
    df.to_csv(os.path.join(cfg["out_dir"], "rsa_results.csv"), index=False)

    # simple second order matrix with labels and heatmap
    names = ["original"] + methods
    tri = np.triu_indices(rdm_orig.shape[0], 1)

    vecs = {"original": rdm_orig[tri]}
    for m in methods:
        vecs[m] = rdm_meth[m][tri]

    S = np.zeros((len(names), len(names)), np.float32)
    for i, ni in enumerate(names):
        for j, nj in enumerate(names):
            S[i, j] = spearmanr(vecs[ni], vecs[nj]).statistic

    # save as .npy for quick programmatic reuse
    np.save(os.path.join(cfg["out_dir"], "second_order.npy"), S)

    # save as labeled CSV
    S_df = pd.DataFrame(S, index=names, columns=names)
    S_df.to_csv(os.path.join(cfg["out_dir"], "second_order.csv"), float_format="%.6f")

    # save a heatmap PNG
    fig, ax = plt.subplots(
        figsize=(max(6.8, 0.8*len(names)), max(5.2, 0.6*len(names))),
        dpi=220
    )
    sns.heatmap(
        S_df, vmin=0, vmax=1, cmap="viridis",
        annot=True, fmt=".2f", cbar=True,
        xticklabels=True, yticklabels=True, ax=ax
    )
    ax.set_title("Second order correlation of RDMs")
    plt.tight_layout()
    fig.savefig(os.path.join(cfg["out_dir"], "second_order_matrix.png"))
    plt.close(fig)


if __name__ == "__main__":
    main()
