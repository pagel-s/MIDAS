import torch
from argparse import Namespace
import yaml
import sys
import numpy as np  
from pathlib import Path
from rdkit import Chem
import random

import sys
from pathlib import Path
import os

REPO_ROOT = os.path.abspath(os.path.join(os.getcwd(), ".."))

if REPO_ROOT not in sys.path:
    sys.path.append(os.path.join(REPO_ROOT, "MIDAS"))

from lightning_modules import LigandPocketDDPM

def to_ns(d):
    if isinstance(d, dict):
        return Namespace(**{k: to_ns(v) for k, v in d.items()})
    return d

def load_model_using_config(config, ckpt_path=None):
    with open(config, "r") as f:
        cfg = yaml.safe_load(f)
    args = to_ns(cfg)

    # args.wandb_params.mode = "disabled"  # disable wandb in notebook runs
    args.enable_progress_bar = True


    # Required histogram file from the processed dataset
    histogram_file = Path(args.datadir, "size_distribution.npy")
    histogram = np.load(histogram_file).tolist()

    # Build LightningModule with text conditioning
    pl_module = LigandPocketDDPM(
        outdir=Path(args.logdir, args.run_name),
        dataset=args.dataset,
        datadir=args.datadir,
        batch_size=args.batch_size,
        lr=args.lr,
        egnn_params=args.egnn_params,
        diffusion_params=args.diffusion_params,
        num_workers=args.num_workers,
        augment_noise=args.augment_noise,
        augment_rotation=args.augment_rotation,
        clip_grad=args.clip_grad,
        eval_epochs=args.eval_epochs,
        eval_params=args.eval_params,
        visualize_sample_epoch=args.visualize_sample_epoch,
        visualize_chain_epoch=args.visualize_chain_epoch,
        auxiliary_loss=args.auxiliary_loss,
        loss_params=args.loss_params,
        mode=args.mode,
        node_histogram=histogram,
        pocket_representation=args.pocket_representation,
        text_model_name=args.text_model_name,
        text_embeddings_path=args.text_embeddings_path,
        # text_csv=TEXT_CSV,
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if ckpt_path is not None:
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        _, _ = pl_module.load_state_dict(checkpoint['state_dict'], strict=False)
    pl_module = pl_module.to(device)
    pl_module.eval()
    
    return pl_module