## Loading model
import torch
import yaml

from pathlib import Path

from models.model_completion import SnowflakeNet

CKPT_PATHS = [
    (
        Path(__file__).parent / "SnowflakeNet" / "completion" / "ckpt-best-c3d-cd_l2.pth",
        Path(__file__).parent / "completion" / "configs" / "c3d_cd2.yaml"
    ),
    (
        Path(__file__).parent / "SnowflakeNet" / "completion" / "ckpt-best-pcn-cd_l1.pth",
        Path(__file__).parent / "completion" / "configs" / "pcn_cd1.yaml"
    ),
    (
        Path(__file__).parent / "SnowflakeNet" / "completion" / "ckpt-best-pcn-emd.pth",
        Path(__file__).parent / "completion" / "configs" / "pcn_emd.yaml"
    ),
    (
        Path(__file__).parent / "SnowflakeNet" / "completion" / "ckpt-best-shapenet34_21-cd_l2.pth",
        Path(__file__).parent / "completion" / "configs" / "shapenet34.yaml"
    ),
]

CKPT, CONFIG = CKPT_PATHS[0]

with open(CONFIG, "r") as f:
    config = yaml.safe_load(f)
    model = SnowflakeNet(
        dim_feat=config["model"]["dim_feat"],
        num_pc=config["model"]["num_pc"],
        num_p0=config["model"]["num_p0"],
        radius=config["model"]["radius"],
        bounding=config["model"]["bounding"],
        up_factors=config["model"]["up_factors"],
    )
# state_dict = torch.load(CKPT)['model']
state_dict = {k.removeprefix('module.'): v for k, v in torch.load(CKPT)['model'].items()}
model.load_state_dict(state_dict)

print(model)

## Running inference

import argparse
import open3d as o3d
import numpy as np
import os

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32  # must be float, not double (otherwise weird errors happen)

parser = argparse.ArgumentParser()
parser.add_argument("--input_path", help="Path to input point cloud (.ply) file")
parser.add_argument("--output_path", help="Path to output point cloud (.ply) file")
args = parser.parse_args()

pcd: o3d.geometry.PointCloud = o3d.io.read_point_cloud(args.input_path)
pcd_tensor = torch.from_numpy(np.array(pcd.points)).to(DEVICE).to(DTYPE)
model = model.to(DEVICE)

with torch.no_grad():
    _new_points = model(pcd_tensor.unsqueeze(dim=0))  # expects batch and returns list of (1, N, C) tensors
    extra_pcd_points = torch.cat(_new_points, dim=1).squeeze()  # (sum N_i, C)
completed_pcd = o3d.geometry.PointCloud()
completed_pcd.points = o3d.utility.Vector3dVector(extra_pcd_points.cpu().numpy())

if not os.path.exists(Path(args.output_path).parent):
    os.makedirs(Path(args.output_path).parent, exist_ok=True)
o3d.io.write_point_cloud(args.output_path, completed_pcd, write_ascii=True, print_progress=True)
print(f"Generated point cloud with {len(extra_pcd_points)} points!")
