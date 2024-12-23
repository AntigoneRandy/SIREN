from lib.mdatasets import mDateset
from tqdm import tqdm
from lib.models import HiddenDecoder
from torchvision import transforms
import torch
from torch.utils.data import DataLoader
import random
import argparse
import os

def setup_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        default=None
    )
    parser.add_argument(
        "--decoder_path",
        type=str,
        required=True,
        default=None
    )
    parser.add_argument(
        "--gpu_id",
        type=int,
        help="choose a gpu to use",
        default=0
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=""
    )
    parser.add_argument(
        "--output_filename",
        type=str,
        default="output.txt"
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512
    )
    return parser

def mean_var(table):
    all = sum(table)
    n = len(table)
    mean = all / n
    Sum = 0
    for item in table:
        Sum += (item - mean) ** 2
    var = Sum / n
    return mean, var
parser = setup_parser()
args = parser.parse_args()

os.makedirs(args.output_path, exist_ok=True)
UNNORMALIZE_IMAGENET = transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225], std=[1/0.229, 1/0.224, 1/0.225])
NORMALIZE_IMAGENET = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

decoder = HiddenDecoder(
    num_blocks=8,
    num_bits=48,
    channels=64
)


state_dict = torch.load(args.decoder_path)
decoder.get_center(state_dict['center'])
decoder.load_state_dict(state_dict)


device = f"cuda:{args.gpu_id}"


decoder = decoder.to(device)
decoder.requires_grad_(False)
Sum = 0.
center = decoder.center

train_dataset = mDateset(args.dataset_path, args.resolution)
len_dataset = len(train_dataset)
train_dataset = DataLoader(train_dataset, batch_size=1)


res = []
with torch.no_grad():
    for k, batch in enumerate(tqdm(train_dataset)):
        images = batch["image"].to(device)
        images = NORMALIZE_IMAGENET(images)
        pre = decoder(images)
        var = torch.sqrt(torch.norm(pre - center, p=2, dim=1) ** 2 + 1) - 1
        var = var.item()
        res.append(var)

with open(f"{args.output_path}/{args.output_filename}", "w") as f:
    for i, item in enumerate(res):
        if i == 0:
            f.write(f"{item}")
        else:
            f.write(f",{item}")
print(f"file saved in {args.output_path}/{args.output_filename}")
mean, var = mean_var(res)
print(f"average {mean}")
print(f"variance {var}")

