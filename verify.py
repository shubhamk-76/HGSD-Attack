"""
verify.py — evaluate transferability of adversarial images across target models.

For each target model listed in TARGET_MODELS, loads the adversarial images
from --adv_dir (paired with their labels in --input_csv via the ImageNet
loader) and reports both the Attack Success Rate (fraction of
misclassifications) and the standard top-1 accuracy. A final row aggregates
the average across all targets.

Usage:
    python verify.py --adv_dir ./TestFolder \
        --input_csv ./dataset/images.csv \
        --batch_size 10
"""

import os
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models, transforms as T
import timm

from Normalize import Normalize
from loader import ImageNet


# ImageNet normalization shared by every model evaluated here.
MEAN = [0.5, 0.5, 0.5]
STD = [0.5, 0.5, 0.5]

# Target models reported in the main paper.
TARGET_MODELS = [
    "inception_v3",
    "vgg16",
    "mobilenet_v2",
    "resnet50",
    "vit_b_16",
    "swin_base",
]

# Pretrained-classifier factories.
PT_CLASSIFIERS = {
    "inception_v3": lambda: models.inception_v3(
        weights=models.Inception_V3_Weights.IMAGENET1K_V1, aux_logits=True),
    "vgg16":        lambda: models.vgg16(
        weights=models.VGG16_Weights.IMAGENET1K_V1),
    "mobilenet_v2": lambda: models.mobilenet_v2(
        weights=models.MobileNet_V2_Weights.IMAGENET1K_V1),
    "resnet50":     lambda: models.resnet50(
        weights=models.ResNet50_Weights.IMAGENET1K_V2),
    "vit_b_16":     lambda: models.vit_b_16(
        weights=models.ViT_B_16_Weights.IMAGENET1K_V1),
    "swin_base":    lambda: timm.create_model(
        "swin_base_patch4_window7_224", pretrained=True),
}

# Native input resolution per model.
MODEL_INPUT_SIZE = {
    "inception_v3": 299,
    "vgg16":        224,
    "mobilenet_v2": 224,
    "resnet50":     224,
    "vit_b_16":     224,
    "swin_base":    224,
}


def parse_args():
    p = argparse.ArgumentParser(
        description="Verify transferability of adversarial images on multiple target models."
    )
    p.add_argument("--adv_dir", type=str, required=True,
                   help="Directory containing adversarial images.")
    p.add_argument("--input_csv", type=str, default="./dataset/images.csv",
                   help="CSV listing images and labels.")
    p.add_argument("--batch_size", type=int, default=10)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--gpu", type=str, default="0",
                   help="GPU id to expose via CUDA_VISIBLE_DEVICES.")
    return p.parse_args()


def get_logits(model, x):
    """Return raw logits, unwrapping torchvision's InceptionOutputs when present."""
    out = model(x)
    return out.logits if hasattr(out, "logits") else out


def build_model(model_name, device):
    """Load a pretrained classifier and prepend input normalization."""
    if model_name not in PT_CLASSIFIERS:
        raise ValueError(f"Unknown model name: {model_name}")
    backbone = PT_CLASSIFIERS[model_name]().eval().to(device)
    return nn.Sequential(Normalize(MEAN, STD), backbone).eval().to(device)


def evaluate(model_name, adv_dir, input_csv, batch_size, num_workers, device):
    """Return (asr, acc) for one target model over the adversarial images."""
    model = build_model(model_name, device)
    size = MODEL_INPUT_SIZE.get(model_name, 224)

    dataset = ImageNet(
        adv_dir,
        input_csv,
        T.Compose([T.Resize((size, size)), T.ToTensor()]),
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    correct = 0
    total = 0
    with torch.no_grad():
        for images, _, gt in loader:
            images = images.to(device, non_blocking=True)
            gt = gt.to(device, non_blocking=True)
            preds = get_logits(model, images).argmax(1)
            correct += (preds == gt).sum().item()
            total += gt.numel()

    acc = correct / total
    asr = 1.0 - acc
    return asr, acc


def main():
    opt = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Adversarial directory : {opt.adv_dir}")
    print(f"Input CSV             : {opt.input_csv}")
    print(f"Batch size            : {opt.batch_size}")
    print(f"Device                : {device}")
    print("=" * 50)
    print(f"{'Model':<15s} | {'ASR':>8s} | {'ACC':>8s}")
    print("-" * 50)

    results = []
    for name in TARGET_MODELS:
        asr, acc = evaluate(
            model_name=name,
            adv_dir=opt.adv_dir,
            input_csv=opt.input_csv,
            batch_size=opt.batch_size,
            num_workers=opt.num_workers,
            device=device,
        )
        results.append((name, asr, acc))
        print(f"{name:<15s} | {asr:>7.2%} | {acc:>7.2%}")

    avg_asr = sum(asr for _, asr, _ in results) / len(results)
    avg_acc = sum(acc for _, _, acc in results) / len(results)
    print("-" * 50)
    print(f"{'Average':<15s} | {avg_asr:>7.2%} | {avg_acc:>7.2%}")
    print("=" * 50)


if __name__ == "__main__":
    main()