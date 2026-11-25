"""
HGSD: transferable adversarial example generation.

Crafts adversarial examples on ImageNet by combining a Spatial-Diversity (SID)
gradient ensemble with a curvature-aware (NHTR) loss under an MI-FGSM update.

For each attack iteration we:
  1. draw N spatially transformed copies of the current adversarial input
     (random rescale / pad / flip / localized fusion -- see methods.local_fusion);
  2. for every transformed sample evaluate the NHTR objective
        L_NHTR = delta * L(x) + (1 - delta)/2 * ( L(x + h v) + L(x - h v) ),
     where v is the L1-normalized first-order gradient direction and h = 2*alpha;
  3. weight each sample's gradient by a softmin over its curvature proxy
     (flatter regions -> larger weight), aggregate, and take a signed MI-FGSM
     step projected back into the L-inf epsilon ball.

Defaults below reproduce the main-paper configuration exactly.

Usage:
    python attack.py --model inception_v3 \
        --input_dir ./dataset/images --input_csv ./dataset/images.csv \
        --output_dir ./TestFolder
"""

import os
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models, transforms as T
from PIL import Image
from tqdm import tqdm
import timm

from loader import ImageNet
from Normalize import Normalize
from methods import local_fusion


# Input normalization shared by every model in this study (fixed, not swept).
MEAN = [0.5, 0.5, 0.5]
STD = [0.5, 0.5, 0.5]

PIXEL_MAX = 255.0
_EPS = 1e-8          # numerical stabilizer for gradient normalization
INPUT_SIZE = 224     # all surrogates here expect 224x224 inputs


def parse_args():
    p = argparse.ArgumentParser(
        description="HGSD transferable adversarial attack."
    )
    # Data / IO
    p.add_argument("--input_csv", type=str, default="./dataset/images.csv",
                   help="CSV listing images and labels.")
    p.add_argument("--input_dir", type=str, default="./dataset/images")
    p.add_argument("--output_dir", type=str, default="./TestFolder")
    p.add_argument("--model", type=str, default="inception_v3",
                   help="Surrogate model name (see build_surrogate_model).")

    # Attack budget / optimization
    p.add_argument("--max_epsilon", type=float, default=16.0,
                   help="L-inf budget on the 0-255 pixel scale.")
    p.add_argument("--num_iter", type=int, default=10,
                   help="Number of attack iterations.")
    p.add_argument("--momentum", type=float, default=1.0,
                   help="MI-FGSM momentum decay.")

    # SID / NHTR hyperparameters
    p.add_argument("--N", type=int, default=20,
                   help="Number of spatial transformations per iteration.")
    p.add_argument("--K", type=int, default=2,
                   help="Number of fusion blocks per axis in local_fusion.")
    p.add_argument("--p", type=float, default=0.5,
                   help="Per-block probability parameter in local_fusion.")
    p.add_argument("--omega", type=float, default=0.5,
                   help="Blending strength between original and patch in local_fusion.")
    p.add_argument("--delta", type=float, default=0.3,
                   help="Weight on the main CE loss in the NHTR objective.")

    p.add_argument("--batch_size", type=int, default=10)
    p.add_argument("--seed", type=int, default=None,
                   help="Optional RNG seed for reproducibility (default: unseeded).")
    return p.parse_args()


def get_logits(model, x):
    """Return raw logits, unwrapping torchvision's InceptionOutputs when present."""
    out = model(x)
    return out.logits if hasattr(out, "logits") else out


def safe_grad(loss, wrt, retain_graph=True):
    """Gradient of `loss` w.r.t. `wrt`, or None if the two are not connected."""
    return torch.autograd.grad(
        loss, wrt, retain_graph=retain_graph, allow_unused=True
    )[0]


def HGSD(images, gt, model, images_min, images_max, opt):
    """Run the HGSD attack on one batch and return the adversarial batch."""
    eps = opt.max_epsilon / PIXEL_MAX
    alpha = eps / opt.num_iter
    h = 2 * alpha  # finite-difference step for the curvature probes

    x_adv = images.clone().detach().requires_grad_(True)
    momentum_grad = torch.zeros_like(x_adv)

    for _ in range(opt.num_iter):
        # Spatial Diversity: N stochastic views of the current adversarial input.
        transformations = [
            local_fusion(x_adv, opt.K, opt.p, opt.omega) for _ in range(opt.N)
        ]

        grads = []
        traces = []
        for x_t in transformations:
            x_t.requires_grad_(True)

            # First-order direction v at this transformed sample.
            loss_main = F.cross_entropy(get_logits(model, x_t), gt)
            g_main = safe_grad(loss_main, x_t, retain_graph=True)
            if g_main is None:
                continue
            v = g_main / (torch.norm(g_main, p=1, dim=(1, 2, 3), keepdim=True) + _EPS)

            # NHTR objective: central-difference curvature probe along v.
            loss_plus = F.cross_entropy(get_logits(model, x_t + h * v), gt)
            loss_minus = F.cross_entropy(get_logits(model, x_t - h * v), gt)
            nhtr_loss = (
                opt.delta * loss_main
                + ((1 - opt.delta) / 2) * (loss_plus + loss_minus)
            )

            # Prefer the true chain-rule gradient w.r.t. x_adv; fall back to the
            # transformed input if a transform breaks the gradient path.
            grad = safe_grad(nhtr_loss, x_adv, retain_graph=True)
            if grad is None:
                grad = safe_grad(nhtr_loss, x_t, retain_graph=True)
                if grad is None:
                    continue
            grads.append(grad)

            # Curvature proxy used to weight this view (detached scalar).
            traces.append((loss_plus + loss_minus - 2 * loss_main).detach())

        if not grads:
            return x_adv.detach()

        # Smaller curvature -> larger weight (flatter regions transfer better).
        weights = F.softmin(torch.stack(traces).abs(), dim=0)
        total_grad = torch.zeros_like(x_adv)
        for g, w in zip(grads, weights):
            total_grad += g * w

        # MI-FGSM update.
        total_grad = total_grad / (
            torch.norm(total_grad, p=1, dim=(1, 2, 3), keepdim=True) + _EPS
        )
        momentum_grad = opt.momentum * momentum_grad + total_grad
        x_adv = x_adv.detach() + alpha * momentum_grad.sign()
        x_adv = torch.clamp(x_adv, images_min, images_max).requires_grad_(True)

    return x_adv.detach()


def build_surrogate_model(model_name, device):
    """Load a pretrained classifier and prepend input normalization."""
    name = model_name.lower()
    builders = {
        "inception_v3":       lambda: models.inception_v3(
            weights=models.Inception_V3_Weights.IMAGENET1K_V1, aux_logits=True),
        "inception_v4":       lambda: timm.create_model("inception_v4", pretrained=True),
        "resnet50":           lambda: models.resnet50(
            weights=models.ResNet50_Weights.IMAGENET1K_V2),
        "resnet101":          lambda: models.resnet101(
            weights=models.ResNet101_Weights.IMAGENET1K_V2),
        "resnet152":          lambda: models.resnet152(
            weights=models.ResNet152_Weights.IMAGENET1K_V2),
        "vit_b_16":           lambda: models.vit_b_16(
            weights=models.ViT_B_16_Weights.IMAGENET1K_V1),
        "swin_t":             lambda: models.swin_t(
            weights=models.Swin_T_Weights.IMAGENET1K_V1),
        "convnext_t":         lambda: models.convnext_tiny(
            weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1),
        "vit_base_224_timm":  lambda: timm.create_model("vit_base_patch16_224", pretrained=True),
        "swin_base_224_timm": lambda: timm.create_model("swin_base_patch4_window7_224", pretrained=True),
        "deit_base_224_timm": lambda: timm.create_model("deit_base_patch16_224", pretrained=True),
    }
    if name not in builders:
        raise ValueError(f"Unsupported model: {model_name}")

    backbone = builders[name]().eval().to(device)
    return nn.Sequential(Normalize(MEAN, STD), backbone).eval().to(device)


def main():
    opt = parse_args()
    if opt.seed is not None:
        torch.manual_seed(opt.seed)
        np.random.seed(opt.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_surrogate_model(opt.model, device)
    print(f"Using surrogate model: {opt.model}")

    transform = T.Compose([T.Resize((INPUT_SIZE, INPUT_SIZE)), T.ToTensor()])
    dataset = ImageNet(opt.input_dir, opt.input_csv, transforms=transform)
    data_loader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=False)

    os.makedirs(opt.output_dir, exist_ok=True)
    eps = opt.max_epsilon / PIXEL_MAX

    for images, names, labels in tqdm(data_loader):
        images = images.to(device)
        labels = labels.to(device)

        # L-inf ball around the clean image, clipped to valid pixel range.
        images_min = torch.clamp(images - eps, 0.0, 1.0)
        images_max = torch.clamp(images + eps, 0.0, 1.0)

        adv = HGSD(images, labels, model, images_min, images_max, opt)

        adv_np = (
            adv.clamp(0, 1).cpu().numpy().transpose(0, 2, 3, 1) * PIXEL_MAX
        ).astype(np.uint8)
        for i, name in enumerate(names):
            Image.fromarray(adv_np[i]).save(os.path.join(opt.output_dir, name))


if __name__ == "__main__":
    main()