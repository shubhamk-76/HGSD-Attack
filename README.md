# HGSD-Attack

Hessian-Guided Spatial Diversity: Boosting Adversarial Transferability via Weighted Curvature Suppression

## Setup

### Models
Most surrogate and target models are loaded automatically from torchvision /
timm pretrained weights on first use, so no manual download is required.

For additional baselines (e.g. adversarially-trained defenses), download the
PyTorch-converted checkpoints from
[tf_to_pytorch_model](https://github.com/ylhz/tf_to_pytorch_model) and place
them under `./models/`.

### Dataset
We use the standard 1000-image ImageNet-compatible subset. Follow the
instructions in [SSA](https://github.com/yuyang-long/SSA) to obtain the data,
then place it under `./dataset/`:
```
./dataset/
├── images.csv
└── images/
    ├── 0.png
    ├── 1.png
    └── ...
```

## To Generate adversarial examples run the command below

`attack.py` crafts adversarial images using a chosen surrogate and writes them
to `--output_dir`.

```bash
python attack.py \
    --model inception_v3 \
    --input_dir ./dataset/images \
    --input_csv ./dataset/images.csv \
    --output_dir ./TestFolder
```


## To evaluate the attack run the command below
`verify.py` evaluates the attack success rate (ASR) and clean-label accuracy
(ACC) of a folder of adversarial images on all target models reported in the
paper.

```bash
python verify.py \
    --adv_dir ./TestFolder \
    --input_csv ./dataset/images.csv \
    --batch_size 10
```


## Reproducibility
All experiments use seed 42 by default. Results may vary by ±2-3% 
across different GPU architectures due to hardware-level numerical 
differences (cuDNN, tensor cores). Paper results were obtained on 
NVIDIA L40 and our CUDA Version is 13.0.