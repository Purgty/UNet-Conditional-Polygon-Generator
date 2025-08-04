# Ayna UNET Model 🎨

A deep learning model that colorizes grayscale polygon images based on a given color name using a custom-built **Conditional UNet** architecture.

## 🚀 Features

- **Conditional UNet**: Color embedding fused into UNet bottleneck.
- **Mixed Precision Training**: Faster training with `torch.cuda.amp`.
- **Transfer Learning Support**: Resume from pretrained checkpoints.
- **Flexible Inference**: Predict single or batch of images.
- **WandB Integration**: Track experiments, losses & predictions.

## 🧠 Model Overview

- **Encoder**: Downsample grayscale input into deep features.
- **Color Embedding**: Encoded via `nn.Embedding`.
- **Bottleneck**: Color vector concatenated with deep features.
- **Decoder**: Upsamples with skip connections.
- **Output**: RGB image via final convolution.

## 📦 Installation

```bash
git clone https://github.com/YOUR_USERNAME/Ayna-UNET-Model.git
cd Ayna-UNET-Model

conda create -n ayna python=3.10
conda activate ayna

pip install torch torchvision torchaudio wandb matplotlib numpy Pillow
```

## 📁 Dataset Structure

```
dataset/
├── training/
│   ├── inputs/
│   ├── outputs/
│   └── data.json
└── validation/
    ├── inputs/
    ├── outputs/
    └── data.json
```

Each `data.json` contains mappings of:
```json
{ "input_polygon": "img1.png", "colour": "red", "output_image": "img1_colored.png" }
```

## 🏋️ Training

Edit hyperparams in `train.py` and run:

```bash
python train.py
```

## ♻️ Transfer Learning

To fine-tune from a checkpoint, set `pretrained_model_path` in `train.py`.

## 🖼️ Inference

Set `model_path`, `input_image_path`, and `colour` in `inference.py`, then run:

```bash
python inference.py
```

For multiple images, point `input_path` to a folder.

## 🛠 Troubleshooting

- **State Dict Mismatch**: Ensure `embed_dim` and `img_size` match between training and inference configs.
- **CUDA OOM**: Reduce `batch_size` or `img_size`.

## 🙏 Acknowledgements

- **PyTorch** for modeling
- **WandB** for experiment tracking
- **Pillow, NumPy, Matplotlib** for I/O and visualizations
