# infer_single_binary.py
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import warnings
warnings.filterwarnings("ignore")

import Config as config
from nets.FAA import FAA


def ensure_dir(p):
    if not os.path.exists(p):
        os.makedirs(p)


def load_checkpoint_flex(model, ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt.get("state_dict", ckpt)

    from collections import OrderedDict
    fixed = OrderedDict()
    for k, v in state.items():
        fixed[k[7:]] = v if k.startswith("module.") else v
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError:
        model.load_state_dict(fixed, strict=False)
    return model


def to_tensor_from_cv(img, n_channels, img_size):
    """将OpenCV图像转为(1,C,H,W)张量, 值域[0,1]"""
    H, W = img_size, img_size
    if n_channels == 1:
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        img = img.astype(np.float32) / 255.0
        img = img[None, :, :]          # (1,H,W)
    else:
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))  # (C,H,W)
    img = np.expand_dims(img, 0)            # (1,C,H,W)
    return torch.from_numpy(img)


def predict_single_binary(img_path, weights="nudt.pth.tar",
                          out_dir="./single_pred", thresh=0.5,
                          use_sigmoid=False):
    """
    只做前景二值预测并保存PNG
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 构建模型（示例以FAA为例）
    assert config.model_name == "FAA", "此脚本示范FAA模型"
    model = FAA(n_channels=config.n_channels, n_classes=config.n_labels)

    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model = model.to(device).eval()

    # 加载权重
    model = load_checkpoint_flex(model, weights, device)
    print("Model loaded!")

    # 读取与预处理
    assert os.path.exists(img_path), f"Image not found: {img_path}"
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    tensor = to_tensor_from_cv(img, config.n_channels, config.img_size).to(device)

    # 前向 + 阈值化
    with torch.no_grad():
        out = model(tensor)                      # 期望 (1,1,H,W)
        if use_sigmoid:
            out = torch.sigmoid(out)
        pred = (out > thresh).float().cpu().numpy()[0, 0]  # (H,W) 0/1

    # 保存
    ensure_dir(out_dir)
    base = os.path.splitext(os.path.basename(img_path))[0]
    save_path = os.path.join(out_dir, f"{base}_pred.png")
    cv2.imwrite(save_path, (pred * 255).astype(np.uint8))
    print(f"Saved: {save_path}")
    return save_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Binary foreground prediction (single image)")
    parser.add_argument("--img", default="000133.png", type=str, help="Path to image")
    parser.add_argument("--weights", default="nudt.pth.tar", type=str, help="Checkpoint path")
    parser.add_argument("--outdir", default="./single_pred", type=str, help="Output directory")
    parser.add_argument("--thresh", default=0.5, type=float, help="Binarization threshold")
    parser.add_argument("--use_sigmoid", action="store_true",
                        help="Apply sigmoid before threshold (若模型输出logits则打开)")
    args = parser.parse_args()

    predict_single_binary(
        img_path=args.img,
        weights=args.weights,
        out_dir=args.outdir,
        thresh=args.thresh,
        use_sigmoid=args.use_sigmoid
    )
