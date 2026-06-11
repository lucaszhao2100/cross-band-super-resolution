import argparse
import torch
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import mean_squared_error
from models import ESPCN
from utils import preprocess
import os
from PIL import Image
import PIL.Image as pil_image


def normalize_image(image):
    np_image = np.array(image).astype(np.float32)
    normalized_image = (np_image - np.min(np_image)) / (np.max(np_image) - np.min(np_image))
    return Image.fromarray((normalized_image * 255).astype(np.uint8))


def downsample(image, scale):
    width, height = image.size
    new_width = width // scale
    new_height = height // scale
    return image.resize((new_width, new_height), resample=Image.BICUBIC)


def process_band(weights_file, image_file, scale, save_dir):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = ESPCN(scale_factor=scale).to(device)
    model.load_state_dict(torch.load(weights_file, map_location=device))
    model.eval()

    # 打开图像并转换为浮点数
    lr_full = Image.open(image_file).convert('L')
    lr_full_np = np.array(lr_full).astype(np.float32)
    lr_min, lr_max = np.min(lr_full_np), np.max(lr_full_np)
    lr_full=normalize_image(lr_full_np)
    # Downsample image
    # lr_full = downsample(lr_full, scale)

    # Bicubic upsample to original size, ensuring value range fits the original
    bicubic = lr_full.resize((lr_full.width * scale, lr_full.height * scale),
                                    resample=pil_image.BICUBIC)
    bicubic_np = np.array(bicubic)

    # 恢复双三次插值结果的值域
    # bicubic_rescaled = (bicubic_np - np.min(bicubic_np)) / (np.max(bicubic_np) - np.min(bicubic_np)) * (lr_max - lr_min) + lr_min
    # bicubic_rescaled = bicubic_rescaled.astype(np.uint8)

    # 保存双三次插值的结果
    bicubic_save_image = Image.fromarray(bicubic_np)
    bicubic_save_path = os.path.join(save_dir, os.path.basename(image_file).replace('.tif', f'_bicubic_x{scale}.tif'))
    bicubic_save_image.save(bicubic_save_path)

    # 处理并预测低分辨率图像
    lr_tensor = preprocess(lr_full, device)

    with torch.no_grad():
        preds = model(lr_tensor).clamp(0.0, 1.0)

    preds_np = (preds.cpu().numpy().squeeze() * 255).astype(np.uint8)  # 恢复到原始像素值范围

    # 保存模型预测结果
    pred_save_image = Image.fromarray(preds_np)
    pred_save_path = os.path.join(save_dir, os.path.basename(image_file).replace('.tif', f'_ESPCN_x{scale}.tif'))
    pred_save_image.save(pred_save_path)

    # 计算与原始高分辨率图像的SSIM和PSNR
    ssim_score = 0
    psnr_score = 0

    # 计算双三次插值的PSNR
    psnr_score_bic = 0

    return ssim_score, psnr_score, psnr_score_bic


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights-file',
                        default=r"D:\Projects\new-Duolong\4.3 opt\trial_17\best.pth",
                        type=str)
    parser.add_argument('--scale', type=int, default=2)
    parser.add_argument('--input-dir', default=r"D:\Projects\DuolongRS\Pre_Date_30\new\swir_mosaic_dl_single",
                        type=str)
    parser.add_argument('--save-dir', default=r"D:\Projects\DuolongRS\Pre_Date_30\new\swir_mosaic_dl_single_ESPCN",
                        type=str)
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    for root, _, files in os.walk(args.input_dir):
        for file in files:
            if file.endswith(".tif"):
                image_file = os.path.join(root, file)
                ssim_score, psnr_score, psnr_score_bic = process_band(args.weights_file, image_file, args.scale,
                                                                      args.save_dir)
