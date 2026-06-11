import argparse
import torch
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
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
    """使用双三次插值对图像进行下采样"""
    width, height = image.size
    new_width = width // scale
    new_height = height // scale
    return image.resize((new_width, new_height), resample=Image.BICUBIC)


def process_band(weights_file, image_file, scale, save_dir):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = ESPCN(scale_factor=scale).to(device)
    model.load_state_dict(torch.load(weights_file, map_location=device))
    model.eval()

    # 打开原始图像
    lr_full = Image.open(image_file).convert('L')  # 转换为灰度图像
    lr_full_np = np.array(lr_full).astype(np.float32)
    lr_full=normalize_image(lr_full_np)
    # 对图像进行下采样
    lr_downsampled = downsample(lr_full, scale)

    # 对下采样后的图像进行双三次插值
    bicubic = lr_downsampled.resize((lr_downsampled.width * scale, lr_downsampled.height * scale),
                                    resample=pil_image.BICUBIC)
    bicubic_np = np.array(bicubic) # 保持原始像素值范围

    # 保存经过双三次插值的图像
    bicubic_save_image = Image.fromarray(bicubic_np)
    bicubic_save_path = os.path.join(save_dir, os.path.basename(image_file).replace('.tif', f'_bicubic_x{scale}.tif'))
    bicubic_save_image.save(bicubic_save_path)

    # 将下采样后的图像预处理为模型输入的格式
    lr_tensor = preprocess(lr_downsampled, device)

    # 使用模型进行超分辨率预测
    with torch.no_grad():
        preds = model(lr_tensor).clamp(0.0, 1.0)  # 保持输出范围在[0, 1]之间

    preds_np = (preds.cpu().numpy().squeeze() * 255).astype(np.uint8)  # 恢复到原始像素值范围

    # 保存模型预测结果
    pred_save_image = Image.fromarray(preds_np)
    pred_save_path = os.path.join(save_dir, os.path.basename(image_file).replace('.tif', f'_ESPCN_x{scale}.tif'))
    pred_save_image.save(pred_save_path)

    # 计算SSIM和PSNR（不进行归一化）
    ssim_score = ssim(preds_np, np.array(lr_full), data_range=255)  # 使用255作为数据范围
    psnr_score = psnr(preds_np, np.array(lr_full), data_range=255)
    psnr_score_bic = psnr(bicubic_np, np.array(lr_full), data_range=255)

    return ssim_score, psnr_score, psnr_score_bic


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights-file', default=r"D:\Projects\new-Duolong\4.3 opt\trial_17\best.pth", type=str)
    parser.add_argument('--scale', type=int, default=2)
    parser.add_argument('--input-dir', default=r"D:\Projects\DuolongRS\Pre_Date_30\new\vnir_mosaic_dl_single", type=str)
    parser.add_argument('--save-dir',
                        default=r"D:\Projects\DuolongRS\Pre_Date_30\new\vnir_mosaic_dl_single_new", type=str)
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    ssim_sum = 0.0
    psnr_sum = 0.0
    psnr_bic_sum = 0.0
    total_images = 0

    # 遍历输入目录中的所有图像
    for root, _, files in os.walk(args.input_dir):
        for file in files:
            if file.endswith(".tif"):
                image_file = os.path.join(root, file)
                ssim_score, psnr_score, psnr_score_bic = process_band(args.weights_file, image_file, args.scale,
                                                                      args.save_dir)

                ssim_sum += ssim_score
                psnr_sum += psnr_score
                psnr_bic_sum += psnr_score_bic
                total_images += 1

    # 计算平均SSIM和PSNR
    if total_images > 0:
        ssim_avg = ssim_sum / total_images
        psnr_avg = psnr_sum / total_images
        psnr_bic_sum_avg = psnr_bic_sum / total_images

        print(f'Average SSIM: {ssim_avg:.4f}')
        print(f'Average PSNR: {psnr_avg:.4f}')
        print(f'Average PSNR_BIC: {psnr_bic_sum_avg:.4f}')
    else:
        print('No .tif files found in the specified directory.')
