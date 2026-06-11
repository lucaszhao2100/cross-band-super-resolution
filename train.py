import argparse
import os
import copy

import torch
from torch import nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim

from models import ESPCN
from datasets import TrainDataset, EvalDataset
from utils import AverageMeter, calc_psnr

def write_log(log_dir, epoch, eval_psnr, eval_ssim, train_loss, eval_loss):
    with open(os.path.join(log_dir, 'psnr_log.txt'), 'a') as f:
        f.write(f'Epoch {epoch}: {eval_psnr:.4f}\n')

    with open(os.path.join(log_dir, 'ssim_log.txt'), 'a') as f:
        f.write(f'Epoch {epoch}: {eval_ssim:.4f}\n')

    with open(os.path.join(log_dir, 'train_loss_log.txt'), 'a') as f:
        f.write(f'Epoch {epoch}: {train_loss:.6f}\n')

    with open(os.path.join(log_dir, 'eval_loss_log.txt'), 'a') as f:
        f.write(f'Epoch {epoch}: {eval_loss:.6f}\n')


def calc_ssim(img1, img2):
    img1 = img1.cpu().numpy().squeeze()
    img2 = img2.cpu().numpy().squeeze()
    return ssim(img1, img2, data_range=img2.max() - img2.min())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-dir', type=str,
                        default=r"D:\data\PRISMA_DATA\TRAIN.h5")
    parser.add_argument('--eval-file', type=str,
                        default=r"D:\data\PRISMA_DATA\EVAL.h5")
    parser.add_argument('--outputs-dir', type=str,
                        default=r'D:\data\PRISMA_DATA\NEW_output')
    parser.add_argument('--weights-file', type=str)
    parser.add_argument('--scale', type=int, default=6)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--num-epochs', type=int, default=500)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--seed', type=int, default=123)
    args = parser.parse_args()

    args.outputs_dir = os.path.join(args.outputs_dir, 'x{}'.format(args.scale))

    if not os.path.exists(args.outputs_dir):
        os.makedirs(args.outputs_dir)

    cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    torch.manual_seed(args.seed)

    model = ESPCN(scale_factor=args.scale).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    log_dir = os.path.join(args.outputs_dir, "log")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    print(f"Training with {args.train_dir}")


    train_dataset = TrainDataset(args.train_dir)
    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  num_workers=args.num_workers,
                                  pin_memory=True)

    eval_dataset = EvalDataset(args.eval_file)
    eval_dataloader = DataLoader(dataset=eval_dataset, batch_size=1)

    best_weights = None
    best_epoch = 0
    best_psnr = 0.0
    best_ssim = 0.0

    for epoch in range(args.num_epochs):
        model.train()
        epoch_train_loss = AverageMeter()

        with tqdm(total=(len(train_dataset) - len(train_dataset) % args.batch_size), ncols=80) as t:
            t.set_description(f'epoch: {epoch}/{args.num_epochs - 1}')

            for data in train_dataloader:
                inputs, labels = data

                inputs = inputs.to(device)
                labels = labels.to(device)

                preds = model(inputs)

                loss = criterion(preds, labels)

                epoch_train_loss.update(loss.item(), len(inputs))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                t.set_postfix(loss='{:.6f}'.format(epoch_train_loss.avg))
                t.update(len(inputs))

        torch.save(model.state_dict(), os.path.join(log_dir, f'epoch_{epoch}.pth'))

        model.eval()
        epoch_psnr = AverageMeter()
        epoch_ssim = AverageMeter()
        epoch_eval_loss = AverageMeter()

        for data in eval_dataloader:
            inputs, labels = data

            inputs = inputs.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                preds = model(inputs).clamp(0.0, 1.0)

            eval_loss = criterion(preds, labels)
            epoch_eval_loss.update(eval_loss.item(), len(inputs))

            epoch_psnr.update(calc_psnr(preds, labels), len(inputs))
            epoch_ssim.update(calc_ssim(preds, labels), len(inputs))


        write_log(log_dir, epoch, epoch_psnr.avg, epoch_ssim.avg, epoch_train_loss.avg, epoch_eval_loss.avg)

        if epoch_psnr.avg > best_psnr:
            best_epoch = epoch
            best_psnr = epoch_psnr.avg
            best_ssim = epoch_ssim.avg
            best_weights = copy.deepcopy(model.state_dict())

    print(f'Best epoch: {best_epoch}, PSNR: {best_psnr:.4f}, SSIM: {best_ssim:.4f}')
    torch.save(best_weights, os.path.join(log_dir, 'best.pth'))
