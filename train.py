import os
import torch
import argparse
from tqdm import tqdm
from torchmetrics.functional import peak_signal_noise_ratio, structural_similarity_index_measure
from torch.utils.data import DataLoader
from model.mobilesr import MOBILESR
from utils.dataset import train_Dataset, valid_Dataset
from utils.metric import AverageMeter


parser = argparse.ArgumentParser()
# 모델학습 공통사항
parser.add_argument('--batch_size', type=int, default=1, help='배치 크기')
parser.add_argument('--root_dir', type=str, default='../data/DF2K_SPLIT', help='데이터셋 루트 디렉토리 경로')
parser.add_argument('--preload', action='store_true', help='데이터셋 미리 로드 여부')
parser.add_argument('--patch', type=int, default=4, help='패치의 갯수')
parser.add_argument('--patch_size', type=int, default=512, help='패치의 크기')
parser.add_argument('--hr_size', type=int, default=256, help='고해상도 이미지 크기')
parser.add_argument('--lr_size', type=int, default=64, help='저해상도 이미지 크기')
parser.add_argument('--epochs', type=int, default=50, help='훈련 에포크 수')
parser.add_argument('--save_freq', type=int, default=10, help='모델 저장 빈도')


if __name__ == '__main__':
    opt = parser.parse_args()
    DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    upscale_ratio = opt.hr_size // opt.lr_size
    print(f"# DEVICE = {DEVICE}, UPSCALE_RATIO = {upscale_ratio}")
    
    # 데이터셋 준비
    train_datas_root = os.path.join(opt.root_dir, 'train')
    valid_datas_root = os.path.join(opt.root_dir, 'valid')

    train_dataset = train_Dataset(train_datas_root, opt.preload, opt.patch, opt.patch_size,
                                  opt.hr_size, opt.lr_size)
    valid_dataset = valid_Dataset(valid_datas_root, opt.preload, opt.hr_size, opt.lr_size)

    actual_batch_size = opt.batch_size*opt.patch
    train_loader = DataLoader(train_dataset, batch_size=actual_batch_size, shuffle=True,
                              collate_fn=train_dataset.collate_fn)
    valid_loader = DataLoader(valid_dataset, actual_batch_size, shuffle=False,)

    # 모델 준비
    net = MOBILESR(upscaling_factor=upscale_ratio)
    net = net.to(DEVICE)

    # 손실 함수
    criterion = torch.nn.L1Loss()

    # 최적화
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

    # 스케줄러
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=10
    )

    # Metric
    train_loss_meter = AverageMeter()
    psnr_meter = AverageMeter()
    ssim_meter = AverageMeter()

    for e in range(1, opt.epochs+1):
        net = net.train()
        train_bar = tqdm(train_loader, ncols=120)
        for lr, hr in train_bar:
            lr, hr = lr.to(DEVICE), hr.to(DEVICE)
            sr = net(lr)

            optimizer.zero_grad()

            loss = criterion(sr, hr)
            if torch.isnan(loss):
                print(f"#[ERROR] loss is NAN")
                continue
            loss.backward()
            optimizer.step()
            train_loss_meter.update(loss, 1)
            train_bar.set_description(
                f"# TRAIN [{e}/{opt.epochs}] loss_avg = {train_loss_meter.avg:.5f}"
            )
        lr_scheduler.step()

        train_loss_meter.reset()
        net = net.eval()
        valid_bar = tqdm(valid_loader, ncols=120)
        for lr, hr in valid_loader:
            lr, hr = lr.to(DEVICE), hr.to(DEVICE)
            with torch.no_grad():
                sr = net(lr)
            
            psnr_score = peak_signal_noise_ratio(sr, hr)
            ssim_score = structural_similarity_index_measure(sr, hr)
            psnr_meter.update(psnr_score)
            ssim_meter.update(ssim_score)

            valid_bar.set_description(
                f"# VALID [{e}/{opt.epochs}] PSNR={psnr_meter.avg:.5f} SSIM={ssim_meter.avg:.5f}"
            )