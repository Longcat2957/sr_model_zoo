import os
import torch
import argparse
from utils.dataset import train_Dataset, valid_Dataset

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=8, help='배치 크기')
parser.add_argument('--root_dir', type=str, default='../data/DF2K_SPLIT', help='데이터셋 루트 디렉토리 경로')
parser.add_argument('--preload', action='store_true', help='데이터셋 미리 로드 여부')
parser.add_argument('--patch', type=int, default=16, help='패치의 갯수')
parser.add_argument('--patch_size', type=int, default=512, help='패치의 크기')
parser.add_argument('--hr_size', type=int, default=256, help='고해상도 이미지 크기')
parser.add_argument('--lr_size', type=int, default=64, help='저해상도 이미지 크기')
parser.add_argument('--epochs', type=int, default=50, help='훈련 에포크 수')
parser.add_argument('--save_freq', type=int, default=10, help='모델 저장 빈도')


if __name__ == '__main__':
    opt = parser.parse_args()
    # 데이터셋 준비
    train_datas_root = os.path.join(opt.root_dir, 'train')
    valid_datas_root = os.path.join(opt.root_dir, 'valid')

    train_dataset = train_Dataset(train_datas_root, opt.preload, opt.patch, opt.patch_size,
                                  opt.hr_size, opt.lr_size)
    valid_dataset = valid_Dataset(valid_datas_root, opt.preload, opt.hr_size, opt.lr_size)

    lr, hr = train_dataset[0]
    print(lr.shape, hr.shape)