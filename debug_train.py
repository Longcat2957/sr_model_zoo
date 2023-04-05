import sys
import os
import torch
from utils.dataset import train_Dataset, valid_Dataset
from utils.colortext import init_colorama, color_print

if __name__ == '__main__':
    train_root = "../data/sr_sample_datas/train"
    valid_root = "../data/sr_sample_datas/valid"
    # print(os.listdir(train_root))

    train_dataset = train_Dataset(train_root, True, 16, 360, 256, 64)
    valid_dataset = valid_Dataset(valid_root, True, 256, 64)

    lr, hr = train_dataset[0]
    # print(lr.shape, hr.shape)

    lr, hr = valid_dataset[0]
    # print(lr.shape, hr.shape)

    # init_colorama()
    sample_string = color_print("Hello world", "red", False)