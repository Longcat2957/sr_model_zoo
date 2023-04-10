import torch
import cv2
import numpy as np
from model.base import load_model
from model.rlfn import RLFN
from model.safmn import SAFMN
from model.mobilesr import MOBILESR
from utils.io import sisr_preprocess, sisr_postprocess, to_tensor

if __name__ == "__main__":
    DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    net = RLFN().to(DEVICE)
    # net = SAFMN(36).to(DEVICE)
    # net = MOBILESR().to(DEVICE)
    try:
        net = load_model(net, "./test_weights/rlfn_final_400.pth")
        # net = load_model(net, "./test_weights/SAFMN_x4_final_300.pth")
        # net = load_model(net, "./test_weights/mobilesr_final_200.pth")
    except:
        print(f"Failed to load model ... ")
    net = net.eval()

    orig_lr_tensor = sisr_preprocess("./test_weights/ms3_01.png").unsqueeze(0).to(DEVICE)
    lr_tensor = orig_lr_tensor.clone()
    for i in range(3):
        with torch.no_grad():
            lr_tensor = net(lr_tensor)
            B, C, H, W = lr_tensor.shape
            nH, nW = H // 4, W // 4
        sr_np_ndarray = sisr_postprocess(lr_tensor)[0]
        sr_np_resized = cv2.resize(sr_np_ndarray, dsize=(nW, nH))
        lr_tensor = to_tensor(sr_np_resized).unsqueeze(0).to(DEVICE)

    cv2.imwrite("if_rep_sr_model.png", sr_np_ndarray[:, :, ::-1])
