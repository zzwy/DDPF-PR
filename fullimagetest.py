#-*- coding:utf-8 -*-
import os
import os.path as osp
import sys
import time
import glob
import logging
import argparse
from torch.utils.data import DataLoader
from skimage.metrics.simple_metrics import peak_signal_noise_ratio
from tqdm import tqdm
from dataset.dataset_sig17 import SIG17_Validation_Dataset
from models.PFShdr import PFShdr
from models.loss import JointReconPerceptualLoss
from utils.utils import *

parser = argparse.ArgumentParser(description="Test Setting")
parser.add_argument("--dataset_dir", type=str, default='None',#dataset
                        help='dataset directory')
parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='disables CUDA training')
parser.add_argument('--test_batch_size', type=int, default=1, metavar='N',
                        help='testing batch size (default: 1)')
parser.add_argument('--num_workers', type=int, default=1, metavar='N',
                        help='number of workers to fetch data (default: 1)')
parser.add_argument('--patch_size', type=int, default=256)
parser.add_argument('--pretrained_model', type=str, default='None')#checkpoint
parser.add_argument('--test_best', action='store_true', default=False)
parser.add_argument('--save_results', action='store_true', default=True)
parser.add_argument('--save_dir', type=str, default="./results/sig")
parser.add_argument('--model_arch', type=int, default=0)

def main():
    # Settings
    args = parser.parse_args()

    # pretrained_model
    print(">>>>>>>>> Start Testing >>>>>>>>>")
    print("Load weights from: ", args.pretrained_model)

    # cuda and devices
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    criterion = JointReconPerceptualLoss().to(device)
    # model architecture
    model_dict = {
        0: PFShdr(in_chans=6,embed_dim=64, depths=[6,6,6], num_heads=[8,8,8], mlp_ratio=2, ), 
    }

    model = model_dict[args.model_arch].to(device)
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(args.pretrained_model,map_location='cpu')['state_dict'])

    model.eval()

    datasets = SIG17_Validation_Dataset(root_dir=args.dataset_dir, is_training=False, crop=False, crop_size=512)
    dataloader = DataLoader(dataset=datasets, batch_size=1, num_workers=1, shuffle=False)
    psnr_l = AverageMeter()
    ssim_l = AverageMeter()
    psnr_mu = AverageMeter()
    ssim_mu = AverageMeter()
    val_loss = AverageMeter()
    for idx, img_dataset in enumerate(dataloader):
        # pred_img, label = test_single_img(model, img_dataset, device)
        with torch.no_grad():
            batch_ldr0, batch_ldr1, batch_ldr2, label = img_dataset['input0'].to(device), \
                                                img_dataset['input1'].to(device), \
                                                img_dataset['input2'].to(device), \
                                                img_dataset['label'].to(device),
            pred_img = model(batch_ldr0, batch_ldr1, batch_ldr2)
            loss = criterion(pred_img, label)
            pred_img = torch.squeeze(pred_img.detach().cpu()).numpy().astype(np.float32)
            label = torch.squeeze(label.detach().cpu()).numpy().astype(np.float32)
        pred_hdr = pred_img.copy()
        pred_hdr = pred_hdr.transpose(1, 2, 0)#[..., ::-1]

        scene_psnr_l = peak_signal_noise_ratio(label, pred_img, data_range=1.0)
        label_mu = range_compressor(label)
        pred_img_mu = range_compressor(pred_img)

        scene_psnr_mu = peak_signal_noise_ratio(label_mu, pred_img_mu, data_range=1.0)
        # ssim-l
        pred_img = np.clip(pred_img * 255.0, 0., 255.).transpose(1, 2, 0)
        label = np.clip(label * 255.0, 0., 255.).transpose(1, 2, 0)
        scene_ssim_l = calculate_ssim(pred_img, label)
        # ssim-\mu
        pred_img_mu = np.clip(pred_img_mu * 255.0, 0., 255.).transpose(1, 2, 0)
        label_mu = np.clip(label_mu * 255.0, 0., 255.).transpose(1, 2, 0)
        scene_ssim_mu = calculate_ssim(pred_img_mu, label_mu)

        psnr_l.update(scene_psnr_l)
        ssim_l.update(scene_ssim_l)
        psnr_mu.update(scene_psnr_mu)
        ssim_mu.update(scene_ssim_mu)
        val_loss.update(loss.item())

        # save results
        if args.save_results:
            if not osp.exists(args.save_dir):
                os.makedirs(args.save_dir)
            save_hdr(os.path.join(args.save_dir, '{}_pred.hdr'.format(idx)), pred_hdr)
    print('Validation set: Average Loss: {:.4f}'.format(val_loss.avg))
    print("Average PSNR_mu: {:.4f}  PSNR_l: {:.4f}".format(psnr_mu.avg, psnr_l.avg))
    print("Average SSIM_mu: {:.4f}  SSIM_l: {:.4f}".format(ssim_mu.avg, ssim_l.avg))
    print(">>>>>>>>> Finish Testing >>>>>>>>>")


if __name__ == '__main__':
    main()




