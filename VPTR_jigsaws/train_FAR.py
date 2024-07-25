import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

from pathlib import Path
from datetime import datetime

from model import VPTREnc, VPTRDec, VPTRDisc, init_weights, VPTRFormerFAR
from model import GDL, MSELoss, GANLoss
from utils import get_dataloader, save_ckpt, load_ckpt, set_seed, AverageMeters, init_loss_dict, write_summary, resume_training
from utils import visualize_batch_clips

import logging

set_seed(2021)

def resize_tensor(tensor, size):
    N, T, C, H, W = tensor.shape
    tensor = tensor.view(-1, C, H, W)
    tensor_resized = F.interpolate(tensor, size=size, mode='bilinear', align_corners=False)
    tensor_resized = tensor_resized.view(N, T, C, size[0], size[1])
    return tensor_resized

def cal_lossD(VPTR_Disc, fake_imgs, real_imgs, lam_gan):
    pred_fake = VPTR_Disc(fake_imgs.detach().flatten(0, 1))
    loss_D_fake = gan_loss(pred_fake, False)
    pred_real = VPTR_Disc(real_imgs.flatten(0,1))
    loss_D_real = gan_loss(pred_real, True)
    loss_D = (loss_D_fake + loss_D_real) * 0.5 * lam_gan
    return loss_D, loss_D_fake, loss_D_real

def cal_lossT(fake_imgs, real_imgs, VPTR_Disc, lam_gan):
    #ensure tensors have the same spatial dimensions
    if fake_imgs.shape[-2:] != real_imgs.shape[-2:]:
        N, T, C, H, W = fake_imgs.shape
        #Flatten batch and time dimensions before resizing
        fake_imgs = fake_imgs.flatten(0,1)
        fake_imgs = F.interpolate(fake_imgs, size=(real_imgs.shape[-2], real_imgs.shape[-1]), mode='bilinear', align_corners=False)
        #Reshape back to the flattened shape (N*T, C, H, W) before permuting back to (N, T, C, H, W)
        fake_imgs = fake_imgs.reshape(N * T, C, real_imgs.shape[-2], real_imgs.shape[-1]) #correct reshape
        #premute back to (N, T, C, H, W)
        fake_imgs = fake_imgs.view(N, T, C, real_imgs.shape[-2], real_imgs.shape[-1])
        
    T_MSE_loss = mse_loss(fake_imgs, real_imgs)
    T_GDL_loss = gdl_loss(real_imgs, fake_imgs)
    if VPTR_Disc is not None:
        assert lam_gan is not None, "Please input lam_gan"
        pred_fake = VPTR_Disc(fake_imgs.flatten(0, 1))
        loss_T_gan = gan_loss(pred_fake, True)
        loss_T = T_GDL_loss + T_MSE_loss + lam_gan * loss_T_gan
    else:
        loss_T_gan = torch.zeros(1)
        loss_T = T_GDL_loss + T_MSE_loss
    return loss_T, T_GDL_loss, T_MSE_loss, loss_T_gan

def single_iter(VPTR_Enc, VPTR_Dec, VPTR_Disc, VPTR_Transformer, optimizer_T, optimizer_D, sample, device, lam_gan, train_flag=True):
    past_frames, future_frames = sample
    past_frames = past_frames.to(device)
    future_frames = future_frames.to(device)
    
    with torch.no_grad():
        x = torch.cat([past_frames, future_frames[:, :-1, ...]], dim=1) #concatenate past frames and future frames except the last frame
        gt_feats = VPTR_Enc(x) #encode the concatenated frames
        
    if train_flag:
        VPTR_Transformer=VPTR_Transformer.train()
        VPTR_Transformer.zero_grad(set_to_none=True)
        VPTR_Dec.zero_grad(set_to_none=True)
        
        pred_future_feats = VPTR_Transformer(gt_feats) #predict the future features
        pred_frames = VPTR_Dec(pred_future_feats) #decode the predicted future features
        
        if optimizer_D is not None:
            assert lam_gan is not None, "Please input lam_gan"
            #update discriminator
            VPTR_Disc=VPTR_Disc.train()
            for p in VPTR_Disc.parameters():
                p.requires_grad_(True)
            VPTR_Disc.zero_grad(set_to_none=True)
            loss_D, loss_D_fake, loss_D_real = cal_lossD(VPTR_Disc, pred_frames, future_frames, lam_gan)
            loss_D.backward()
            optimizer_D.step()
        
            for p in VPTR_Disc.parameters():
                p.requires_grad_(False)


        pred_frames_resized = resize_tensor(pred_frames, (future_frames.shape[-2], future_frames.shape[-1]))
        
        ##update transformer(generator)
        loss_T, T_GDL_loss, T_MSE_loss, loss_T_gan = cal_lossT(pred_frames_resized, torch.cat([past_frames[:, 1:, ...], future_frames], dim=1), VPTR_Disc, lam_gan)
        loss_T.backward()
        nn.utils.clip_grad_norm_(VPTR_Transformer.parameters(), max_grad_norm, norm_type=2)
        optimizer_T.step()

    else:
        if optimizer_D is not None:
            VPTR_Disc=VPTR_Disc.eval()
        VPTR_Transformer= VPTR_Transformer.eval()
        with torch.no_grad():
            pred_future_feats = VPTR_Transformer(gt_feats)
            pred_frames = VPTR_Dec(pred_future_feats)
            if optimizer_D is not None:
                loss_D, loss_D_fake, loss_D_real = cal_lossD(VPTR_Disc, pred_frames, future_frames, lam_gan)
            else:
                loss_D, loss_D_fake, loss_D_real = torch.zeros(1), torch.zeros(1), torch.zeros(1)
                
                
            pred_frames_resized = resize_tensor(pred_frames, (future_frames.shape[-2], future_frames.shape[-1]))
            loss_T, T_GDL_loss, T_MSE_loss, loss_T_gan = cal_lossT(pred_frames, torch.cat([past_frames[:, 1:, ...], future_frames], dim=1), VPTR_Disc, lam_gan)
    
    if optimizer_D is None:        
        loss_D, loss_D_fake, loss_D_real = torch.zeros(1), torch.zeros(1), torch.zeros(1)

    iter_loss_dict = {'T_total': loss_T.item(), 'T_MSE': T_MSE_loss.item(), 'T_GDL': T_GDL_loss.item(), 'T_gan': loss_T_gan.item(), 'Dtotal': loss_D.item(), 'Dfake': loss_D_fake.item(), 'Dreal': loss_D_real.item()}
    
    return iter_loss_dict

if __name__ == '__main__':
    set_seed(2021)
    ckpt_save_dir = Path('C:\\VPTR_jigsaws\\jigsaws_suturing\\VPTR_ckpts\\JIGSAWS_FAR_ckpt')
    tensorboard_save_dir = Path('C:\\VPTR_jigsaws\\jigsaws_suturing\\VPTR_ckpts\\JIGSAWS_FAR_tensorboard')
    resume_AE_ckpt = Path('C:\\VPTR_jigsaws\\jigsaws_suturing\\VPTR_ckpts\\JIGSAWS_ResNetAE_MSEGDLgan_ckpt').joinpath('epoch_100.tar')
    #resume_ckpt = ckpt_save_dir.joinpath('epoch_100.tar')
    resume_ckpt = None
    
    ####Set the logger###

    if not Path(ckpt_save_dir).exists():
        Path(ckpt_save_dir).mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO,
                        datefmt='%a, %d %b %Y %H:%M:%S',
                        format='%(asctime)s - %(message)s',
                        filename=ckpt_save_dir.joinpath('train_log.log').absolute().as_posix(),
                        filemode='a')

    start_epoch = 0
    summary_writer = SummaryWriter(tensorboard_save_dir.absolute().as_posix())
    num_past_frames = 10
    num_future_frames = 20
    encH, encW, encC = 8, 8, 528
    img_channels = 3
    epochs = 100
    N = 4
    #AE_lr = 2e-4
    Transformer_lr = 1e-4
    max_grad_norm = 1.0
    rpe = False
    lam_gan = 0.001
    dropout = 0.1
    device = torch.device('cuda:0')

    ##################### Init Dataset ###########################
    data_set_name = 'Suturing'
    dataset_dir = 'C:\\VPTR_jigsaws\\jigsaws_suturing\\frames_split\\'
    train_loader, test_loader, renorm_transform = get_dataloader(data_set_name, N, dataset_dir, num_past_frames=num_past_frames, num_future_frames=num_future_frames)

    ##################### Init Models and Optimizer ###########################
    VPTR_Enc = VPTREnc(img_channels, feat_dim=encC, n_downsampling=3).to(device)
    VPTR_Dec = VPTRDec(img_channels, feat_dim=encC, n_downsampling=3, out_layer='Tanh').to(device)
    VPTR_Enc = VPTR_Enc.eval()
    VPTR_Dec = VPTR_Dec.eval()
    VPTR_Disc = None
    #VPTR_Disc = VPTRDisc(img_channels, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d).to(device)
    #VPTR_Disc = VPTR_Disc.eval()
    #init_weights(VPTR_Disc)
    init_weights(VPTR_Enc)
    init_weights(VPTR_Dec)
    
    VPTR_Transformer = VPTRFormerFAR(num_past_frames=num_past_frames, num_future_frames=num_future_frames, encH=encH, encW=encW, d_model=encC, 
                                     nhead=8, num_encoder_layers=12, dropout=dropout, window_size=4, Spatial_FFN_hidden_ratio=4).to(device)
    
    optimizer_D = None
    #optimizer_D = torch.optim.Adam(params=VPTR_Disc.parameters(), lr=Transformer_lr, betas=(0.5, 0.999))
    optimizer_T = torch.optim.AdamW(params=VPTR_Transformer.parameters(), lr=Transformer_lr)


    Transformer_parameters = sum(p.numel() for p in VPTR_Transformer.parameters() if p.requires_grad)

    print(f"Transformer num_parameters: {Transformer_parameters}")


    ##################### Init Criterion ###########################
    loss_name_list = ['T_MSE', 'T_GDL', 'T_gan', 'T_total', 'Dtotal', 'Dfake', 'Dreal']
    loss_dict = init_loss_dict(loss_name_list)
    mse_loss = MSELoss()
    gdl_loss = GDL(alpha=1)
    #gan_loss = GANLoss('vanilla', target_real_label=1.0, target_fake_label=0.0).to(device)

    # Load the trained autoencoder
    loss_dict, start_epoch = resume_training({'VPTR_Enc': VPTR_Enc, 'VPTR_Dec': VPTR_Dec}, {}, resume_AE_ckpt, loss_name_list)

    if resume_ckpt is not None:
        loss_dict, start_epoch = resume_training({'VPTR_Transformer': VPTR_Transformer}, {'optimizer_T': optimizer_T}, resume_ckpt, loss_name_list)

    ##################### Training loop ###########################
    for epoch in range(start_epoch+1, start_epoch + epochs+1):
        epoch_st = datetime.now()
        
        #train
        EpochAveMeter = AverageMeters(loss_name_list)
        for idx, sample in enumerate(train_loader, 0):
            iter_loss_dict = single_iter(VPTR_Enc, VPTR_Dec, VPTR_Disc, VPTR_Transformer, optimizer_T, optimizer_D, sample, device, lam_gan, train_flag=True)
            EpochAveMeter.iter_update(iter_loss_dict)
            
        loss_dict = EpochAveMeter.epoch_update(loss_dict, epoch, train_flag=True)
        write_summary(summary_writer, loss_dict, train_flag=True)

        EpochAveMeter = AverageMeters(loss_name_list)
        for idx, sample in enumerate(test_loader, 0):
            iter_loss_dict = single_iter(VPTR_Enc, VPTR_Dec, VPTR_Disc, VPTR_Transformer, optimizer_T, optimizer_D, sample, device, lam_gan, train_flag=False)
            EpochAveMeter.iter_update(iter_loss_dict)
        loss_dict = EpochAveMeter.epoch_update(loss_dict, epoch, train_flag=False)
        write_summary(summary_writer, loss_dict, train_flag=False)

        save_ckpt({'VPTR_Enc': VPTR_Enc, 'VPTR_Dec': VPTR_Dec, 'VPTR_Transformer': VPTR_Transformer}, {'optimizer_T': optimizer_T}, epoch, loss_dict, ckpt_save_dir)

        epoch_time = datetime.now() - epoch_st
        print(f'epoch {epoch}', EpochAveMeter.meters['T_total'])
        print(f"Estimated remaining training time: {epoch_time.total_seconds() / 3600. * (start_epoch + epochs - epoch)} Hours")
