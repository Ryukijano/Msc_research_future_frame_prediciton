from PIL import Image
from pathlib import Path
import torch
import torchvision.transforms as transforms
from pathlib import Path
import shutil
from collections import OrderedDict
import torchvision.utils as vutils

import os
def resume_training(module_dict, optimizer_dict, resume_ckpt, loss_name_list=None, map_location=None):
    modules_state_dict, optimizers_state_dict, start_epoch, history_loss_dict, _ = load_ckpt(resume_ckpt, map_location)
    for k, m in module_dict.items():
        state_dict = modules_state_dict[k]
        new_state_dict = OrderedDict()
        for sk, sv in state_dict.items():
            if sk.startswith('.model'):
                nk = 'encoder' + sk  # Add 'encoder' prefix
            else:
                nk = sk
            new_state_dict[nk] = sv
        try:
            m.load_state_dict(new_state_dict, strict=False)
        except RuntimeError as e:
            print(f"Error loading state dict for {k}: {e}")
    
    # Rest of the function remains the same
            for sk, sv in state_dict.items():
                nk = sk[7:] # remove `module.`
                new_state_dict[nk] = sv
            m.load_state_dict(new_state_dict)

    if map_location is None:
        loss_dict = init_loss_dict(loss_name_list, history_loss_dict)
        return loss_dict, start_epoch
    else:
        return start_epoch, history_loss_dict


class AverageMeters(object):
    def __init__(self, loss_name_list):
        self.loss_name_list = loss_name_list
        self.meters = {}
        for name in loss_name_list:
            self.meters[name] = BatchAverageMeter(name, ':.10e')
    
    def iter_update(self, iter_loss_dict):
        for k, v in iter_loss_dict.items():
            self.meters[k].update(v)
    
    def epoch_update(self, loss_dict, epoch, train_flag = True):
        if train_flag:
            for k, v in loss_dict.items():
                try:
                    v.train.append(self.meters[k].avg)
                except AttributeError:
                    pass
                except KeyError:
                    v.train.append(0)
        else:
            for k, v in loss_dict.items():
                try:
                    v.val.append(self.meters[k].avg)
                except AttributeError:
                    pass
                except KeyError:
                    v.val.append(0)
        loss_dict['epochs'] = epoch

        return loss_dict

def gather_AverageMeters(aveMeter_list):
    """
    average the avg value from different rank
    Args:
        aveMeter_list: list of AverageMeters objects
    """
    AM0 = aveMeter_list[0]
    name_list = AM0.loss_name_list

    return_AM = AverageMeters(name_list)
    for name in name_list:
        avg_val = 0
        for am in aveMeter_list:
            rank_avg = am.meters[name].avg
            avg_val += rank_avg
        avg_val = avg_val/len(aveMeter_list)
        return_AM.meters[name].avg = avg_val
    
    return return_AM


class Loss_tuple(object):
    def __init__(self):
        self.train = []
        self.val = []

def init_loss_dict(loss_name_list, history_loss_dict = None):
    loss_dict = {}
    for name in loss_name_list:
        loss_dict[name] = Loss_tuple()
    loss_dict['epochs'] = 0

    if history_loss_dict is not None:
        for k, v in history_loss_dict.items():
            loss_dict[k] = v

        for k, v in loss_dict.items():
            if k not in history_loss_dict:
                lt = Loss_tuple()
                lt.train = [0] * history_loss_dict['epochs']
                lt.val = [0] * history_loss_dict['epochs']
                loss_dict[k] = lt

    return loss_dict

def write_summary(summary_writer, in_loss_dict, train_flag = True):
    loss_dict = in_loss_dict.copy()
    del loss_dict['epochs']
    if train_flag:
        for k, v in loss_dict.items():
            for i in range(len(v.train)):
                summary_writer.add_scalars(k, {'train': v.train[i]}, i+1)
    else:
        for k, v in loss_dict.items():
            for i in range(len(v.val)):
                summary_writer.add_scalars(k, {'val': v.val[i]}, i+1)

def save_ckpt(Modules_dict, Optimizers_dict, epoch, loss_dict, save_dir):
    #Save checkpoints every epoch
    if not Path(save_dir).exists():
        Path(save_dir).mkdir(parents=True, exist_ok=True) 
    ckpt_file = Path(save_dir).joinpath(f"epoch_{epoch}.tar")
    ckpt_codes = read_code_files()

    module_state_dict = {}
    for k, m in Modules_dict.items():
        module_state_dict[k] = m.state_dict()
    optim_state_dict = {}
    for k, m in Optimizers_dict.items():
        optim_state_dict[k] = m.state_dict()
    torch.save({
        'epoch': epoch,
        'loss_dict': loss_dict, #{loss_name: [train_loss_list, val_loss_list]}
        'Module_state_dict': module_state_dict,
        'optimizer_state_dict': optim_state_dict,
        'code': ckpt_codes
    }, ckpt_file.absolute().as_posix())

def load_ckpt(ckpt_file, map_location = None):
    ckpt = torch.load(ckpt_file, map_location = map_location)

    epoch = ckpt["epoch"]
    loss_dict = ckpt["loss_dict"]
    Modules_state_dict = ckpt['Module_state_dict']
    Optimizers_state_dict = ckpt['optimizer_state_dict']
    code = ckpt['code']

    return Modules_state_dict, Optimizers_state_dict, epoch, loss_dict, code

def visualize_batch_clips(gt_past_frames_batch, gt_future_frames_batch, pred_frames_batch, file_dir, renorm_transform=None, desc=None):
    """
    Visualizes batches of video clips, handling different past/future frame lengths.
    """
    if not Path(file_dir).exists():
        Path(file_dir).mkdir(parents=True, exist_ok=True)

    def save_clip(clip, file_name):
        imgs = []
        if renorm_transform is not None:
            clip = renorm_transform(clip)
            clip = torch.clamp(clip, min=0., max=1.0)
        for i in range(clip.shape[0]):
            img = transforms.ToPILImage()(clip[i, ...])
            imgs.append(img)

        imgs[0].save(str(Path(file_name).absolute()), save_all=True, append_images=imgs[1:])

    # Get the number of past and future frames from the input tensors
    past_frames = gt_past_frames_batch.shape[1]
    future_frames = gt_future_frames_batch.shape[1]
    pred_future_frames = pred_frames_batch.shape[1]

    # Create a tensor to hold all frames side by side
    total_frames = past_frames + future_frames
    N, _, C, H, W = gt_past_frames_batch.shape
    batch = torch.zeros(N, total_frames, C, H, W * 3)

    # Fill in the frames
    batch[:, :past_frames, :, :, :W] = gt_past_frames_batch
    batch[:, past_frames:past_frames + future_frames, :, :, W:2*W] = gt_future_frames_batch[:, :future_frames, :, :, :]
    batch[:, past_frames:past_frames + pred_future_frames, :, :, 2*W:] = pred_frames_batch[:, :future_frames, :, :, :]

    batch = batch.view(N * total_frames, C, H, W * 3)

    if renorm_transform:
        batch = renorm_transform(batch)

    vutils.save_image(batch, file_dir.joinpath(f'batch_clip_{desc}.png'), nrow=total_frames, normalize=True)



def read_code_files():
    code_files = ['train_AutoEncoder.py', 'dataset.py', 'model.py', 'utils.py']
    code_snapshots = {}
    for file_name in code_files:
        if os.path.exists(file_name):
            with open(file_name, 'r') as f:
                code_snapshots[file_name] = f.read()
    return code_snapshots

def write_code_files(code_file_dict, parent_dir):
    """
    Write the saved code file dictionary to disk
    parent_dir: directory to place all the saved code files
    """
    for k, v in code_file_dict.items():
        file_path = Path(parent_dir).joinpath(k)
        if not file_path.exists():
            file_path.parent.mkdir(parents = True, exist_ok=True)
        with open(file_path, 'ab') as f:
            f.write(v)

class BatchAverageMeter(object):
    """Computes and stores the average and current value
    https://github.com/pytorch/examples/blob/cedca7729fef11c91e28099a0e45d7e98d03b66d/imagenet/main.py#L363
    """
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

def parameters_count(model):
    """
    for name, param in model.named_parameters():
        print(name, param.size())
    """
    count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters are {float(count)/1e6} Million")
    return count
    