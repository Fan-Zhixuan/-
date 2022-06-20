import argparse
import torch
import torch.nn as nn
from torch.utils import data, model_zoo
import numpy as np
import pickle
from torch.autograd import Variable
import torch.optim as optim
import scipy.misc
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import sys
import os
import os.path as osp
import random
import time
import yaml
# from tensorboardX import SummaryWriter

from trainer import AD_Trainer
from loss import CrossEntropy2d
from utils.tool import adjust_learning_rate, adjust_learning_rate_D, Timer 
from dataset.cityscapes_train_dataset import cityscapesDataSet
from dataset.robot_dataset import robotDataSet
from dataloader import HDataSet

SNAPSHOT_DIR = './snapshots/'

def get_arguments():
    """Parse all the arguments provided from the CLI.
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Unet Network")
    parser.add_argument("--snapshot-dir", type=str, default=SNAPSHOT_DIR,
                        help="Where to save snapshots of the model.")
    return parser.parse_args()


args = get_arguments()

# save opts
if not os.path.exists(args.snapshot_dir):
    os.makedirs(args.snapshot_dir)

with open('%s/opts.yaml'%args.snapshot_dir, 'w') as fp:
    yaml.dump(vars(args), fp, default_flow_style=False)
def main():
    """Create the model and start the training."""

    w, h = map(int, args.input_size.split(','))
    args.input_size = (w, h)

    w, h = map(int, args.crop_size.split(','))
    args.crop_size = (h, w)

    w, h = map(int, args.input_size_target.split(','))
    args.input_size_target = (w, h)

    cudnn.enabled = True
    cudnn.benchmark = True


    str_ids = args.gpu_ids.split(',')
    gpu_ids = []
    for str_id in str_ids:
        gid = int(str_id)
        if gid >=0:
            gpu_ids.append(gid)

    num_gpu = len(gpu_ids)
    args.multi_gpu = False
    if num_gpu>1:
        args.multi_gpu = True
        Trainer = AD_Trainer(args)
        Trainer.G = torch.nn.DataParallel( Trainer.G, gpu_ids)
        Trainer.D1 = torch.nn.DataParallel( Trainer.D1, gpu_ids)
        Trainer.D2 = torch.nn.DataParallel( Trainer.D2, gpu_ids)
    else:
        Trainer = AD_Trainer(args)

    print(Trainer)

    trainloader = data.DataLoader(
        cityscapesDataSet(args.data_dir, args.data_list, max_iters=args.num_steps * args.iter_size * args.batch_size,
                    resize_size=args.input_size,
                    crop_size=args.crop_size,
                    scale=True, mirror=True, mean=IMG_MEAN, autoaug = args.autoaug),
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)

    trainloader_iter = enumerate(trainloader)

    targetloader = data.DataLoader(robotDataSet(args.data_dir_target, args.data_list_target,
                                                     max_iters=args.num_steps * args.iter_size * args.batch_size,
                                                     resize_size=args.input_size_target,
                                                     crop_size=args.crop_size,
                                                     scale=False, mirror=args.random_mirror, mean=IMG_MEAN,
                                                     set=args.set, autoaug = args.autoaug_target),
                                   batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                                   pin_memory=True, drop_last=True)


    targetloader_iter = enumerate(targetloader)

    # set up tensor board
    if args.tensorboard:
        args.log_dir += '/'+ os.path.basename(args.snapshot_dir)
        if not os.path.exists(args.log_dir):
            os.makedirs(args.log_dir)

        writer = SummaryWriter(args.log_dir)

    for i_iter in range(args.num_steps):

        loss_seg_value1 = 0
        loss_adv_target_value1 = 0
        loss_D_value1 = 0

        loss_seg_value2 = 0
        loss_adv_target_value2 = 0
        loss_D_value2 = 0


        adjust_learning_rate(Trainer.gen_opt , i_iter, args)
        adjust_learning_rate_D(Trainer.dis1_opt, i_iter, args)
        adjust_learning_rate_D(Trainer.dis2_opt, i_iter, args)

        for sub_i in range(args.iter_size):

            # train G

            # train with source

            _, batch = trainloader_iter.__next__()
            _, batch_t = targetloader_iter.__next__()

            images, labels, _, _ = batch
            images = images.cuda()
            labels = labels.long().cuda()
            images_t, labels_t, _, _ = batch_t
            images_t = images_t.cuda()
            labels_t = labels_t.long().cuda()

            with Timer("Elapsed time in update: %f"):
                loss_seg1, loss_seg2, loss_adv_target1, loss_adv_target2, loss_me, loss_kl, pred1, pred2, pred_target1, pred_target2, val_loss = Trainer.gen_update(images, images_t, labels, labels_t, i_iter)
                loss_seg_value1 += loss_seg1.item() / args.iter_size
                loss_seg_value2 += loss_seg2.item() / args.iter_size
                loss_adv_target_value1 += loss_adv_target1 / args.iter_size
                loss_adv_target_value2 += loss_adv_target2 / args.iter_size
                loss_me_value = loss_me

                if args.lambda_adv_target1 > 0 and args.lambda_adv_target2 > 0:
                    loss_D1, loss_D2 = Trainer.dis_update(pred1, pred2, pred_target1, pred_target2)
                    loss_D_value1 += loss_D1.item()
                    loss_D_value2 += loss_D2.item()
                else:
                    loss_D_value1 = 0
                    loss_D_value2 = 0

        del pred1, pred2, pred_target1, pred_target2

        if args.tensorboard:
            scalar_info = {
                'loss_seg1': loss_seg_value1,
                'loss_seg2': loss_seg_value2,
                'loss_adv_target1': loss_adv_target_value1,
                'loss_adv_target2': loss_adv_target_value2,
                'loss_me_target': loss_me_value,
                'loss_kl_target': loss_kl,
                'loss_D1': loss_D_value1,
                'loss_D2': loss_D_value2,
                'val_loss': val_loss,
            }

            if i_iter % 100 == 0:
                for key, val in scalar_info.items():
                    writer.add_scalar(key, val, i_iter)

        print('exp = {}'.format(args.snapshot_dir))
        print(
        '\033[1m iter = %8d/%8d \033[0m loss_seg1 = %.3f loss_seg2 = %.3f loss_me = %.3f  loss_kl = %.3f loss_adv1 = %.3f, loss_adv2 = %.3f loss_D1 = %.3f loss_D2 = %.3f, val_loss=%.3f'%(i_iter, args.num_steps, loss_seg_value1, loss_seg_value2, loss_me_value, loss_kl, loss_adv_target_value1, loss_adv_target_value2, loss_D_value1, loss_D_value2, val_loss))

        # clear loss
        del loss_seg1, loss_seg2, loss_adv_target1, loss_adv_target2, loss_me, loss_kl, val_loss

        if i_iter >= args.num_steps_stop - 1:
            print('save model ...')
            torch.save(Trainer.G.state_dict(), osp.join(args.snapshot_dir, 'GTA5_' + str(args.num_steps_stop) + '.pth'))
            torch.save(Trainer.D1.state_dict(), osp.join(args.snapshot_dir, 'GTA5_' + str(args.num_steps_stop) + '_D1.pth'))
            torch.save(Trainer.D2.state_dict(), osp.join(args.snapshot_dir, 'GTA5_' + str(args.num_steps_stop) + '_D2.pth'))
            break

        if i_iter % args.save_pred_every == 0 and i_iter != 0:
            print('taking snapshot ...')
            torch.save(Trainer.G.state_dict(), osp.join(args.snapshot_dir, 'GTA5_' + str(i_iter) + '.pth'))
            torch.save(Trainer.D1.state_dict(), osp.join(args.snapshot_dir, 'GTA5_' + str(i_iter) + '_D1.pth'))
            torch.save(Trainer.D2.state_dict(), osp.join(args.snapshot_dir, 'GTA5_' + str(i_iter) + '_D2.pth'))

    if args.tensorboard:
        writer.close()


if __name__ == '__main__':
    main()