#!/usr/bin/env python
# coding=utf-8
from data import *
from utils.augmentations import SSDAugmentation
from layers.modules import MultiBoxLoss
from ssd import build_ssd
import os
import datetime
import copy
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.utils.data as data
from torch.optim import lr_scheduler
import argparse


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


parser = argparse.ArgumentParser(
    description='Single Shot MultiBox Detector Training With Pytorch')
train_set = parser.add_mutually_exclusive_group()
parser.add_argument('--dataset_root', default=VOC_ROOT,
                    help='Dataset root directory path')
parser.add_argument('--basenet', default='vgg16_reducedfc.pth',
                    help='Pretrained base model')
parser.add_argument('--batch_size', default=4, type=int,
                    help='Batch size for training')
parser.add_argument('--resume', default=None, type=str,
                    help='Checkpoint state_dict file to resume training from')
parser.add_argument('--start_iter', default=0, type=int,
                    help='Resume training at this iter')
parser.add_argument('--num_workers', default=4, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--cuda', default=False, type=str2bool,
                    help='Use CUDA to train model')
parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='Momentum value for optim')
parser.add_argument('--weight_decay', default=5e-4, type=float,
                    help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float,
                    help='Gamma update for SGD')
parser.add_argument('--epochs', default=2, type=int, help='recycle times')
parser.add_argument('--visdom', default=False, type=str2bool,
                    help='Use visdom for loss visualization')
parser.add_argument('--save_folder', default='weights/',
                    help='Directory for saving checkpoint models')
args = parser.parse_args()


if torch.cuda.is_available():
    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if not args.cuda:
        print("WARNING: It looks like you have a CUDA device, but aren't " +
              "using CUDA.\nRun with --cuda for optimal training speed.")
        torch.set_default_tensor_type('torch.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)


def train():
    cfg = voc
    dataset = VOCDetection(args.dataset_root, transform=SSDAugmentation(cfg['min_dim'], MEANS))

    # indices = torch.randperm(len(dataset)).tolist()
    # dataset = torch.utils.data.Subset(dataset, indices[:60])

    ssd_net = build_ssd('train', cfg['min_dim'], cfg['num_classes'])
    net = ssd_net

    if args.cuda:
        net = torch.nn.DataParallel(ssd_net)
        cudnn.benchmark = True

    vgg_weights = torch.load(args.save_folder + args.basenet)
    print('Loading base network...')
    ssd_net.vgg.load_state_dict(vgg_weights)

    if args.cuda:
        net = net.cuda()

    ssd_net.extras.apply(weights_init)
    ssd_net.loc.apply(weights_init)
    ssd_net.conf.apply(weights_init)

    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    criterion = MultiBoxLoss(cfg['num_classes'], 0.5, True, 0, True, 3, 0.5, False, args.cuda)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=args.gamma)

    net.train()
    # print('Training SSD on:', dataset.name)
    print('Using the specified args:')
    print(args)

    _pre = 'gpu-' if args.cuda else 'cpu-'
    model_path = os.path.join(args.save_folder, _pre + datetime.datetime.now().strftime('%Y%m%d%H%M%S') + '.pth')
    best_loss = 1000
    best_model_wts = copy.deepcopy(ssd_net.state_dict())

    # os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    data_loader = data.DataLoader(dataset, args.batch_size, num_workers=args.num_workers, shuffle=True, collate_fn=detection_collate, pin_memory=True)
    num_epochs = args.epochs
    for epoch in range(num_epochs):
        exp_lr_scheduler.step()
        # loss counters
        count = 0
        loss_value = 0.0
        for images, targets in data_loader:
            if args.cuda:
                images = Variable(images.cuda())
                targets = [Variable(ann.cuda(), volatile=True) for ann in targets]
            else:
                images = Variable(images)
                targets = [Variable(ann, volatile=True) for ann in targets]
            # forward
            out = net(images)
            # backprop
            optimizer.zero_grad()
            loss_l, loss_c = criterion(out, targets)
            loss = loss_l + loss_c
            loss.backward()
            optimizer.step()
            print('epoch %d  count: %d || Loss: %.4f || best_loss: %.4f' % (epoch, count, loss.item(), best_loss))
            # t1 = time.time()
            # loc_loss += loss_l.item()
            # conf_loss += loss_c.item()
            count += 1
            loss_value += loss.item()

        # 每次都保存一下
        torch.save(ssd_net.state_dict(), model_path.replace('.pth', '-{}.pth'.format(epoch)))

        loss_value = loss_value / count
        print('epoch %d  loss_value: %.4f || best_loss: %.4f ||' % (epoch, loss_value, best_loss))
        if loss_value < best_loss:
            best_loss = loss_value
            best_model_wts = copy.deepcopy(ssd_net.state_dict())

    print('Finished! loss_value: {:.4f} , save model to {}'.format(best_loss, model_path))
    torch.save(best_model_wts, model_path)


def xavier(param):
    init.xavier_uniform(param)


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        xavier(m.weight.data)
        m.bias.data.zero_()


if __name__ == '__main__':
    train()
