#!/usr/bin/env python
# coding=utf-8
import os
import cv2
import argparse
import torch
from PIL import Image, ImageDraw, ImageFont
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from data import VOC_ROOT, VOC_CLASSES as labelmap
from data import VOCAnnotationTransform, VOCDetection, BaseTransform, VOC_CLASSES
from ssd import build_ssd

parser = argparse.ArgumentParser(description='Single Shot MultiBox Detection')
parser.add_argument('--trained_model', default='weights/ssd_300_VOC0712.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--save_folder', default='eval/', type=str,
                    help='Dir to save results')
parser.add_argument('--visual_threshold', default=0.6, type=float,
                    help='Final confidence threshold')
parser.add_argument('--cuda', default=False, type=bool,
                    help='Use cuda to train model')
parser.add_argument('--voc_root', default=VOC_ROOT, help='Location of VOC root directory')
parser.add_argument('-f', default=None, type=str, help="Dummy arg so we can load in Jupyter Notebooks")
args = parser.parse_args()

if args.cuda and torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)


def test_net(save_folder, net, cuda, testset, transform, thresh):
    # dump predictions and assoc. ground truth to text file for now
    filename = save_folder+'test1.txt'
    num_images = len(testset)
    for i in range(num_images):
        print('Testing image {:d}/{:d}....'.format(i+1, num_images))
        img = testset.pull_image(i)
        img_id, annotation = testset.pull_anno(i)
        x = torch.from_numpy(transform(img)[0]).permute(2, 0, 1)
        x = Variable(x.unsqueeze(0))

        with open(filename, mode='a') as f:
            f.write('\nGROUND TRUTH FOR: '+img_id+'\n')
            for box in annotation:
                f.write('label: '+' || '.join(str(b) for b in box)+'\n')
        if cuda:
            x = x.cuda()

        y = net(x)      # forward pass
        detections = y.data

        im_det = img.copy()
        h, w, _ = im_det.shape
        # im_det = Image.open(img_id)
        # draw = ImageDraw.Draw(im_det)
        # w, h = im_det.size

        need_save = False
        # scale each detection back up to the image
        scale = torch.Tensor([w, h, w, h])
        for i in range(detections.size(1)):
            j = 0
            while detections[0, i, j, 0] >= 0.6:
                item = (detections[0, i, j, 1:]*scale).cpu().numpy()
                item = [int(n) for n in item]
                chinese = labelmap[i-1]
                # print(chinese+'gt\n\n')
                if chinese[0] == '带':
                    chinese = 'P_Battery_Core'
                else:
                    chinese = 'P_Battery_No_Core'

                cv2.rectangle(im_det, (item[0], item[1]), (item[2], item[3]), (0, 255, 255), 2)
                cv2.putText(im_det, chinese, (item[0], item[1] - 5), 0, 0.6, (0, 255, 255), 2)
                # draw.rectangle((item[0], item[1], item[2], item[3]), outline='red', width=2)
                # draw.textsize()
                # draw.text((item[0], item[1] - 5), chinese, font=font, fill='red')
                need_save = True
                j += 1

        if need_save:
            dst_path = img_id.replace('Image', 'ImageTarget')
            dst_dir = os.path.dirname(dst_path)
            if not os.path.exists(dst_dir):
                os.makedirs(dst_dir)
            # im_det.save(dst_path)
            cv2.imwrite(dst_path, im_det)


def test_voc():
    # load net
    num_classes = len(VOC_CLASSES) + 1 # +1 background
    net = build_ssd('test', 300, num_classes) # initialize SSD
    net.load_state_dict(torch.load(args.trained_model))
    net.eval()
    print('Finished loading model!')
    # load data
    testset = VOCDetection(args.voc_root, target_transform=VOCAnnotationTransform())
    if args.cuda:
        net = net.cuda()
        cudnn.benchmark = True
    # evaluation
    test_net(args.save_folder, net, args.cuda, testset,
             BaseTransform(net.size, (104, 117, 123)),
             thresh=args.visual_threshold)

if __name__ == '__main__':
    test_voc()
