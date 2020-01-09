#!/usr/bin/env python
# coding=utf-8
"""VOC Dataset Classes

Original author: Francisco Massa
https://github.com/fmassa/vision/blob/voc_dataset/torchvision/datasets/voc.py

Updated by: Ellis Brown, Max deGroot
"""
from .config import HOME
import os.path as osp
import os
import torch
import torch.utils.data as data
import cv2
import numpy as np

VOC_CLASSES = ('带电芯充电宝', '不带电芯充电宝')

# note: if you used our download scripts, this should be right
# ./data/study
VOC_ROOT = osp.join(HOME, "data/VOCdevkit/")


class VOCAnnotationTransform(object):
    """Transforms a VOC annotation into a Tensor of bbox coords and label index
    Initilized with a dictionary lookup of classnames to indexes

    Arguments:
        class_to_ind (dict, optional): dictionary lookup of classnames -> indexes
            (default: alphabetic indexing of VOC's 20 classes)
        keep_difficult (bool, optional): keep difficult instances or not
            (default: False)
        height (int): height
        width (int): width
    """

    def __init__(self, class_to_ind=None, keep_difficult=False):
        self.class_to_ind = class_to_ind or dict(
            zip(VOC_CLASSES, range(len(VOC_CLASSES))))
        self.keep_difficult = keep_difficult

    def __call__(self, target, width, height):
        """
        Arguments:
            target (annotation) : the target annotation to be made usable
                will be an ET.Element
        Returns:
            a list containing lists of bounding boxes  [bbox coords, class name]
        """
        res = []
        with open(target, 'r', encoding='utf-8') as f1:
            dataread = f1.readlines()
        for annotation in dataread:
            temp = annotation.split()
            name = temp[1]
            # 只读两类
            if name not in VOC_CLASSES:
                continue
            xmin = int(temp[2]) / width
            # 只读取V视角的
            if xmin > 1:
                continue
            if xmin < 0:
                xmin = 0
            ymin = int(temp[3]) / height
            if ymin < 0:
                ymin = 0
            xmax = int(temp[4]) / width
            if xmax > 1:
                xmax = 1
            ymax = int(temp[5]) / height
            if ymax > 1:
                ymax = 1
            label_idx = VOC_CLASSES.index(name)
            res += [[xmin, ymin, xmax, ymax, label_idx]]
        return res


class VOCDetection(data.Dataset):
    """VOC Detection Dataset Object

    input is image, target is annotation

    Arguments:
        root (string): filepath to VOCdevkit folder.
        image_set (string): imageset to use (eg. 'train', 'val', 'test')
        transform (callable, optional): transformation to perform on the
            input image
        target_transform (callable, optional): transformation to perform on the
            target `annotation`
            (eg: take in caption string, return tensor of word indices)
        dataset_name (string, optional): which dataset to load
            (default: 'VOC2007')
    """

    def __init__(self, root,
                 # image_sets=['core_3000', 'coreless_3000'],
                 transform=None, target_transform=VOCAnnotationTransform(),
                 dataset_name='ChargePal'):
        self.root = root
        # self.image_set = image_sets
        self.transform = transform
        self.target_transform = target_transform
        self.name = dataset_name
        self.ids = list()
        self.anno_path_arr = list()
        if not isinstance(root, (list, tuple)):
            root = [root]
        idx = 0
        for dir_name in root:
            img_dir = os.path.join(dir_name, 'Image')
            anno_dir = os.path.join(dir_name, 'Annotation')
            for name in sorted(os.listdir(img_dir)):
                if not name.endswith('jpg'):
                    continue
                anno_path = os.path.join(anno_dir, name).replace('.jpg', '.txt')
                if not os.path.isfile(anno_path):
                    continue
                img_path = os.path.join(img_dir, name)
                try:
                    img = cv2.imread(img_path)
                    height, width, channels = img.shape
                except Exception:
                    print('>>>error image>>>> {}'.format(img_path))
                    continue
                self.ids.append(img_path)
                self.anno_path_arr.append(anno_path)
                try:
                    self.pull_item(idx)
                    idx += 1
                except Exception:
                    self.ids = self.ids[:-1]
                    self.anno_path_arr = self.anno_path_arr[:-1]

    def __getitem__(self, index):
        im, gt, h, w = self.pull_item(index)

        return im, gt

    def __len__(self):
        return len(self.ids)

    def pull_item(self, index):
        img_id = self.ids[index]

        target = self.anno_path_arr[index]
        img = cv2.imread(img_id)
        height, width, channels = img.shape

        if self.target_transform is not None:
            target = self.target_transform(target, width, height)

        if self.transform is not None:
            target = np.array(target)
            img, boxes, labels = self.transform(img, target[:, :4], target[:, 4])
            # to rgb
            img = img[:, :, (2, 1, 0)]
            # img = img.transpose(2, 0, 1)
            target = np.hstack((boxes, np.expand_dims(labels, axis=1)))
        return torch.from_numpy(img).permute(2, 0, 1), target, height, width
        # return torch.from_numpy(img), target, height, width

    def pull_image(self, index):
        '''Returns the original image object at index in PIL form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            PIL img
        '''
        img_id = self.ids[index]
        return cv2.imread(img_id, cv2.IMREAD_COLOR)

    def get_image_name(self, index):
        img_id = self.ids[index]
        return img_id.split('/')[-1].split('.')[0]

    def pull_anno(self, index):
        '''Returns the original annotation of image at index

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to get annotation of
        Return:
            list:  [img_id, [(label, bbox coords),...]]
                eg: ('001718', [('dog', (96, 13, 438, 332))])
        '''
        img_id = self.ids[index]
        gt = self.target_transform(self.anno_path_arr[index], 1, 1)
        return img_id, gt

    def pull_tensor(self, index):
        '''Returns the original image at an index in tensor form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            tensorized version of img, squeezed
        '''
        return torch.Tensor(self.pull_image(index)).unsqueeze_(0)
