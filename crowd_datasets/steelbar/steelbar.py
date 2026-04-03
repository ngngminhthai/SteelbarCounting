import os
import random
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import cv2
import glob
import json
import torchvision.transforms as T
import torchvision.transforms.functional as TF

class Steelbar(Dataset):
    def __init__(self, data_root, transform=None, train=False, patch=False, flip=False):
        self.root_path = data_root
        self.img_dir = os.path.join(self.root_path, 'images')
        self.lbl_dir = os.path.join(self.root_path, 'labels')

        # load image names
        img_files = sorted(glob.glob(os.path.join(self.img_dir, '*.jpg')))
        
        # simple split, e.g., 80% train, 20% val
        split_idx = int(len(img_files) * 0.8)
        if train:
            self.img_list = img_files[:split_idx]
        else:
            self.img_list = img_files[split_idx:]

        self.nSamples = len(self.img_list)
        self.transform = transform
        self.train = train
        self.patch = patch
        self.flip = flip
        self.color_jitter = T.ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05
        ) if train else None

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'

        img_path = self.img_list[index]
        img_name = os.path.basename(img_path)
        lbl_name = img_name.replace('.jpg', '.json')
        gt_path = os.path.join(self.lbl_dir, lbl_name)
        
        # load image and ground truth
        img, point = load_data((img_path, gt_path), self.train)

        if self.train:
            img, point = self._augment(img, point)

        if self.transform is not None:
            img = self.transform(img)

        point = [point]
        img = torch.as_tensor(img, dtype=torch.float32)
        # pack up related infos
        target = [{} for i in range(len(point))]
        for i, _ in enumerate(point):
            target[i]['point'] = torch.Tensor(point[i])
            image_id = index
            image_id = torch.Tensor([image_id]).long()
            target[i]['image_id'] = image_id
            target[i]['labels'] = torch.ones([point[i].shape[0]]).long()

        return img, target

    def _augment(self, img, point):
        width, height = img.size

        # Random 90-degree rotations preserve geometry for point counting.
        angle = random.choice([0, 90, 180, 270])
        if angle:
            img = TF.rotate(img, angle, expand=True)
            point = self._rotate_points(point, width, height, angle)
            width, height = img.size

        if random.random() > 0.5:
            img = TF.hflip(img)
            if len(point) > 0:
                point[:, 0] = width - point[:, 0]

        if random.random() > 0.5:
            img = TF.vflip(img)
            if len(point) > 0:
                point[:, 1] = height - point[:, 1]

        if self.color_jitter is not None:
            img = self.color_jitter(img)

        return img, point

    @staticmethod
    def _rotate_points(point, width, height, angle):
        if len(point) == 0:
            return point

        rotated = point.copy()
        x = point[:, 0].copy()
        y = point[:, 1].copy()

        if angle == 90:
            rotated[:, 0] = y
            rotated[:, 1] = width - x
        elif angle == 180:
            rotated[:, 0] = width - x
            rotated[:, 1] = height - y
        elif angle == 270:
            rotated[:, 0] = height - y
            rotated[:, 1] = x

        return rotated

def load_data(img_gt_path, train):
    img_path, gt_path = img_gt_path
    img = cv2.imread(img_path)
    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
    points = []
    with open(gt_path, 'r') as f_label:
        data = json.load(f_label)
        if "points" in data:
            points = data["points"]
            
    # Ensure shape is (N, 2) even if points is empty
    points_arr = np.array(points, dtype=np.float32)
    if len(points_arr) == 0:
        points_arr = np.zeros((0, 2), dtype=np.float32)
        
    return img, points_arr

# random crop augmentation
def random_crop(img, den, num_patch=4):
    half_h = 128
    half_w = 128
    result_img = np.zeros([num_patch, img.shape[0], half_h, half_w])
    result_den = []
    # crop num_patch for each image
    for i in range(num_patch):
        start_h = random.randint(0, img.size(1) - half_h)
        start_w = random.randint(0, img.size(2) - half_w)
        end_h = start_h + half_h
        end_w = start_w + half_w
        # copy the cropped rect
        result_img[i] = img[:, start_h:end_h, start_w:end_w]
        
        # copy the cropped points
        if len(den) > 0:
            idx = (den[:, 0] >= start_w) & (den[:, 0] <= end_w) & (den[:, 1] >= start_h) & (den[:, 1] <= end_h)
            record_den = den[idx].copy()
            record_den[:, 0] -= start_w
            record_den[:, 1] -= start_h
        else:
            record_den = np.zeros((0, 2))

        result_den.append(record_den)

    return result_img, result_den
