import os
import torch.utils.data as data
from PIL import Image
from torchvision.transforms import Compose, Resize, RandomCrop, CenterCrop, RandomHorizontalFlip, ToTensor, Normalize
import random
from ezc3d import c3d
import numpy as np
import torch

# class dataset_single(data.Dataset):
#   def __init__(self, opts, setname, input_dim):
#     self.dataroot = opts.dataroot
#     images = os.listdir(os.path.join(self.dataroot, opts.phase + setname))
#     self.img = [os.path.join(self.dataroot, opts.phase + setname, x) for x in images]
#     self.size = len(self.img)
#     self.input_dim = input_dim

#     # setup image transformation
#     transforms = [Resize((opts.resize_size, opts.resize_size), Image.BICUBIC)]
#     transforms.append(CenterCrop(opts.crop_size))
#     transforms.append(ToTensor())
#     transforms.append(Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))
#     self.transforms = Compose(transforms)
#     print('%s: %d images'%(setname, self.size))
#     return

#   def __getitem__(self, index):
#     data = self.load_img(self.img[index], self.input_dim)
#     return data

#   def load_img(self, img_name, input_dim):
#     img = Image.open(img_name).convert('RGB')
#     img = self.transforms(img)
#     if input_dim == 1:
#       img = img[0, ...] * 0.299 + img[1, ...] * 0.587 + img[2, ...] * 0.114
#       img = img.unsqueeze(0)
#     return img

#   def __len__(self):
#     return self.size

class dataset_unpair(data.Dataset):
  def __init__(self, opts):
    self.dataroot = opts.dataroot

    # A
    images_A = os.listdir(os.path.join(self.dataroot, opts.phase + 'A'))
    self.A = [os.path.join(self.dataroot, opts.phase + 'A', x) for x in images_A if x.endswith('.c3d')]

    # B
    images_B = os.listdir(os.path.join(self.dataroot, opts.phase + 'B'))
    self.B = [os.path.join(self.dataroot, opts.phase + 'B', x) for x in images_B if x.endswith('.c3d')]

    self.A_size = len(self.A)
    self.B_size = len(self.B)
    self.dataset_size = max(self.A_size, self.B_size)
    self.input_dim_A = opts.input_dim_a
    self.input_dim_B = opts.input_dim_b

    # setup image transformation
    # transforms = [Resize((opts.resize_size, opts.resize_size), Image.BICUBIC)]
    # if opts.phase == 'train':
    #   transforms.append(RandomCrop(opts.crop_size))
    # else:
    #   transforms.append(CenterCrop(opts.crop_size))
    # if not opts.no_flip:
    #   transforms.append(RandomHorizontalFlip())
    # transforms.append(ToTensor())
    # transforms.append(Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))
    # self.transforms = Compose(transforms)
    # print('A: %d, B: %d images'%(self.A_size, self.B_size))
    return

  def __getitem__(self, index):
    if self.dataset_size == self.A_size:
      data_A = self.load_img(self.A[index], self.input_dim_A)
      data_B = self.load_img(self.B[random.randint(0, self.B_size - 1)], self.input_dim_B)
      label_A = self.load_label(self.A[index])
      label_B = self.load_label(self.B[random.randint(0, self.B_size - 1)])
    else:
      data_A = self.load_img(self.A[random.randint(0, self.A_size - 1)], self.input_dim_A)
      data_B = self.load_img(self.B[index], self.input_dim_B)
      label_A = self.load_label(self.A[random.randint(0, self.A_size - 1)])
      label_B = self.load_label(self.B[index])
    return data_A, label_A, data_B, label_B



  def load_img(self, img_name, input_dim):
    # img = Image.open(img_name).convert('RGB')
    # img = self.transforms(img)
    # if input_dim == 1:
    #   img = img[0, ...] * 0.299 + img[1, ...] * 0.587 + img[2, ...] * 0.114
    #   img = img.unsqueeze(0)
    # return img
    sequence = c3d(img_name)
    point_data = sequence['data']['points'][0:3,:,:]/1500
    img = torch.from_numpy(point_data).float()
    return img



  def load_label(self, img_name):
    filename = os.path.split(img_name)[1]
    name = filename.split('_')
    return name[0]


  def __len__(self):
    return self.dataset_size
