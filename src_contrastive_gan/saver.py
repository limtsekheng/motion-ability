import os
import torchvision
from tensorboardX import SummaryWriter
import numpy as np
from PIL import Image
from ezc3d import c3d

# tensor to PIL Image
def tensor2img(img):
  img = img[0].cpu().float().numpy()
  if img.shape[0] == 1:
    img = np.tile(img, (3, 1, 1))
  img = (np.transpose(img, (1, 2, 0)) + 1) / 2.0 * 255.0
  return img.astype(np.uint8)

# save a set of images
def save_imgs(imgs, names, path):
  if not os.path.exists(path):
    os.mkdir(path)
  for img, name in zip(imgs, names):
    img = tensor2img(img)
    img = Image.fromarray(img)
    img.save(os.path.join(path, name + '.png'))

class Saver():
  def __init__(self, opts):
    self.display_dir = os.path.join(opts.display_dir, opts.name)
    self.model_dir = os.path.join(opts.result_dir, opts.name)
    self.image_dir = os.path.join(self.model_dir, 'images')
    self.display_freq = opts.display_freq
    self.img_save_freq = opts.img_save_freq
    self.model_save_freq = opts.model_save_freq

    # make directory
    if not os.path.exists(self.display_dir):
      os.makedirs(self.display_dir)
    if not os.path.exists(self.model_dir):
      os.makedirs(self.model_dir)
    if not os.path.exists(self.image_dir):
      os.makedirs(self.image_dir)

    # create tensorboard writer
    self.writer = SummaryWriter(logdir=self.display_dir)

  # write losses and images to tensorboard
  def write_display(self, total_it, model):
    if (total_it + 1) % self.display_freq == 0:
      # write loss
      members = [attr for attr in dir(model) if not callable(getattr(model, attr)) and not attr.startswith("__") and 'loss' in attr]
      for m in members:
        self.writer.add_scalar(m, getattr(model, m), total_it)
      # write img
      image_dis = torchvision.utils.make_grid(model.image_display, nrow=model.image_display.size(0)//2)/2 + 0.5
      self.writer.add_image('Image', image_dis, total_it)

  # save result images
  def write_img(self, ep, model):
    if (ep + 1) % self.img_save_freq == 0:
      assembled_images = model.assemble_outputs()

      for i in range(6):
        img_filename = '%s/gen_%05d_%d.c3d' % (self.image_dir, ep, i)
        # torchvision.utils.save_image(assembled_images / 2 + 0.5, img_filename, nrow=1)
        new = c3d()
        new['parameters']['POINT']['RATE']['value'] = [200]
        new['parameters']['POINT']['LABELS']['value'] = [str(x) for x in range(72)]
        new['data']['points'] = np.ones((4,72,200))
        new['data']['points'][0:3,:,:] = assembled_images[i].astype('float64') * 1500
        new.write(img_filename)

    # elif ep == -1:
    #   assembled_images = model.assemble_outputs().astype('float64')
    #   img_filename = '%s/gen_last.jpg' % (self.image_dir, ep)
    #   # torchvision.utils.save_image(assembled_images / 2 + 0.5, img_filename, nrow=1)
    #   new = c3d()
    #   new['parameters']['POINT']['RATE']['value'] = [200]
    #   new['parameters']['POINT']['LABELS']['value'] = [str(x) for x in range(72)]
    #   new['data']['points'] = assembled_images
    #   new.write(img_filename)

  # save model
  def write_model(self, ep, total_it, model):
    if (ep + 1) % self.model_save_freq == 0:
      print('--- save the model @ ep %d ---' % (ep))
      model.save('%s/%05d.pth' % (self.model_dir, ep), ep, total_it)
    elif ep == -1:
      model.save('%s/last.pth' % self.model_dir, ep, total_it)

