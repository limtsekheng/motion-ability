import torch
from options import TrainOptions
from dataset import dataset_unpair
from model import DRIT
from saver import Saver
from torch.utils.tensorboard import SummaryWriter

def main():
  # parse options
  parser = TrainOptions()
  opts = parser.parse()

  # daita loader
  print('\n--- load dataset ---')
  dataset = dataset_unpair(opts)
  train_loader = torch.utils.data.DataLoader(dataset, batch_size=opts.batch_size, shuffle=True, num_workers=opts.nThreads)

  # model
  print('\n--- load model ---')
  model = DRIT(opts)
  model.setgpu(opts.gpu)
  if opts.resume is None:
    model.initialize()
    ep0 = -1
    total_it = 0
  else:
    ep0, total_it = model.resume(opts.resume)
  model.set_scheduler(opts, last_ep=ep0)
  ep0 += 1
  print('start the training at epoch %d'%(ep0))

  # saver for display and output
  saver = Saver(opts)

  # train
  print('\n--- train ---')
  torch.autograd.set_detect_anomaly(True)   #---------------------------------------
  max_it = 500000

  writer = SummaryWriter()
  
  for ep in range(ep0, opts.n_ep):
    for it, (images_a, labels_a, images_b, labels_b) in enumerate(train_loader):
      if images_a.size(0) != opts.batch_size or images_b.size(0) != opts.batch_size:
        continue

      # input data
      images_a = images_a.cuda(opts.gpu).detach()
      images_b = images_b.cuda(opts.gpu).detach()
      # print(len(train_loader))

      # # update model
      # if (it + 1) % opts.d_iter != 0 and it < len(train_loader) - 2:
      #   model.update_D_content(images_a, images_b)
      #   continue
      # else:
      #   model.update_D(images_a, images_b)
      #   model.update_EG()

      model.input_for_forward(images_a, labels_a, images_b, labels_b)
      model.forward()
      model.update_EG()
      model.update_D(images_a, images_b)

      # save to display file
      # if not opts.no_display_img:
      #   saver.write_display(total_it, model)

      print('total_it: %d (ep %d, it %d), lr %08f' % (total_it, ep, it, model.gen_opt.param_groups[0]['lr']))
      total_it += 1
      if total_it >= max_it:
        saver.write_img(-1, model)
        saver.write_model(-1, model)
        break

    # decay learning rate
    if opts.n_ep_decay > -1:
      model.update_lr()

    # save result image
    saver.write_img(ep, model)

    # Save network weights
    saver.write_model(ep, total_it, model)

    writer.add_scalar('Loss/train', model.G_loss, ep)
    writer.add_scalar('ContrastiveLoss/train', model.contrastive_loss, ep)
    writer.add_scalar('CrossContentLoss/train', model.cross_content_A_loss + model.cross_content_B_loss, ep)
    writer.add_scalar('SelfReconLoss/train', model.l1_recon_AA_loss + model.l1_recon_BB_loss, ep)
    writer.add_scalar('CrossReconLoss/train', model.l1_recon_A_loss + model.l1_recon_B_loss, ep)

  writer.flush()
  writer.close()

  return

if __name__ == '__main__':
  main()
