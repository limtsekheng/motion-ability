import  torch
import torch.nn as nn
from torch.autograd import Variable
import functools
from torch.optim import lr_scheduler
import torch.nn.functional as F

####################################################################
#------------------------- Discriminators --------------------------
####################################################################
class Dis_content(nn.Module):

  def __init__(self):
    super(Dis_content, self).__init__()
    model = [ResBlock_start(channel_in = 8, channel_out = 4)]
    self.model = nn.Sequential(*model)

  def forward(self, x):
    x = x.view(x.size(0), x.size(1), 1)
    output = self.model(x)
    output = output.view(-1)
    outs = []
    outs.append(output)
    return outs


class Dis_domain(nn.Module):

  def __init__(self):
    super(Dis_domain, self).__init__()

    model = [ResBlock_start(channel_in = 216, channel_out = 256)] # Nx256x100
    model += [ResBlock(channel_in = 256, channel_out = 1024, stride_1 = 2, padding_1 = 1, stride_3 = 2, padding_3 = 0)]   # Nx1024x50
    model += [ResBlock(channel_in = 1024, channel_out = 1024, stride_1 = 2, padding_1 = 1, stride_3 = 2, padding_3 = 0)]    # Nx1024x25
    model += [ResBlock(channel_in = 1024, channel_out = 1024, stride_1 = 2, padding_1 = 1, stride_3 = 2, padding_3 = 0)]    # Nx1024x13
    model += [ResBlock(channel_in = 1024, channel_out = 1024)]    # Nx1024x13
    model += [nn.LeakyReLU(negative_slope=0.1, inplace=False)]  # Nx1024x13
    model += [nn.Conv1d(in_channels = 1024, out_channels = 1, kernel_size = 3, stride = 1, padding = 1)]
    self.model = nn.Sequential(*model)   

  def forward(self, x):
    output = self.model(x)
    outs = []
    outs.append(output)
    return outs


####################################################################
#---------------------------- Encoders -----------------------------
####################################################################
class E_content(nn.Module):

  def __init__(self):
    super(E_content, self).__init__()
    model_a = [ResBlock_start(channel_in = 216, channel_out = 256)] # Nx256x100
    model_a += [ResBlock(channel_in = 256, channel_out = 1024, stride_1 = 2, padding_1 = 1, stride_3 = 2, padding_3 = 0)]   # Nx1024x50
    model_a += [ResBlock(channel_in = 1024, channel_out = 1024, stride_1 = 2, padding_1 = 1, stride_3 = 2, padding_3 = 0)]    # Nx1024x25
    model_a += [ResBlock(channel_in = 1024, channel_out = 1024, stride_1 = 2, padding_1 = 1, stride_3 = 2, padding_3 = 0)]    # Nx1024x13
    model_a += [ResBlock(channel_in = 1024, channel_out = 1024)]    # Nx1024x13
    model_a += [ResBlock(channel_in = 1024, channel_out = 1024)]    # Nx1024x13
    model_a += [nn.AdaptiveAvgPool1d(1)]    # Nx1024x1
    self.model_a = nn.Sequential(*model_a)
    self.fc_a = nn.Sequential(*[nn.Linear(1024, 8)])

    model_b = [ResBlock_start(channel_in = 216, channel_out = 256)] # Nx256x100
    model_b += [ResBlock(channel_in = 256, channel_out = 1024, stride_1 = 2, padding_1 = 1, stride_3 = 2, padding_3 = 0)]   # Nx1024x50
    model_b += [ResBlock(channel_in = 1024, channel_out = 1024, stride_1 = 2, padding_1 = 1, stride_3 = 2, padding_3 = 0)]    # Nx1024x25
    model_b += [ResBlock(channel_in = 1024, channel_out = 1024, stride_1 = 2, padding_1 = 1, stride_3 = 2, padding_3 = 0)]    # Nx1024x13
    model_b += [ResBlock(channel_in = 1024, channel_out = 1024)]    # Nx1024x13
    model_b += [ResBlock(channel_in = 1024, channel_out = 1024)]    # Nx1024x13
    model_b += [nn.AdaptiveAvgPool1d(1)]    # Nx1024x1
    self.model_b = nn.Sequential(*model_b)
    self.fc_b = nn.Sequential(*[nn.Linear(1024, 8)])

  def forward(self, xa, xb):
    outputA = self.model_a(xa)
    outputA = outputA.view(xa.size(0), -1)
    outputA = self.fc_a(outputA)
    outputB = self.model_b(xb)
    outputB = outputB.view(xb.size(0), -1)
    outputB = self.fc_b(outputB)
    return outputA, outputB


  def forward_a(self, xa):
    outputA = self.model_a(xa)
    outputA = outputA.view(xa.size(0), -1)
    outputA = self.fc_a(outputA)
    return outputA

  def forward_b(self, xb):
    outputB = self.model_b(xb)
    outputB = outputB.view(xb.size(0), -1)
    outputB = self.fc_b(outputB)
    return outputB


class E_attr(nn.Module):

  def __init__(self):
    super(E_attr, self).__init__()
    model_a = [ResBlock_start(channel_in = 216, channel_out = 256)] # Nx256x100
    model_a += [ResBlock(channel_in = 256, channel_out = 1024, stride_1 = 2, padding_1 = 1, stride_3 = 2, padding_3 = 0)]   # Nx1024x50
    model_a += [ResBlock(channel_in = 1024, channel_out = 1024, stride_1 = 2, padding_1 = 1, stride_3 = 2, padding_3 = 0)]    # Nx1024x25
    model_a += [ResBlock(channel_in = 1024, channel_out = 1024, stride_1 = 2, padding_1 = 1, stride_3 = 2, padding_3 = 0)]    # Nx1024x13
    model_a += [ResBlock(channel_in = 1024, channel_out = 1024)]    # Nx1024x13
    model_a += [ResBlock(channel_in = 1024, channel_out = 1024)]    # Nx1024x13
    model_a += [nn.AdaptiveAvgPool1d(1)]    # Nx1024x1
    self.model_a = nn.Sequential(*model_a)    
    self.fc_mean_a = nn.Sequential(*[nn.Linear(1024, 8)])
    self.fc_var_a = nn.Sequential(*[nn.Linear(1024, 8)])

    model_b = [ResBlock_start(channel_in = 216, channel_out = 256)] # Nx256x100
    model_b += [ResBlock(channel_in = 256, channel_out = 1024, stride_1 = 2, padding_1 = 1, stride_3 = 2, padding_3 = 0)]   # Nx1024x50
    model_b += [ResBlock(channel_in = 1024, channel_out = 1024, stride_1 = 2, padding_1 = 1, stride_3 = 2, padding_3 = 0)]    # Nx1024x25
    model_b += [ResBlock(channel_in = 1024, channel_out = 1024, stride_1 = 2, padding_1 = 1, stride_3 = 2, padding_3 = 0)]    # Nx1024x13
    model_b += [ResBlock(channel_in = 1024, channel_out = 1024)]    # Nx1024x13
    model_b += [ResBlock(channel_in = 1024, channel_out = 1024)]    # Nx1024x13
    model_b += [nn.AdaptiveAvgPool1d(1)]    # Nx1024x1
    self.model_b = nn.Sequential(*model_b)    
    self.fc_mean_b = nn.Sequential(*[nn.Linear(1024, 8)])
    self.fc_var_b = nn.Sequential(*[nn.Linear(1024, 8)])

  def forward(self, xa, xb):
    x_conv_a = self.model_a(xa)
    x_conv_a_flat = x_conv_a.view(xa.size(0), -1)    # Nx1024
    out_mean_a = self.fc_mean_a(x_conv_a_flat)   # Nx8
    out_var_a = self.fc_var_a(x_conv_a_flat)
    x_conv_b = self.model_b(xb)
    x_conv_b_flat = x_conv_b.view(xb.size(0), -1)    # Nx1024
    out_mean_b = self.fc_mean_b(x_conv_b_flat)   # Nx8
    out_var_b = self.fc_var_b(x_conv_b_flat)
    return out_mean_a, out_var_a, out_mean_b, out_var_b

  def forward_a(self, xa):
    x_conv_a = self.model_a(xa)
    x_conv_a_flat = x_conv_a.view(xa.size(0), -1)    # Nx1024
    out_mean_a = self.fc_mean_a(x_conv_a_flat)   # Nx8
    out_var_a = self.fc_var_a(x_conv_a_flat)
    return out_mean_a, out_var_a

  def forward_b(self, xb):
    x_conv_b = self.model_b(xb)
    x_conv_b_flat = x_conv_b.view(xb.size(0), -1)    # Nx1024
    out_mean_b = self.fc_mean_b(x_conv_b_flat)   # Nx8
    out_var_b = self.fc_var_b(x_conv_b_flat)
    return out_mean_b, out_var_b


####################################################################
#--------------------------- Generators ----------------------------
####################################################################

class G(nn.Module):

  def __init__(self):
    super(G, self).__init__()

    fc_a = [nn.Linear(8,1024)]
    self.fc_a = nn.Sequential(*fc_a)
    dec0_a = [nn.ConvTranspose1d(in_channels=1024, out_channels=1024, kernel_size=4, stride=2, padding=1, output_padding = 1)]
    dec0_a += [nn.ConvTranspose1d(in_channels=1024, out_channels=1024, kernel_size=4, stride=2, padding=1, output_padding = 1)]
    dec0_a += [nn.ConvTranspose1d(in_channels=1024, out_channels=1016, kernel_size=3, stride=2, padding=1, output_padding = 0)]
    self.dec0_a = nn.Sequential(*dec0_a)
    dec1_a = [nn.ConvTranspose1d(in_channels=1024, out_channels=504, kernel_size=3, stride=2, padding=1, output_padding = 0)]
    self.dec1_a = nn.Sequential(*dec1_a)
    dec2_a = [nn.ConvTranspose1d(in_channels=512, out_channels=248, kernel_size=3, stride=2, padding=1, output_padding = 1)]
    self.dec2_a = nn.Sequential(*dec2_a)
    dec3_a = [nn.ConvTranspose1d(in_channels=256, out_channels=208, kernel_size=3, stride=2, padding=1, output_padding = 1)]
    self.dec3_a = nn.Sequential(*dec3_a)
    dec4_a = [nn.ConvTranspose1d(in_channels=216, out_channels=216, kernel_size=3, stride=2, padding=1, output_padding = 1)]
    self.dec4_a = nn.Sequential(*dec4_a)

    fc_b = [nn.Linear(8,1024)]
    self.fc_b = nn.Sequential(*fc_b)
    dec0_b = [nn.ConvTranspose1d(in_channels=1024, out_channels=1024, kernel_size=4, stride=2, padding=1, output_padding = 1)]
    dec0_b += [nn.ConvTranspose1d(in_channels=1024, out_channels=1024, kernel_size=4, stride=2, padding=1, output_padding = 1)]
    dec0_b += [nn.ConvTranspose1d(in_channels=1024, out_channels=1016, kernel_size=3, stride=2, padding=1, output_padding = 0)]
    self.dec0_b = nn.Sequential(*dec0_b)
    dec1_b = [nn.ConvTranspose1d(in_channels=1024, out_channels=504, kernel_size=3, stride=2, padding=1, output_padding = 0)]
    self.dec1_b = nn.Sequential(*dec1_b)
    dec2_b = [nn.ConvTranspose1d(in_channels=512, out_channels=248, kernel_size=3, stride=2, padding=1, output_padding = 1)]
    self.dec2_b = nn.Sequential(*dec2_b)
    dec3_b = [nn.ConvTranspose1d(in_channels=256, out_channels=208, kernel_size=3, stride=2, padding=1, output_padding = 1)]
    self.dec3_b = nn.Sequential(*dec3_b)
    dec4_b = [nn.ConvTranspose1d(in_channels=216, out_channels=216, kernel_size=3, stride=2, padding=1, output_padding = 1)]
    self.dec4_b = nn.Sequential(*dec4_b)

  def forward_a(self, z_c, z_a):
    out0 = self.fc_a(z_c)
    out0 = out0.view(out0.size(0), out0.size(1), 1)   # Nx1024x1
    out1 = self.dec0_a(out0)   # Nx1016x13
    z_a_1 = z_a.view(z_a.size(0), z_a.size(1), 1).expand(z_a.size(0), z_a.size(1), out1.size(2))
    latent1 = torch.cat([out1, z_a_1],1)
    out2 = self.dec1_a(latent1)    # Nx504x25
    z_a_2 = z_a.view(z_a.size(0), z_a.size(1), 1).expand(z_a.size(0), z_a.size(1), out2.size(2))
    latent2 = torch.cat([out2, z_a_2],1)  
    out3 = self.dec2_a(latent2)    # Nx248x50
    z_a_3 = z_a.view(z_a.size(0), z_a.size(1), 1).expand(z_a.size(0), z_a.size(1), out3.size(2))
    latent3 = torch.cat([out3, z_a_3],1)
    out4 = self.dec3_a(latent3)    # Nx208x100
    z_a_4 = z_a.view(z_a.size(0), z_a.size(1), 1).expand(z_a.size(0), z_a.size(1), out4.size(2))
    latent4 = torch.cat([out4, z_a_4],1)  
    out5 = self.dec4_a(latent4)    # Nx216x200
    return out5

  def forward_b(self, z_c, z_a):
    out0 = self.fc_b(z_c)
    out0 = out0.view(out0.size(0), out0.size(1), 1)   # Nx1024x1
    out1 = self.dec0_b(out0)   # Nx1016x13
    z_a_1 = z_a.view(z_a.size(0), z_a.size(1), 1).expand(z_a.size(0), z_a.size(1), out1.size(2))
    latent1 = torch.cat([out1, z_a_1],1)
    out2 = self.dec1_b(latent1)    # Nx504x25
    z_a_2 = z_a.view(z_a.size(0), z_a.size(1), 1).expand(z_a.size(0), z_a.size(1), out2.size(2))
    latent2 = torch.cat([out2, z_a_2],1)  
    out3 = self.dec2_b(latent2)    # Nx248x50
    z_a_3 = z_a.view(z_a.size(0), z_a.size(1), 1).expand(z_a.size(0), z_a.size(1), out3.size(2))
    latent3 = torch.cat([out3, z_a_3],1)
    out4 = self.dec3_b(latent3)    # Nx208x100
    z_a_4 = z_a.view(z_a.size(0), z_a.size(1), 1).expand(z_a.size(0), z_a.size(1), out4.size(2))
    latent4 = torch.cat([out4, z_a_4],1)  
    out5 = self.dec4_b(latent4)    # Nx216x200
    return out5


####################################################################
#------------------------- Basic Functions -------------------------
####################################################################
def get_scheduler(optimizer, opts, cur_ep=-1):
  if opts.lr_policy == 'lambda':
    def lambda_rule(ep):
      lr_l = 1.0 - max(0, ep - opts.n_ep_decay) / float(opts.n_ep - opts.n_ep_decay + 1)
      return lr_l
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule, last_epoch=cur_ep)
  elif opts.lr_policy == 'step':
    scheduler = lr_scheduler.StepLR(optimizer, step_size=opts.n_ep_decay, gamma=0.1, last_epoch=cur_ep)
  else:
    return NotImplementedError('no such learn rate policy')
  return scheduler

def meanpoolConv(inplanes, outplanes):
  sequence = []
  sequence += [nn.AvgPool2d(kernel_size=2, stride=2)]
  sequence += [nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=1, padding=0, bias=True)]
  return nn.Sequential(*sequence)

def convMeanpool(inplanes, outplanes):
  sequence = []
  sequence += conv3x3(inplanes, outplanes)
  sequence += [nn.AvgPool2d(kernel_size=2, stride=2)]
  return nn.Sequential(*sequence)

def get_norm_layer(layer_type='instance'):
  if layer_type == 'batch':
    norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
  elif layer_type == 'instance':
    norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
  elif layer_type == 'none':
    norm_layer = None
  else:
    raise NotImplementedError('normalization layer [%s] is not found' % layer_type)
  return norm_layer

def get_non_linearity(layer_type='relu'):
  if layer_type == 'relu':
    nl_layer = functools.partial(nn.ReLU, inplace=True)
  elif layer_type == 'lrelu':
    nl_layer = functools.partial(nn.LeakyReLU, negative_slope=0.2, inplace=False)
  elif layer_type == 'elu':
    nl_layer = functools.partial(nn.ELU, inplace=True)
  else:
    raise NotImplementedError('nonlinearity activitation [%s] is not found' % layer_type)
  return nl_layer
def conv3x3(in_planes, out_planes):
  return [nn.ReflectionPad2d(1), nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=0, bias=True)]

def gaussian_weights_init(m):
  classname = m.__class__.__name__
  if classname.find('Conv') != -1 and classname.find('Conv') == 0:
    m.weight.data.normal_(0.0, 0.02)

####################################################################
#-------------------------- Basic Blocks --------------------------
####################################################################

## The code of LayerNorm is modified from MUNIT (https://github.com/NVlabs/MUNIT)
class LayerNorm(nn.Module):
  def __init__(self, n_out, eps=1e-5, affine=True):
    super(LayerNorm, self).__init__()
    self.n_out = n_out
    self.affine = affine
    if self.affine:
      self.weight = nn.Parameter(torch.ones(n_out, 1, 1))
      self.bias = nn.Parameter(torch.zeros(n_out, 1, 1))
    return
  def forward(self, x):
    normalized_shape = x.size()[1:]
    if self.affine:
      return F.layer_norm(x, normalized_shape, self.weight.expand(normalized_shape), self.bias.expand(normalized_shape))
    else:
      return F.layer_norm(x, normalized_shape)

class BasicBlock(nn.Module):
  def __init__(self, inplanes, outplanes, norm_layer=None, nl_layer=None):
    super(BasicBlock, self).__init__()
    layers = []
    if norm_layer is not None:
      layers += [norm_layer(inplanes)]
    layers += [nl_layer()]
    layers += conv3x3(inplanes, inplanes)
    if norm_layer is not None:
      layers += [norm_layer(inplanes)]
    layers += [nl_layer()]
    layers += [convMeanpool(inplanes, outplanes)]
    self.conv = nn.Sequential(*layers)
    self.shortcut = meanpoolConv(inplanes, outplanes)
  def forward(self, x):
    out = self.conv(x) + self.shortcut(x)
    return out

class LeakyReLUConv2d(nn.Module):
  def __init__(self, n_in, n_out, kernel_size, stride, padding=0, norm='None', sn=False):
    super(LeakyReLUConv2d, self).__init__()
    model = []
    model += [nn.ReflectionPad2d(padding)]
    if sn:
      model += [spectral_norm(nn.Conv2d(n_in, n_out, kernel_size=kernel_size, stride=stride, padding=0, bias=True))]
    else:
      model += [nn.Conv2d(n_in, n_out, kernel_size=kernel_size, stride=stride, padding=0, bias=True)]
    if 'norm' == 'Instance':
      model += [nn.InstanceNorm2d(n_out, affine=False)]
    model += [nn.LeakyReLU(inplace=True)]
    self.model = nn.Sequential(*model)
    self.model.apply(gaussian_weights_init)
    #elif == 'Group'
  def forward(self, x):
    return self.model(x)

class ReLUINSConv2d(nn.Module):
  def __init__(self, n_in, n_out, kernel_size, stride, padding=0):
    super(ReLUINSConv2d, self).__init__()
    model = []
    model += [nn.ReflectionPad2d(padding)]
    model += [nn.Conv2d(n_in, n_out, kernel_size=kernel_size, stride=stride, padding=0, bias=True)]
    model += [nn.InstanceNorm2d(n_out, affine=False)]
    model += [nn.ReLU(inplace=True)]
    self.model = nn.Sequential(*model)
    self.model.apply(gaussian_weights_init)
  def forward(self, x):
    return self.model(x)

class INSResBlock(nn.Module):
  def conv3x3(self, inplanes, out_planes, stride=1):
    return [nn.ReflectionPad2d(1), nn.Conv2d(inplanes, out_planes, kernel_size=3, stride=stride)]
  def __init__(self, inplanes, planes, stride=1, dropout=0.0):
    super(INSResBlock, self).__init__()
    model = []
    model += self.conv3x3(inplanes, planes, stride)
    model += [nn.InstanceNorm2d(planes)]
    model += [nn.ReLU(inplace=True)]
    model += self.conv3x3(planes, planes)
    model += [nn.InstanceNorm2d(planes)]
    if dropout > 0:
      model += [nn.Dropout(p=dropout)]
    self.model = nn.Sequential(*model)
    self.model.apply(gaussian_weights_init)
  def forward(self, x):
    residual = x
    out = self.model(x)
    out += residual
    return out

class MisINSResBlock(nn.Module):
  def conv3x3(self, dim_in, dim_out, stride=1):
    return nn.Sequential(nn.ReflectionPad2d(1), nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=stride))
  def conv1x1(self, dim_in, dim_out):
    return nn.Conv2d(dim_in, dim_out, kernel_size=1, stride=1, padding=0)
  def __init__(self, dim, dim_extra, stride=1, dropout=0.0):
    super(MisINSResBlock, self).__init__()
    self.conv1 = nn.Sequential(
        self.conv3x3(dim, dim, stride),
        nn.InstanceNorm2d(dim))
    self.conv2 = nn.Sequential(
        self.conv3x3(dim, dim, stride),
        nn.InstanceNorm2d(dim))
    self.blk1 = nn.Sequential(
        self.conv1x1(dim + dim_extra, dim + dim_extra),
        nn.ReLU(inplace=False),
        self.conv1x1(dim + dim_extra, dim),
        nn.ReLU(inplace=False))
    self.blk2 = nn.Sequential(
        self.conv1x1(dim + dim_extra, dim + dim_extra),
        nn.ReLU(inplace=False),
        self.conv1x1(dim + dim_extra, dim),
        nn.ReLU(inplace=False))
    model = []
    if dropout > 0:
      model += [nn.Dropout(p=dropout)]
    self.model = nn.Sequential(*model)
    self.model.apply(gaussian_weights_init)
    self.conv1.apply(gaussian_weights_init)
    self.conv2.apply(gaussian_weights_init)
    self.blk1.apply(gaussian_weights_init)
    self.blk2.apply(gaussian_weights_init)
  def forward(self, x, z):
    residual = x
    z_expand = z.view(z.size(0), z.size(1), 1, 1).expand(z.size(0), z.size(1), x.size(2), x.size(3))
    o1 = self.conv1(x)
    o2 = self.blk1(torch.cat([o1, z_expand], dim=1))
    o3 = self.conv2(o2)
    out = self.blk2(torch.cat([o3, z_expand], dim=1))
    out += residual
    return out

class GaussianNoiseLayer(nn.Module):
  def __init__(self,):
    super(GaussianNoiseLayer, self).__init__()
  def forward(self, x):
    if self.training == False:
      return x
    noise = Variable(torch.randn(x.size()).cuda(x.get_device()))
    return x + noise

class ReLUINSConvTranspose2d(nn.Module):
  def __init__(self, n_in, n_out, kernel_size, stride, padding, output_padding):
    super(ReLUINSConvTranspose2d, self).__init__()
    model = []
    model += [nn.ConvTranspose2d(n_in, n_out, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding, bias=True)]
    model += [LayerNorm(n_out)]
    model += [nn.ReLU(inplace=True)]
    self.model = nn.Sequential(*model)
    self.model.apply(gaussian_weights_init)
  def forward(self, x):
    return self.model(x)


class ResBlock_start(nn.Module):
  def __init__(self, channel_in, channel_out):
    super(ResBlock_start, self).__init__()
    model = [nn.Conv1d(channel_in, channel_out, kernel_size = 3, stride=1, padding=1)]
    model += [nn.LeakyReLU(negative_slope=0.1, inplace=False)]
    model += [nn.Conv1d(channel_out, channel_out, kernel_size = 3, stride=1, padding=1)]
    model += [nn.AvgPool1d(kernel_size = 3, stride=2, padding=1)]
    self.model = nn.Sequential(*model)
    res = [nn.AvgPool1d(kernel_size = 3, stride=2, padding=1)]
    res += [nn.Conv1d(channel_in, channel_out, kernel_size = 3, stride=1, padding=1)]
    self.res = nn.Sequential(*res)

  def forward(self, x):
    return self.model(x) + self.res(x)


class ResBlock(nn.Module):
  def __init__(self, channel_in, channel_out, stride_1 = 1, padding_1 = 1, stride_2 = 1, padding_2 = 1, stride_3 = 1, padding_3 = 0):
    super(ResBlock, self).__init__()
    model = [nn.LeakyReLU(negative_slope=0.1, inplace=False)]
    model += [nn.Conv1d(channel_in, channel_out, kernel_size = 3, stride = stride_1, padding = padding_1)]
    model += [nn.LeakyReLU(negative_slope=0.1, inplace=False)]
    model += [nn.Conv1d(channel_out, channel_out, kernel_size = 3, stride = stride_2, padding = padding_2)]
    self.model = nn.Sequential(*model)

    res = [nn.Conv1d(channel_in, channel_out, kernel_size = 1, stride = stride_3, padding = padding_3)]
    self.res = nn.Sequential(*res)

  def forward(self, x):
    return self.model(x) + self.res(x)



####################################################################
#--------------------- Spectral Normalization ---------------------
#  This part of code is copied from pytorch master branch (0.5.0)
####################################################################
class SpectralNorm(object):
  def __init__(self, name='weight', n_power_iterations=1, dim=0, eps=1e-12):
    self.name = name
    self.dim = dim
    if n_power_iterations <= 0:
      raise ValueError('Expected n_power_iterations to be positive, but '
                       'got n_power_iterations={}'.format(n_power_iterations))
    self.n_power_iterations = n_power_iterations
    self.eps = eps
  def compute_weight(self, module):
    weight = getattr(module, self.name + '_orig')
    u = getattr(module, self.name + '_u')
    weight_mat = weight
    if self.dim != 0:
      # permute dim to front
      weight_mat = weight_mat.permute(self.dim,
                                            *[d for d in range(weight_mat.dim()) if d != self.dim])
    height = weight_mat.size(0)
    weight_mat = weight_mat.reshape(height, -1)
    with torch.no_grad():
      for _ in range(self.n_power_iterations):
        v = F.normalize(torch.matmul(weight_mat.t(), u), dim=0, eps=self.eps)
        u = F.normalize(torch.matmul(weight_mat, v), dim=0, eps=self.eps)
    sigma = torch.dot(u, torch.matmul(weight_mat, v))
    weight = weight / sigma
    return weight, u
  def remove(self, module):
    weight = getattr(module, self.name)
    delattr(module, self.name)
    delattr(module, self.name + '_u')
    delattr(module, self.name + '_orig')
    module.register_parameter(self.name, torch.nn.Parameter(weight))
  def __call__(self, module, inputs):
    if module.training:
      weight, u = self.compute_weight(module)
      setattr(module, self.name, weight)
      setattr(module, self.name + '_u', u)
    else:
      r_g = getattr(module, self.name + '_orig').requires_grad
      getattr(module, self.name).detach_().requires_grad_(r_g)

  @staticmethod
  def apply(module, name, n_power_iterations, dim, eps):
    fn = SpectralNorm(name, n_power_iterations, dim, eps)
    weight = module._parameters[name]
    height = weight.size(dim)
    u = F.normalize(weight.new_empty(height).normal_(0, 1), dim=0, eps=fn.eps)
    delattr(module, fn.name)
    module.register_parameter(fn.name + "_orig", weight)
    module.register_buffer(fn.name, weight.data)
    module.register_buffer(fn.name + "_u", u)
    module.register_forward_pre_hook(fn)
    return fn

def spectral_norm(module, name='weight', n_power_iterations=1, eps=1e-12, dim=None):
  if dim is None:
    if isinstance(module, (torch.nn.ConvTranspose1d,
                           torch.nn.ConvTranspose2d,
                           torch.nn.ConvTranspose3d)):
      dim = 1
    else:
      dim = 0
  SpectralNorm.apply(module, name, n_power_iterations, dim, eps)
  return module

def remove_spectral_norm(module, name='weight'):
  for k, hook in module._forward_pre_hooks.items():
    if isinstance(hook, SpectralNorm) and hook.name == name:
      hook.remove(module)
      del module._forward_pre_hooks[k]
      return module
  raise ValueError("spectral_norm of '{}' not found in {}".format(name, module))

