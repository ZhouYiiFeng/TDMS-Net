## torch lib
import torch
import torch.nn as nn
import util.util as util
import os
import cv2
import torch.nn.init as init
from networks.flownet2_pytorch.networks.resample2d_package.resample2d import Resample2d

class TDMSNet(nn.Module):

    def __init__(self, opts, nc_in, nc_out):
        super(TDMSNet, self).__init__()

        self.blocks = opts.blocks
        self.blocks_f = opts.blocks_f
        self.blocks_w = opts.blocks_w
        self.epoch = 0
        nf = opts.nf
        use_bias = (opts.norm == "IN")
        self.opts = opts

        self.flow_warping = Resample2d().to(opts.device)

        ## convolution layers
        self.conv1a = ConvLayer(3 + 3, nf * 1, kernel_size=7, stride=1, bias=use_bias, norm=opts.norm)  ## input: P_t, O_t-1
        self.conv1b = ConvLayer(3 + 3, nf * 1, kernel_size=7, stride=1, bias=use_bias, norm=opts.norm)  ## input: I_t, I_t-1
        self.conv2a = ConvLayer(nf * 1, nf * 2, kernel_size=3, stride=2, bias=use_bias, norm=opts.norm)
        self.conv2b = ConvLayer(nf * 1, nf * 2, kernel_size=3, stride=2, bias=use_bias, norm=opts.norm)
        self.conv3 = ConvLayer(nf * 4, nf * 4, kernel_size=3, stride=2, bias=use_bias, norm=opts.norm)

        # Residual blocks
        self.ResBlocks = nn.ModuleList()
        for b in range(self.blocks):
            self.ResBlocks.append(ResidualBlock(nf * 4, bias=use_bias, norm=opts.norm))
        self.ResBlocks_w = nn.ModuleList()
        for b in range(self.blocks_w):
            self.ResBlocks_w.append(ResidualBlock(nf * 4, bias=use_bias, norm=opts.norm))
        self.ResBlocks_f = nn.ModuleList()
        for b in range(self.blocks_f):
            self.ResBlocks_f.append(ResidualBlock(nf * 4, bias=use_bias, norm=opts.norm))


        # Upsampling Layers
        self.deconv1 = UpsampleConvLayer(nf * 4, nf * 2, kernel_size=3, stride=1, upsample=2, bias=use_bias,norm=opts.norm)
        self.deconv2 = UpsampleConvLayer(nf * 4, nf * 1, kernel_size=3, stride=1, upsample=2, bias=use_bias,norm=opts.norm)
        self.deconv3 = ConvLayer(nf * 2, nc_out, kernel_size=7, stride=1)
        ## output one channel mask
        self.deconv1_w = nn.ConvTranspose2d(nf * 4, nf * 2, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv2_w = nn.ConvTranspose2d(nf * 2, nf * 1, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv3_w = ConvLayer(nf * 1, nc_out, kernel_size=7, stride=1)
        # nn.ConvTranspose2d(ngf * 2, ngf, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv1_f = nn.ConvTranspose2d(nf * 4, nf * 2, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv2_f = nn.ConvTranspose2d(nf * 2, nf * 1, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv3_f = ConvLayer(nf * 1, 2, kernel_size=7, stride=1)

        # Non-linearities
        self.relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

        if opts.isTrain:
            ### criterion and loss recorder
            if opts.loss == 'L2':
                self.criterion = nn.MSELoss(size_average=True)
            elif opts.loss == 'L1':
                self.criterion = nn.L1Loss(size_average=True)
            else:
                raise Exception("Unsupported criterion %s" % opts.loss)

    def forward(self, X):
        self.input = X
        Xa = X[:, :6, :, :]  ## I_t, I_t-1
        Xb = X[:, 6:, :, :]  ## o_1 o_2
        E1a = self.relu(self.conv1a(Xa))
        E1b = self.relu(self.conv1b(Xb))
        E2a = self.relu(self.conv2a(E1a))
        E2b = self.relu(self.conv2b(E1b))
        E3 = self.relu(self.conv3(torch.cat((E2a, E2b), 1)))
        RB = E3
        for b in range(self.blocks):
            RB = self.ResBlocks[b](RB)
        for b in range(self.blocks_f):
            RB_f = self.ResBlocks_f[b](E3)
        for b in range(self.blocks_w):
            RB_w = self.ResBlocks_w[b](E3)
        D2_w = self.relu(self.deconv1_w(RB_w))
        D1_w = self.relu(self.deconv2_w(D2_w))
        Y_w = self.deconv3_w(D1_w)
        self.Y_w = self.sigmoid(Y_w)

        D2 = self.relu(self.deconv1(RB))
        C2 = torch.cat((D2, E2a), 1)
        D1 = self.relu(self.deconv2(C2))
        C1 = torch.cat((D1, E1a), 1)
        Y = self.deconv3(C1)
        self.Y = self.tanh(Y)

        D2_f = self.relu(self.deconv1_f(RB_f))
        D1_f = self.relu(self.deconv2_f(D2_f))
        self.Y_f = self.deconv3_f(D1_f) * 20


        # return self.computeLoss()

    def computeLoss(self, FlowNet =None, vgg=None):

        frame_i1 = self.input[:, :3, :, :]
        frame_i2 = self.input[:, 3:6, :, :]
        frame_o1 = self.input[:, 6:9, :, :]
        frame_p2 = self.input[:, 9:, :, :]


        # self.noc_mask_w = self.noc_mask * self.Y_w
        syn_Y = self.flow_warping(frame_o1, self.Y_f)
        self.frame_syn_o2 = syn_Y * self.Y_w + (1 - self.Y_w) * frame_p2 + self.Y

        if not self.opts.isTrain:
            return self.frame_syn_o2

        real_flow = FlowNet(frame_i2, frame_i1)
        warped_o1_realflow = self.flow_warping(frame_o1, real_flow)
        warp_i1 = self.flow_warping(frame_i1, real_flow)
        self.noc_mask = torch.exp(-self.opts.alpha * torch.sum(frame_i2 - warp_i1, dim=1).pow(2)).unsqueeze(1)

        real_flow12 = FlowNet(frame_i1, frame_i2)
        self.syn_flow12 = FlowNet(frame_o1, self.frame_syn_o2)
        flow_cyc_l = self.criterion(real_flow12, self.syn_flow12)

        f_loss = self.criterion(real_flow, self.Y_f)

        pl_loss = self.criterion(self.frame_syn_o2, frame_p2)

        t_loss = self.criterion(self.frame_syn_o2 * self.noc_mask, warped_o1_realflow * self.noc_mask)

        p_loss = self.computVGGLoss(self.frame_syn_o2, frame_p2, vgg)

        w_loss = self.criterion(self.noc_mask, self.Y_w)
        return self.frame_syn_o2, t_loss, p_loss, pl_loss, f_loss, flow_cyc_l, w_loss

    def saveInnerInfo(self, output_dir):

        # output_dir = os.path.join(self.opts.checkpoints_dir, self.opts.name, "epoch_%d" % self.epoch, iteration)
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
        output_filename_flow12 = os.path.join(output_dir, "%d_flow12.jpg" % self.epoch)
        output_filename_i1 = os.path.join(output_dir, "%d_i1.jpg" % self.epoch )
        output_filename_fs21 = os.path.join(output_dir, "%d_fs21.jpg" % self.epoch)
        output_filename_o1 = os.path.join(output_dir, "%d_o1.jpg" % self.epoch)
        output_filename_y = os.path.join(output_dir, "%d_y.jpg" % self.epoch)
        output_filename_p1 = os.path.join(output_dir, "%d_p1.jpg" % self.epoch)
        output_filename_res = os.path.join(output_dir, "%d_res.jpg" % self.epoch)
        output_filename_ms = os.path.join(output_dir, "%d_msk.jpg" % self.epoch)
        output_filename_w = os.path.join(output_dir, "%d_w.jpg" % self.epoch)

        ma_image = util.tensor2img(self.noc_mask)
        w_image = util.tensor2img(self.Y_w)
        i1_image = util.tensor2img(self.input[:, :3, :, :])
        o1_image = util.tensor2img(self.input[:, 6:9, :, :])
        p1_image = util.tensor2img(self.input[:, 9:, :, :])
        y_image = util.tensor2img(self.Y)
        res_image = util.tensor2img(self.frame_syn_o2)

        f12_image = util.tensor2img(self.syn_flow12)
        f12_image = util.flow_to_rgb(f12_image)
        f_image = util.tensor2img(self.Y_f)
        f_image = util.flow_to_rgb(f_image)

        util.save_img(f_image, output_filename_fs21)
        util.save_img(f12_image, output_filename_flow12)
        util.save_img(w_image, output_filename_w)
        util.save_img(ma_image, output_filename_ms)
        util.save_img(i1_image, output_filename_i1)
        util.save_img(o1_image, output_filename_o1)
        util.save_img(y_image, output_filename_y)
        util.save_img(p1_image, output_filename_p1)
        util.save_img(res_image, output_filename_res)

    def computVGGLoss(self,frame_syn_o2, frame_p2, VGG):
        if self.opts.w_VGG > 0:
            ### normalize
            frame_o2_n = self.normalize_ImageNet_stats(frame_syn_o2)
            frame_p2_n = self.normalize_ImageNet_stats(frame_p2)

            ### extract VGG features
            features_p2 = VGG(frame_p2_n, self.opts.VGGLayers[-1])
            features_o2 = VGG(frame_o2_n, self.opts.VGGLayers[-1])
            VGG_loss_all = []
            for l in self.opts.VGGLayers:
                VGG_loss_all.append(self.criterion(features_o2[l], features_p2[l]))
        return sum(VGG_loss_all)

    def normalize_ImageNet_stats(self, batch):
        mean = torch.zeros_like(batch)
        std = torch.zeros_like(batch)
        mean[:, 0, :, :] = 0.485
        mean[:, 1, :, :] = 0.456
        mean[:, 2, :, :] = 0.406
        std[:, 0, :, :] = 0.229
        std[:, 1, :, :] = 0.224
        std[:, 2, :, :] = 0.225
        batch_out = (batch - mean) / std
        return batch_out


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, norm=None, bias=True):
        super(ConvLayer, self).__init__()

        reflection_padding = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, bias=bias)

        self.norm = norm
        if norm == "BN":
            self.norm_layer = nn.BatchNorm2d(out_channels)
        elif norm == "IN":
            self.norm_layer = nn.InstanceNorm2d(out_channels, track_running_stats=True)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)

        if self.norm in ["BN" or "IN"]:
            out = self.norm_layer(out)

        return out


class UpsampleConvLayer(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None, norm=None, bias=True):
        super(UpsampleConvLayer, self).__init__()

        self.upsample = upsample
        if upsample:
            self.upsample_layer = nn.Upsample(scale_factor=upsample, mode='nearest')

        reflection_padding = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, bias=bias)

        self.norm = norm
        if norm == "BN":
            self.norm_layer = nn.BatchNorm2d(out_channels)
        elif norm == "IN":
            self.norm_layer = nn.InstanceNorm2d(out_channels, track_running_stats=True)

    def forward(self, x):

        x_in = x
        if self.upsample:
            x_in = self.upsample_layer(x_in)
        out = self.reflection_pad(x_in)
        out = self.conv2d(out)

        if self.norm in ["BN" or "IN"]:
            out = self.norm_layer(out)

        return out


class ResidualBlock(nn.Module):

    def __init__(self, channels, norm=None, bias=True):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1, bias=bias, norm=norm)
        self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1, bias=bias, norm=norm)

        self.relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        input = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        out = out + input

        return out


