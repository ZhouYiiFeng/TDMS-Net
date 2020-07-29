import torch
from .base_model import BaseModel
import os

class FlowNet(BaseModel):
    def name(self):
        return 'FlowNet'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        # flownet 2           
        from .flownet2_pytorch import models as flownet2_models
        from .flownet2_pytorch.utils import tools as flownet2_tools
        from .flownet2_pytorch.networks.resample2d_package.resample2d import Resample2d
        
        self.flowNet = flownet2_tools.module_to_dict(flownet2_models)['FlowNet2']().cuda(self.gpu_ids[0])
        print(os.path.abspath("."))
        checkpoint = torch.load('./pretrained_models/FlowNet2_checkpoint.pth.tar')
        self.flowNet.load_state_dict(checkpoint['state_dict'])
        self.freezeFlowNet()
        self.flowNet.eval()
        self.resample = Resample2d().to("cuda" if opt.cuda else "cpu")
        self.downsample = torch.nn.AvgPool2d((2, 2), stride=2, count_include_pad=False).to("cuda" if opt.cuda else "cpu")

    def forward(self, input_A, input_B):
        # input_A: real_B, input_B: real_B_prev
        with torch.no_grad():
            size = input_A.size()
            assert(len(size) == 4 or len(size) == 5)
            if len(size) == 5:
                b, n, c, h, w = size
                input_A = input_A.contiguous().view(-1, c, h, w)
                input_B = input_B.contiguous().view(-1, c, h, w)
                flow = self.compute_flow(input_A, input_B)
                return flow.view(b, n, 2, h, w)
            else:
                return self.compute_flow(input_A, input_B)

    def compute_flow(self, im1, im2):
        # in fact the flowNet compute the flow of im1->im2
        # the warp uses backward ﬂow to advect the previous frame toward the current frame,
        # which means use the **cur frame -> prev frame flow**.
        assert(im1.size()[1] == 3)
        assert(im1.size() == im2.size())        
        old_h, old_w = im1.size()[2], im1.size()[3]
        new_h, new_w = old_h//64*64, old_w//64*64
        if old_h != new_h:
            downsample = torch.nn.Upsample(size=(new_h, new_w), mode='bilinear')
            upsample = torch.nn.Upsample(size=(old_h, old_w), mode='bilinear')
            im1 = downsample(im1)
            im2 = downsample(im2)        
        data1 = torch.cat([im1.unsqueeze(2), im2.unsqueeze(2)], dim=2)            
        flow1 = self.flowNet(data1)
        # conf = (self.norm(im1 - self.resample(im2, flow1)) < 0.02).float() # flow is cur -> pre, so the warp(pre) -> warped cur
        if old_h != new_h:
            flow1 = upsample(flow1) * old_h / new_h
        return flow1.detach()

    def norm(self, t):
        return torch.sum(t*t, dim=1, keepdim=True)   

    def freezeFlowNet(self):
        for param in self.flowNet.parameters():
            param.requires_grad = False