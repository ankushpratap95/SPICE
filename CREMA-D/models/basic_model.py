import torch
import torch.nn as nn
import torch.nn.functional as F
from .backbone import resnet18, resnet50
from .fusion_modules import SumFusion, ConcatFusion, FiLM, GatedFusion, TwoLayerCrossAttention, DeepFusion, AdvancedAdapterModule
import numpy as np
import timm
import torchvision

class AVNet(nn.Module):
    def __init__(self, args):
        super(AVNet, self).__init__()

        n_classes = 6
        self.linear_a = nn.Linear(512, n_classes)
        self.linear_v = nn.Linear(512, n_classes)
        self.linear_last = nn.Linear(1024, n_classes)

        self.audio_net = resnet18(modality='audio')
        self.visual_net = resnet18(modality='visual')


    def forward(self, audio, visual):

        # (64,1,257,188) (64,3,1,224,224)
        a = self.audio_net(audio) # (64,512,9,6)
        a = F.adaptive_avg_pool2d(a, 1) # (64,512,1,1)
        a = torch.flatten(a, 1) # (64,512)

        v = self.visual_net(visual) # (64,512,7,7)
        v = F.adaptive_avg_pool2d(v, 1) # (64,512,1,1)
        v = torch.flatten(v, 1) # (64,512)

        out_m1 = self.linear_a(a)
        out_m2 = self.linear_v(v)
        out_mm = self.linear_last(torch.cat((a, v), dim=1))

        return out_mm, out_m1, out_m2, a, v
    
        
class AVNetFusion(nn.Module):
    def __init__(self, args):
        super().__init__()

        n_classes = 6

        self.audio_net = resnet18(modality='audio')
        self.visual_net = resnet18(modality='visual')

        # fusion backbone
        self.fusion_net = resnet18(modality='fusion')

        self.linear_a = nn.Linear(512, n_classes)
        self.linear_v = nn.Linear(512, n_classes)

        self.linear_last = nn.Linear(512, n_classes)

    def forward(self, audio, visual):

        # ----- unimodal encoders -----
        a = self.audio_net(audio)     # (B,512,9,6)
        v = self.visual_net(visual)   # (B,512,7,7)

        # resize to same spatial size
        # v = F.interpolate(v, size=a.shape[-2:], mode='bilinear', align_corners=False)

        # ----- unimodal heads -----
        a_pool = F.adaptive_avg_pool2d(a,1).flatten(1)
        v_pool = F.adaptive_avg_pool2d(v,1).flatten(1)

        out_m1 = self.linear_a(a_pool)
        out_m2 = self.linear_v(v_pool)

        # ----- multimodal fusion -----
        # resize to same spatial size
        v = F.interpolate(v, size=a.shape[-2:], mode='bilinear', align_corners=False)
        fusion = torch.cat((a, v), dim=1)   # (B,1024,9,6)

        f = self.fusion_net(fusion)

        f = F.adaptive_avg_pool2d(f,1).flatten(1)

        out_mm = self.linear_last(f)

        return out_mm, out_m1, out_m2, a_pool, v_pool        



class CLIP_2(nn.Module):
    def __init__(self, args):
        super(CLIP_2, self).__init__()

        n_classes = 6
        self.linear_a = nn.Linear(512, n_classes)
        self.linear_v = nn.Linear(192, n_classes)
        self.linear_last = nn.Linear(704, n_classes)

        self.audio_net = resnet18(modality='audio')
        self.visual_net = timm.create_model('deit_tiny_patch16_224', pretrained=False)
        self.visual_net.reset_classifier(0)

    def forward(self, audio, visual):

        # (64,1,257,188) (64,3,1,224,224)
        a = self.audio_net(audio) # (64,512,9,6)
        a = F.adaptive_avg_pool2d(a, 1) # (64,512,1,1)
        a = torch.flatten(a, 1) # (64,512)

        # v = self.visual_net(visual) # (64,512,7,7)
        # v = F.adaptive_avg_pool2d(v, 1) # (64,512,1,1)
        # v = torch.flatten(v, 1) # (64,512)
        B, T, C, H, W = visual.shape
        visual = visual.view(B, C* T, H, W)  # reshape
        v = self.visual_net(visual)

        out_m1 = self.linear_a(a)
        out_m2 = self.linear_v(v)
        out_mm = self.linear_last(torch.cat((a, v), dim=1))

        # return out_mm, a, v
        return out_mm, out_m1, out_m2, a, v
