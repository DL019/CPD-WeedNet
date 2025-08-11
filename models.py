import torch.nn as nn
from typing import Optional
import torch
import torch.nn.functional as F
from .transformer import Transformer
from einops import rearrange

__all__ = ['CSP_MUIB', 'FPA', 'DFS']


def make_divisible(
        value: float,
        divisor: int,
        min_value: Optional[float] = None,
        round_down_protect: bool = True,
) -> int:
    if min_value is None:
        min_value = divisor
    new_value = max(min_value, int(value + divisor / 2) // divisor * divisor)
    if round_down_protect and new_value < 0.9 * value:
        new_value += divisor
    return int(new_value)


def conv_2d(inp, oup, kernel_size=3, stride=1, groups=1, bias=False, norm=True, act=True):
    conv = nn.Sequential()
    padding = (kernel_size - 1) // 2
    conv.add_module('conv', nn.Conv2d(inp, oup, kernel_size, stride, padding, bias=bias, groups=groups))
    if norm:
        conv.add_module('BatchNorm2d', nn.BatchNorm2d(oup))
    if act:
        conv.add_module('Activation', nn.ReLU6())
    return conv


class UniversalInvertedBottleneckBlock(nn.Module):
    def __init__(self,
                 inp,
                 oup,
                 start_dw_kernel_size=3,
                 middle_dw_kernel_size=3,
                 middle_dw_downsample=1,
                 stride=1,
                 expand_ratio=1
                 ):
        super().__init__()
        self.start_dw_kernel_size = start_dw_kernel_size
        if self.start_dw_kernel_size:
            stride_ = stride if not middle_dw_downsample else 1
            self._start_dw_ = conv_2d(inp, inp, kernel_size=start_dw_kernel_size, stride=stride_, groups=inp, act=False)
        expand_filters = make_divisible(inp * expand_ratio, 8)
        self._expand_conv = conv_2d(inp, expand_filters, kernel_size=1)
        # Middle depthwise conv.
        self.middle_dw_kernel_size = middle_dw_kernel_size
        if self.middle_dw_kernel_size:
            stride_ = stride if middle_dw_downsample else 1
            self._middle_dw = conv_2d(expand_filters, expand_filters, kernel_size=middle_dw_kernel_size, stride=stride_,
                                      groups=expand_filters)
        self._proj_conv = conv_2d(expand_filters, oup, kernel_size=1, stride=1, act=False)

    def forward(self, x):
        if self.start_dw_kernel_size:
            x = self._start_dw_(x)
        x = self._expand_conv(x)
        if self.middle_dw_kernel_size:
            x = self._middle_dw(x)
        x = self._proj_conv(x)
        return x


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    default_act = nn.SiLU()

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True, dw=False):
        super().__init__()
        self.dw = dw
        if self.dw and k > 1 and c1 == c2:
            self.m_dw = nn.Conv2d(c1, c1, k, s, autopad(k, p, d), groups=c1, dilation=d, bias=False)
            self.bn_dw = nn.BatchNorm2d(c1)
            self.act_dw = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
            self.m_pw = nn.Conv2d(c1, c2, 1, 1, 0, groups=g, bias=False)  # Pointwise
            self.conv = None  # Not used directly if dw
        elif self.dw and k > 1:  # General DW + PW
            self.m_dw = nn.Conv2d(c1, c1, k, s, autopad(k, p, d), groups=c1, dilation=d, bias=False)
            self.bn_dw = nn.BatchNorm2d(c1)
            self.act_dw = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
            self.m_pw = nn.Conv2d(c1, c2, 1, 1, 0, groups=g, bias=False)  # Pointwise
            self.conv = None
        else:
            self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)

        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        if self.dw and hasattr(self, 'm_dw'):
            x = self.act_dw(self.bn_dw(self.m_dw(x)))
            x = self.act(self.bn(self.m_pw(x)))
            return x
        else:
            return self.act(self.bn(self.conv(x)))


class uibres(nn.Module):
    def __init__(self, c1, c2, shutcat=True):
        super().__init__()
        self.shutcat = shutcat
        self.uib1 = UniversalInvertedBottleneckBlock(c1, c1)
        self.uib2 = UniversalInvertedBottleneckBlock(c1, c1)
        # self.SiLU = nn.SiLU()

    def forward(self, x):
        identity = x
        x1 = self.uib1(x)
        x2 = self.uib2(x1)
        if self.shutcat:
            out = x2 + identity
        else:
            out = x2
        return out

class CSP_MUIB(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=False, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c + c1, c2, 1)
        self.m = nn.ModuleList(
            uibres(self.c, self.c, shortcut)
            for _ in range(n)
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        raw_x = x

        y = list(self.cv1(x).chunk(2, 1))
        y[0] = self.maxpool(y[0])

        for m in self.m:
            y.append(m(y[-1]))

        combined = torch.cat(y + [raw_x], dim=1)

        return self.cv2(combined)

class FPA(nn.Module):
    def __init__(self, c1, c2, e=0.5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        _c = int(c2 * e)
        self.down = nn.Sequential(
            nn.Conv2d(c1[0], _c, 1, 1),
            nn.Sigmoid(),
            nn.AvgPool2d(2,2),
        )
        self.c11 = nn.Conv2d(c1[1], _c, 1, 1)
        self.up = nn.Sequential(
            nn.Conv2d(c1[2], _c, 1, 1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
        )
        self.upup = nn.Sequential(
            nn.Conv2d(c1[3], _c, 1, 1),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True),
        )
        self.plus = nn.MaxPool2d(3, 1, 1)
        self.conv1 = Conv(c2, _c, 1, 1,act=False)

        self.branch1_conv1x3 = nn.Conv2d(c2, c2, kernel_size=(1, 3), stride=1, padding=(0, 1), bias=False)
        self.branch1_bn1 = nn.BatchNorm2d(c2)
        self.branch1_act1 = nn.SiLU()
        self.branch1_conv3x1 = nn.Conv2d(c2, c2, kernel_size=(3, 1), stride=1, padding=(1, 0), bias=False)
        self.branch1_bn2 = nn.BatchNorm2d(c2)
        self.branch1_act2 = nn.SiLU()

        self.branch2_conv1x5 = nn.Conv2d(c2, c2, kernel_size=(1, 5), stride=1, padding=(0, 2), bias=False)
        self.branch2_bn1 = nn.BatchNorm2d(c2)
        self.branch2_act1 = nn.SiLU()
        self.branch2_conv5x1 = nn.Conv2d(c2, c2, kernel_size=(5, 1), stride=1, padding=(2, 0), bias=False)
        self.branch2_bn2 = nn.BatchNorm2d(c2)
        self.branch2_act2 = nn.SiLU()

        self.branches_fusion_conv = Conv(c2*2, c2, 1, 1)

    def forward(self, x):
        x_2, x_3, x_4, x_5 = x
        p4_up = self.up(x_4)
        p3_adj = self.c11(x_3)
        p2_down = self.down(x_2)
        p5_upup = self.upup(x_5)

        p234 = (p2_down * p3_adj) + p4_up
        fused = torch.cat((p234, p5_upup), dim=1)
        fused_after = self.conv1(fused)

        b1 = self.branch1_conv1x3(fused_after)
        b1 = self.branch1_bn1(b1)
        b1 = self.branch1_act1(b1)
        b1 = self.branch1_conv3x1(b1)
        b1 = self.branch1_bn2(b1)
        b1 = self.branch1_act2(b1)

        b2 = self.branch2_conv1x5(fused_after)
        b2 = self.branch2_bn1(b2)
        b2 = self.branch2_act1(b2)
        b2 = self.branch2_conv5x1(b2)
        b2 = self.branch2_bn2(b2)
        b2 = self.branch2_act2(b2)

        output = torch.cat((b1, b2), dim=1)
        output = output + fused
        return output

class DFS(nn.Module):
    def __init__(self, c1, c2, e=0.5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        _c = int(c2 * e)
        self.down = nn.Sequential(
            nn.Conv2d(c1[0], _c, 1, stride=1),  # 128-256 (下采样)
            nn.Sigmoid(),
            nn.AvgPool2d(2, 2),
        )
        self.c11 = Conv(c1[1], _c, 1)
        self.up = nn.Sequential(
            nn.Conv2d(c1[2], _c, 1),  # 512-256
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
        )
        self.ph = 2
        self.pw = 2
        self.branch1_conv1x3 = nn.Conv2d(_c, _c, kernel_size=(1, 3), stride=1, padding=(0, 1), bias=False)
        self.branch1_bn1 = nn.BatchNorm2d(_c)
        self.branch1_act1 = nn.SiLU()
        self.branch1_conv3x1 = nn.Conv2d(_c, _c, kernel_size=(3, 1), stride=1, padding=(1, 0), bias=False)
        self.branch1_bn2 = nn.BatchNorm2d(_c)
        self.branch1_act2 = nn.SiLU()

        self.branch2_conv1x5 = nn.Conv2d(_c, _c, kernel_size=(1, 5), stride=1, padding=(0, 2), bias=False)
        self.branch2_bn1 = nn.BatchNorm2d(_c)
        self.branch2_act1 = nn.SiLU()
        self.branch2_conv5x1 = nn.Conv2d(_c, _c, kernel_size=(5, 1), stride=1, padding=(2, 0), bias=False)
        self.branch2_bn2 = nn.BatchNorm2d(_c)
        self.branch2_act2 = nn.SiLU()

        self.branches_fusion_conv = Conv(c2 * 2, c2, 1, 1)
        self.plus = nn.MaxPool2d(3,1,1)
        self.trans = Transformer(_c, 4,4, 8, 240)
        self.conv3 = Conv(c2, _c,1, 1)
        self.conv4= Conv(_c, c2, 1, 1, act=False)

    def forward(self, x):
        x_3, x_4, x_5 = x
        _, _, h, w = x_4.shape
        p5_up = self.up(x_5)
        p4_adj = self.c11(x_4)
        p3_down = self.down(x_3)
        fused = (p3_down * p4_adj) + p5_up

        b1 = self.branch1_conv1x3(fused)
        b1 = self.branch1_bn1(b1)
        b1 = self.branch1_act1(b1)
        b1 = self.branch1_conv3x1(b1)
        b1 = self.branch1_bn2(b1)
        b1 = self.branch1_act2(b1)

        b2 = self.branch2_conv1x5(fused)
        b2 = self.branch2_bn1(b2)
        b2 = self.branch2_act1(b2)
        b2 = self.branch2_conv5x1(b2)
        b2 = self.branch2_bn2(b2)
        b2 = self.branch2_act2(b2)

        output = torch.cat((b1, b2),dim=1)
        output = self.conv3(output)
        x = rearrange(output, 'b d (h ph) (w pw) -> b (ph pw) (h w) d', ph=self.ph, pw=self.pw)
        tra = self.trans(x)
        x = rearrange(tra, 'b (ph pw) (h w) d -> b d (h ph) (w pw)', h=h // self.ph, w=w // self.pw, ph=self.ph,
                      pw=self.pw)
        y = self.conv4(x + fused)
        return y

class Add(nn.Module):
    def __init__(self, c1):
        super().__init__()

    def forward(self, x):
        if isinstance(x, list):
            y = x[0] + x[1]
        return y



