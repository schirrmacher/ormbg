import torch
import torch.nn as nn
import torch.nn.functional as F

# https://github.com/xuebinqin/DIS/blob/main/IS-Net/models/isnet.py


class ReluBatchNormConv(nn.Module):
    def __init__(self, in_channels=4, out_channels=4, dirate=1, stride=1):
        super(ReluBatchNormConv, self).__init__()

        self.conv_s1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            padding=1 * dirate,
            dilation=1 * dirate,
        )
        self.bn_s1 = nn.BatchNorm2d(out_channels)
        self.relu_s1 = nn.ReLU(inplace=True)

    def forward(self, x):

        hx = x
        xout = self.relu_s1(self.bn_s1(self.conv_s1(hx)))

        return xout


bce_loss = nn.BCELoss(size_average=True)


def muti_loss_fusion(preds, target):
    loss0 = 0.0
    loss = 0.0

    for i in range(0, len(preds)):
        if preds[i].shape[2] != target.shape[2] or preds[i].shape[3] != target.shape[3]:
            tmp_target = F.interpolate(
                target, size=preds[i].size()[2:], mode="bilinear", align_corners=True
            )
            loss = loss + bce_loss(preds[i], tmp_target)
        else:
            loss = loss + bce_loss(preds[i], target)
        if i == 0:
            loss0 = loss
    return loss0, loss


def _upsample_like(src, tar):

    src = F.interpolate(src, size=tar.shape[2:], mode="bilinear")

    return src


class RSU7(nn.Module):

    def __init__(self, in_channels=4, mid_ch=12, out_channels=4, img_size=512):
        super(RSU7, self).__init__()

        self.in_ch = in_channels
        self.mid_ch = mid_ch
        self.out_ch = out_channels

        self.rebnconvin = ReluBatchNormConv(in_channels, out_channels, dirate=1)

        self.rebnconv1 = ReluBatchNormConv(out_channels, mid_ch, dirate=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)

        self.rebnconv2 = ReluBatchNormConv(mid_ch, mid_ch, dirate=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)

        self.rebnconv3 = ReluBatchNormConv(mid_ch, mid_ch, dirate=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)

        self.rebnconv4 = ReluBatchNormConv(mid_ch, mid_ch, dirate=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)

        self.rebnconv5 = ReluBatchNormConv(mid_ch, mid_ch, dirate=1)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)

        self.rebnconv6 = ReluBatchNormConv(mid_ch, mid_ch, dirate=1)

        self.rebnconv7 = ReluBatchNormConv(mid_ch, mid_ch, dirate=2)

        self.rebnconv6d = ReluBatchNormConv(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv5d = ReluBatchNormConv(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv4d = ReluBatchNormConv(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv3d = ReluBatchNormConv(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv2d = ReluBatchNormConv(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv1d = ReluBatchNormConv(mid_ch * 2, out_channels, dirate=1)

    def forward(self, x):

        hx = x
        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)

        hx4 = self.rebnconv4(hx)
        hx = self.pool4(hx4)

        hx5 = self.rebnconv5(hx)
        hx = self.pool5(hx5)

        hx6 = self.rebnconv6(hx)

        hx7 = self.rebnconv7(hx6)

        hx6d = self.rebnconv6d(torch.cat((hx7, hx6), 1))
        hx6dup = _upsample_like(hx6d, hx5)

        hx5d = self.rebnconv5d(torch.cat((hx6dup, hx5), 1))
        hx5dup = _upsample_like(hx5d, hx4)

        hx4d = self.rebnconv4d(torch.cat((hx5dup, hx4), 1))
        hx4dup = _upsample_like(hx4d, hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))

        return hx1d + hxin


class RSU6(nn.Module):

    def __init__(self, in_channels=4, mid_ch=12, out_channels=4):
        super(RSU6, self).__init__()

        self.rebnconvin = ReluBatchNormConv(in_channels, out_channels, dirate=1)

        self.rebnconv1 = ReluBatchNormConv(out_channels, mid_ch, dirate=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)

        self.rebnconv2 = ReluBatchNormConv(mid_ch, mid_ch, dirate=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)

        self.rebnconv3 = ReluBatchNormConv(mid_ch, mid_ch, dirate=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)

        self.rebnconv4 = ReluBatchNormConv(mid_ch, mid_ch, dirate=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)

        self.rebnconv5 = ReluBatchNormConv(mid_ch, mid_ch, dirate=1)
        self.rebnconv6 = ReluBatchNormConv(mid_ch, mid_ch, dirate=2)

        self.rebnconv5d = ReluBatchNormConv(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv4d = ReluBatchNormConv(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv3d = ReluBatchNormConv(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv2d = ReluBatchNormConv(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv1d = ReluBatchNormConv(mid_ch * 2, out_channels, dirate=1)

    def forward(self, x):

        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)

        hx4 = self.rebnconv4(hx)
        hx = self.pool4(hx4)

        hx5 = self.rebnconv5(hx)

        hx6 = self.rebnconv6(hx5)

        hx5d = self.rebnconv5d(torch.cat((hx6, hx5), 1))
        hx5dup = _upsample_like(hx5d, hx4)

        hx4d = self.rebnconv4d(torch.cat((hx5dup, hx4), 1))
        hx4dup = _upsample_like(hx4d, hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))

        return hx1d + hxin


class RSU5(nn.Module):

    def __init__(self, in_channels=4, mid_ch=12, out_channels=4):
        super(RSU5, self).__init__()

        self.rebnconvin = ReluBatchNormConv(in_channels, out_channels, dirate=1)

        self.rebnconv1 = ReluBatchNormConv(out_channels, mid_ch, dirate=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)

        self.rebnconv2 = ReluBatchNormConv(mid_ch, mid_ch, dirate=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)

        self.rebnconv3 = ReluBatchNormConv(mid_ch, mid_ch, dirate=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)

        self.rebnconv4 = ReluBatchNormConv(mid_ch, mid_ch, dirate=1)

        self.rebnconv5 = ReluBatchNormConv(mid_ch, mid_ch, dirate=2)

        self.rebnconv4d = ReluBatchNormConv(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv3d = ReluBatchNormConv(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv2d = ReluBatchNormConv(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv1d = ReluBatchNormConv(mid_ch * 2, out_channels, dirate=1)

    def forward(self, x):

        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)

        hx4 = self.rebnconv4(hx)

        hx5 = self.rebnconv5(hx4)

        hx4d = self.rebnconv4d(torch.cat((hx5, hx4), 1))
        hx4dup = _upsample_like(hx4d, hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))

        return hx1d + hxin


class RSU4(nn.Module):

    def __init__(self, in_channels=4, mid_ch=12, out_channels=4):
        super(RSU4, self).__init__()

        self.rebnconvin = ReluBatchNormConv(in_channels, out_channels, dirate=1)

        self.rebnconv1 = ReluBatchNormConv(out_channels, mid_ch, dirate=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)

        self.rebnconv2 = ReluBatchNormConv(mid_ch, mid_ch, dirate=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)

        self.rebnconv3 = ReluBatchNormConv(mid_ch, mid_ch, dirate=1)

        self.rebnconv4 = ReluBatchNormConv(mid_ch, mid_ch, dirate=2)

        self.rebnconv3d = ReluBatchNormConv(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv2d = ReluBatchNormConv(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv1d = ReluBatchNormConv(mid_ch * 2, out_channels, dirate=1)

    def forward(self, x):

        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)

        hx4 = self.rebnconv4(hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))

        return hx1d + hxin


class RSU4F(nn.Module):

    def __init__(self, in_channels=4, mid_channels=12, out_channels=4):
        super(RSU4F, self).__init__()

        self.rebnconvin = ReluBatchNormConv(in_channels, out_channels, dirate=1)

        self.rebnconv1 = ReluBatchNormConv(out_channels, mid_channels, dirate=1)

        self.rebnconv2 = ReluBatchNormConv(mid_channels, mid_channels, dirate=2)
        self.rebnconv3 = ReluBatchNormConv(mid_channels, mid_channels, dirate=4)
        self.rebnconv4 = ReluBatchNormConv(mid_channels, mid_channels, dirate=8)

        self.rebnconv3d = ReluBatchNormConv(mid_channels * 2, mid_channels, dirate=4)
        self.rebnconv2d = ReluBatchNormConv(mid_channels * 2, mid_channels, dirate=2)
        self.rebnconv1d = ReluBatchNormConv(mid_channels * 2, out_channels, dirate=1)

    def forward(self, x):

        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx2 = self.rebnconv2(hx1)
        hx3 = self.rebnconv3(hx2)

        hx4 = self.rebnconv4(hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4, hx3), 1))
        hx2d = self.rebnconv2d(torch.cat((hx3d, hx2), 1))
        hx1d = self.rebnconv1d(torch.cat((hx2d, hx1), 1))

        return hx1d + hxin


class ORMBGTeacher(nn.Module):

    def __init__(self, in_channels=4, out_channels=1):
        super(ORMBGTeacher, self).__init__()

        self.conv_in = nn.Conv2d(
            in_channels=in_channels, out_channels=64, kernel_size=3, stride=2, padding=1
        )
        self.pool_in = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)

        self.stage1 = RSU7(64, 32, 64)
        self.pool12 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)

        self.stage2 = RSU6(64, 32, 128)
        self.pool23 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)

        self.stage3 = RSU5(128, 64, 256)
        self.pool34 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)

        self.stage4 = RSU4(256, 128, 512)
        self.pool45 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)

        self.stage5 = RSU4F(512, 256, 512)
        self.pool56 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)

        self.stage6 = RSU4F(512, 256, 512)

        # decoder
        self.stage5d = RSU4F(1024, 256, 512)
        self.stage4d = RSU4(1024, 128, 256)
        self.stage3d = RSU5(512, 64, 128)
        self.stage2d = RSU6(256, 32, 64)
        self.stage1d = RSU7(128, 16, 64)

        self.side1 = nn.Conv2d(
            in_channels=64, out_channels=out_channels, kernel_size=3, padding=1
        )
        self.side2 = nn.Conv2d(
            in_channels=64, out_channels=out_channels, kernel_size=3, padding=1
        )
        self.side3 = nn.Conv2d(
            in_channels=128, out_channels=out_channels, kernel_size=3, padding=1
        )
        self.side4 = nn.Conv2d(
            in_channels=256, out_channels=out_channels, kernel_size=3, padding=1
        )
        self.side5 = nn.Conv2d(
            in_channels=512, out_channels=out_channels, kernel_size=3, padding=1
        )
        self.side6 = nn.Conv2d(
            in_channels=512, out_channels=out_channels, kernel_size=3, padding=1
        )

    def compute_loss(self, preds, targets):
        return muti_loss_fusion(preds, targets)

    def forward(self, x):

        hx = x

        hxin = self.conv_in(hx)

        # stage 1
        hx1 = self.stage1(hxin)
        hx = self.pool12(hx1)

        # stage 2
        hx2 = self.stage2(hx)
        hx = self.pool23(hx2)

        # stage 3
        hx3 = self.stage3(hx)
        hx = self.pool34(hx3)

        # stage 4
        hx4 = self.stage4(hx)
        hx = self.pool45(hx4)

        # stage 5
        hx5 = self.stage5(hx)
        hx = self.pool56(hx5)

        # stage 6
        hx6 = self.stage6(hx)
        hx6up = _upsample_like(hx6, hx5)

        # -------------------- decoder --------------------
        hx5d = self.stage5d(torch.cat((hx6up, hx5), 1))
        hx5dup = _upsample_like(hx5d, hx4)

        hx4d = self.stage4d(torch.cat((hx5dup, hx4), 1))
        hx4dup = _upsample_like(hx4d, hx3)

        hx3d = self.stage3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)

        hx2d = self.stage2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)

        hx1d = self.stage1d(torch.cat((hx2dup, hx1), 1))

        # side output
        d1 = self.side1(hx1d)
        d1 = _upsample_like(d1, x)

        d2 = self.side2(hx2d)
        d2 = _upsample_like(d2, x)

        d3 = self.side3(hx3d)
        d3 = _upsample_like(d3, x)

        d4 = self.side4(hx4d)
        d4 = _upsample_like(d4, x)

        d5 = self.side5(hx5d)
        d5 = _upsample_like(d5, x)

        d6 = self.side6(hx6)
        d6 = _upsample_like(d6, x)

        return [
            F.sigmoid(d1),
            F.sigmoid(d2),
            F.sigmoid(d3),
            F.sigmoid(d4),
            F.sigmoid(d5),
            F.sigmoid(d6),
        ], [hx1d, hx2d, hx3d, hx4d, hx5d, hx6]
