# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle import ParamAttr
from paddle.nn import Conv2D
from paddle.regularizer import L2Decay
from ppdet.core.workspace import register, serializable
from paddle.nn.initializer import KaimingNormal

from ..shape_spec import ShapeSpec
from ..backbones.lcnet import DepthwiseSeparable
from .csp_pan import ConvBNLayer, Channel_T, DPModule

__all__ = ['LCPAN']

class Flatten(nn.Layer):
    def forward(self, x):
        return x.reshape([x.shape[0], -1])

class ConvBNLayer(nn.Layer):
    def __init__(
        self,
        num_channels,
        filter_size,
        num_filters,
        stride,
        num_groups=1,
        act="hard_swish",
    ):
        super().__init__()
        self.conv = nn.Conv2D(
            in_channels=num_channels,
            out_channels=num_filters,
            kernel_size=filter_size,
            stride=stride,
            padding=(filter_size - 1) // 2,
            groups=num_groups,
            weight_attr=ParamAttr(initializer=KaimingNormal()),
            bias_attr=False,
        )
        self.bn = nn.BatchNorm2D(
            num_filters,
            weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
            bias_attr=ParamAttr(regularizer=L2Decay(0.0)),
        )
        if act == "hard_swish":
            self.act = nn.Hardswish()
        elif act == "relu6":
            self.act = nn.ReLU6()
        else:
            self.act = nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x

class DoubleConv(nn.Layer):
    """(convolution => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2D(in_channels, mid_channels, kernel_size=3, stride=1, padding=1, bias_attr=False),
            nn.ReLU(),
            nn.Conv2D(mid_channels, out_channels, kernel_size=3, stride=1, padding=1, bias_attr=False),
            nn.ReLU()
        )

    def forward(self, x):
        return self.double_conv(x)

class Up_direct(nn.Layer):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, bilinear=False):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.Conv2DTranspose(in_channels, in_channels // 2, kernel_size=4, stride=2, padding=1, bias_attr=False)
            self.conv = DoubleConv(in_channels // 2, out_channels)

    def forward(self, x1):
        x = self.up(x1)
        x = self.conv(x)
        return x

class OutConv(nn.Layer):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2D(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.conv(x)

class RH(nn.Layer):
    """图像自重构头"""
    def __init__(self, in_channels=96, out_channels=3):
        super(RH, self).__init__()
        self.up1 = Up_direct(in_channels, 48)
        self.up2 = Up_direct(48, 24)
        self.out_conv = OutConv(24, out_channels)
    
    def forward(self, x):
        P0 = self.up1(x)
        P0 = self.up2(P0)
        r_img = self.out_conv(P0)
        return r_img

class DGFE(nn.Layer):
    """差异引导特征增强"""
    def __init__(self, gate_channels=256, reduction_ratio=16, pool_types=['avg', 'max']):
        super(DGFE, self).__init__()
        # 简化MLP结构，保持核心通道注意力功能
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
        )
        self.pool_types = pool_types
        # 新增可学习残差权重
        self.res_alpha = self.create_parameter(
            shape=[1],
            default_initializer=paddle.nn.initializer.Constant(0.7),
            attr=ParamAttr(regularizer=L2Decay(0.0001))
        )

    def forward(self, x, difference_map, learnable_thresh):
        # 差异图二值化
        difference_map_mask = (paddle.sign(difference_map - learnable_thresh) + 1) * 0.5
        # 上采样掩码到特征图尺寸
        feat_difference_map = F.interpolate(difference_map_mask, size=(x.shape[2], x.shape[3]), mode='nearest')
        channel_att_sum = 0
        for pool_type in self.pool_types:
            if pool_type == 'avg':
                pool_feat = F.avg_pool2d(x, kernel_size=x.shape[2:])
            else:  # max pool
                pool_feat = F.max_pool2d(x, kernel_size=x.shape[2:])
            channel_att_sum += self.mlp(pool_feat)
        scale = F.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
        # 生成空间注意力矩阵
        feat_diff_mat = feat_difference_map.repeat_interleave(x.shape[1], axis=1)
        # 特征增强+残差连接
        enhanced_feat = x * scale * feat_diff_mat
        final_feat = self.res_alpha * enhanced_feat + (1 - self.res_alpha) * x
        return final_feat

@register
@serializable
class LCPAN(nn.Layer):
    """融入SR-TOD的LCPAN"""
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=5,
                 num_features=3,
                 use_depthwise=True,
                 act='hard_swish',
                 spatial_scales=[0.25, 0.125, 0.0625, 0.03125],
                 srtod_in_channels=96,
                 srtod_out_channels=3):
        super(LCPAN, self).__init__()
        # 原LCPAN核心逻辑保持不变
        self.conv_t = Channel_T(in_channels, out_channels, act=act)
        self.in_channels = [out_channels] * len(spatial_scales)
        self.out_channels = out_channels
        self.spatial_scales = spatial_scales
        self.num_features = num_features
        self.final_fuse_convs = nn.LayerList()
        for i in range(len(in_channels)):
            self.final_fuse_convs.append(
                ConvBNLayer(
                    num_channels=out_channels * 2,
                    filter_size=1,
                    num_filters=out_channels,
                    stride=1,
                    act=act,
                )
            )
        conv_func = DPModule if use_depthwise else ConvBNLayer
        
        NET_CONFIG = {
            "block1": [
                [kernel_size, out_channels * 2, out_channels * 2, 1, False],
                [kernel_size, out_channels * 2, out_channels, 1, False],
            ],
            "block2": [
                [kernel_size, out_channels * 2, out_channels * 2, 1, False],
                [kernel_size, out_channels * 2, out_channels, 1, False],
            ]
        }

        if self.num_features == 4:
            self.first_top_conv = conv_func(
                self.in_channels[0], self.in_channels[0], kernel_size, stride=2, act=act)
            self.second_top_conv = conv_func(
                self.in_channels[0], self.in_channels[0], kernel_size, stride=2, act=act)

        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.top_down_blocks = nn.LayerList()
        for idx in range(len(self.in_channels) - 1, 0, -1):
            self.top_down_blocks.append(
                nn.Sequential(* [
                    DepthwiseSeparable(
                        num_channels=in_c,
                        num_filters=out_c,
                        dw_size=k,
                        stride=s,
                        use_se=se)
                    for i, (k, in_c, out_c, s, se) in enumerate(NET_CONFIG["block1"])
                ]))

        self.downsamples = nn.LayerList()
        self.bottom_up_blocks = nn.LayerList()
        for idx in range(len(self.in_channels) - 1):
            self.downsamples.append(
                conv_func(
                    self.in_channels[idx],
                    self.in_channels[idx],
                    kernel_size=kernel_size,
                    stride=2,
                    act=act))
            self.bottom_up_blocks.append(
                nn.Sequential(* [
                    DepthwiseSeparable(
                        num_channels=in_c,
                        num_filters=out_c,
                        dw_size=k,
                        stride=s,
                        use_se=se)
                    for i, (k, in_c, out_c, s, se) in enumerate(NET_CONFIG["block2"])
                ]))
        
        # SR-TOD核心模块
        self.rh = RH(in_channels=srtod_in_channels, out_channels=srtod_out_channels)
        self.dgfe = DGFE(gate_channels=self.in_channels[0])
        self.learnable_thresh = self.create_parameter(
            shape=[1],
            default_initializer=paddle.nn.initializer.Constant(0.0156862),
            attr=ParamAttr(trainable=True)  # 允许训练时更新
        )

    def forward(self, inputs, img_inputs=None):
        """
        Args:
            inputs (tuple[Tensor]): Backbone输出的多尺度特征
            img_inputs (Tensor, optional): 原始输入图像（训练/推理时传入）
        Returns:
            tuple[Tensor]: 增强后的LCPAN特征
        """
        assert len(inputs) == len(self.in_channels)
        inputs = self.conv_t(inputs)

        # 原LCPAN特征融合逻辑保持不变
        inner_outs = [inputs[-1]]
        for idx in range(len(self.in_channels) - 1, 0, -1):
            feat_heigh = inner_outs[0]
            feat_low = inputs[idx - 1]
            upsample_feat = self.upsample(feat_heigh)
            inner_out = self.top_down_blocks[len(self.in_channels) - 1 - idx](
                paddle.concat([upsample_feat, feat_low], 1))
            inner_outs.insert(0, inner_out)
        inner_outs.pop()

        outs = [inner_outs[0]]
        for idx in range(len(self.in_channels) - 2):
            feat_low = outs[-1]
            feat_height = inner_outs[idx + 1]
            downsample_feat = self.downsamples[idx](feat_low)
            out = self.bottom_up_blocks[idx](paddle.concat(
                [downsample_feat, feat_height], 1))
            outs.append(out)

        # SR-TOD特征增强
        if img_inputs is not None and self.training:
            shallow_feat = inputs[0]
            r_img = self.rh(shallow_feat)
            # 差异图计算
            difference_map = paddle.abs(r_img - img_inputs).mean(axis=1, keepdim=True)
            # 用改进后的DGFE增强特征
            enhanced_shallow_feat = self.dgfe(shallow_feat, difference_map, self.learnable_thresh)
            # 替换浅层特征
            inputs = list(inputs)
            inputs[0] = enhanced_shallow_feat
            inputs = tuple(inputs)

        final_outs = []
        for i, out_feat in enumerate(outs):
            origin_feat = inputs[i]  # Channel_T 之后的原始特征

            concat_feat = paddle.concat([out_feat, origin_feat], axis=1)
            fused_feat = self.final_fuse_convs[i](concat_feat)
            final_outs.append(fused_feat)

            # print(
            #     f"第{i}层最终融合：PAN {out_feat.shape} + "
            #     f"Origin {origin_feat.shape} → "
            #     f"{concat_feat.shape} → {fused_feat.shape}"
            # )

        return tuple(final_outs)

    @property
    def out_shape(self):
        return [
            ShapeSpec(
                channels=self.out_channels, stride=1. / s)
            for s in self.spatial_scales
        ]

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {'in_channels': [i.channels for i in input_shape]}