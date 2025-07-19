from typing import Any, Optional, Union, Callable

from segmentation_models_pytorch.base import (
    ClassificationHead,
    SegmentationHead,
    SegmentationModelPrior,
)
from segmentation_models_pytorch.encoders import get_encoder
from segmentation_models_pytorch.base.hub_mixin import supports_config_loading

from .decoder import SegformerDecoder

from models.deep_hough_transform.HT_cuda import HTIHT_Cuda
from models.deep_hough_transform.HT import HTIHT_Cpu
from models.overlock.overlock import DynamicConvBlock
from models.BiPriorNet import QueryModule

class Segformer(SegmentationModelPrior):
    """Segformer is simple and efficient design for semantic segmentation with Transformers

    Args:
        encoder_name: Name of the classification model that will be used as an encoder (a.k.a backbone)
            to extract features of different spatial resolution
        encoder_depth: A number of stages used in encoder in range [3, 5]. Each stage generate features
            two times smaller in spatial dimensions than previous one (e.g. for depth 0 we will have features
            with shapes [(N, C, H, W),], for depth 1 - [(N, C, H, W), (N, C, H // 2, W // 2)] and so on).
            Default is 5
        encoder_weights: One of **None** (random initialization), **"imagenet"** (pre-training on ImageNet) and
            other pretrained weights (see table with available weights for each encoder_name)
        decoder_segmentation_channels: A number of convolution filters in segmentation blocks, default is 256
        in_channels: A number of input channels for the model, default is 3 (RGB images)
        classes: A number of classes for output mask (or you can think as a number of channels of output mask)
        activation: An activation function to apply after the final convolution layer.
            Available options are **"sigmoid"**, **"softmax"**, **"logsoftmax"**, **"tanh"**, **"identity"**,
            **callable** and **None**. Default is **None**.
        upsampling: A number to upsample the output of the model, default is 4 (same size as input)
        aux_params: Dictionary with parameters of the auxiliary output (classification head). Auxiliary output is build
            on top of encoder if **aux_params** is not **None** (default). Supported params:
                - classes (int): A number of classes
                - pooling (str): One of "max", "avg". Default is "avg"
                - dropout (float): Dropout factor in [0, 1)
                - activation (str): An activation function to apply "sigmoid"/"softmax"
                    (could be **None** to return logits)
        kwargs: Arguments passed to the encoder class ``__init__()`` function. Applies only to ``timm`` models. Keys with ``None`` values are pruned before passing.

    Returns:
        ``torch.nn.Module``: **Segformer**

    .. _Segformer:
        https://arxiv.org/abs/2105.15203

    """

    @supports_config_loading
    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_depth: int = 5,
        encoder_weights: Optional[str] = "imagenet",
        decoder_segmentation_channels: int = 256,
        in_channels: int = 3,
        classes: int = 1,
        activation: Optional[Union[str, Callable]] = None,
        upsampling: int = 4,
        aux_params: Optional[dict] = None,
        img_size = 512,
        num_queries = 32,
        angle_res=3, rho_res=1,
        **kwargs: dict[str, Any],
    ):
        super().__init__()

        self.img_size = img_size
        self.downsampling_rate = [1, 2, 4, 8, 16, 32]

        if encoder_name == "mit-b1" or encoder_name == "mit-b4":
            self.channels = [3, 0, 64, 128, 320, 512]
        else:
            self.channels = [3, 0, 64, 128, 320, 512]
        
        self.query_module3 = QueryModule(num_queries, d_model=self.channels[-4])
        self.query_module4 = QueryModule(num_queries, d_model=self.channels[-3])
        self.query_module5 = QueryModule(num_queries, d_model=self.channels[-2])
        # self.query_module4 = QueryModule(num_queries, d_model=256)

        self.ht3 = HTIHT_Cpu(num_queries, num_queries, img_size // 4, img_size // 4, angle_res, rho_res)
        self.ht4 = HTIHT_Cpu(num_queries, num_queries, img_size // 8, img_size // 8, angle_res, rho_res)
        self.ht5 = HTIHT_Cpu(num_queries, num_queries, img_size // 16, img_size // 16, angle_res, rho_res)
        # self.ht6 = HTIHT_Cuda(num_queries, num_queries, img_size // 16, img_size // 16, angle_res, rho_res)

        # 添加DynamicConvBlock融合模块
        self.fuse3 = DynamicConvBlock(
            dim=self.channels[2],                  # 匹配p1_query的通道数
            ctx_dim=num_queries,       # 匹配p1_prior的通道数
            kernel_size=7,             # 根据特征图大小调整
            smk_size=3,                # 小核处理局部细节
            is_first=True,             # 首层不需要h_r
            drop_path=0.1,             # 正则化
            mlp_ratio=2                # 控制计算量
        )
        # 为每个层级添加融合模块
        self.fuse4 = DynamicConvBlock(self.channels[3], num_queries, 5, 3, is_first=True)
        self.fuse5 = DynamicConvBlock(self.channels[4], num_queries, 5, 3, is_first=True)  # 高层特征用小核
        # self.fuse4 = DynamicConvBlock(self.channels[5], num_queries, 5, 3, is_first=True)


        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
            **kwargs,
        )

        self.decoder = SegformerDecoder(
            encoder_channels=self.encoder.out_channels,
            encoder_depth=encoder_depth,
            segmentation_channels=decoder_segmentation_channels,
        )

        self.segmentation_head = SegmentationHead(
            in_channels=decoder_segmentation_channels,
            out_channels=classes,
            activation=activation,
            kernel_size=1,
            upsampling=upsampling,
        )

        if aux_params is not None:
            self.classification_head = ClassificationHead(
                in_channels=self.encoder.out_channels[-1], **aux_params
            )
        else:
            self.classification_head = None

        self.name = "segformer-{}".format(encoder_name)
        self.initialize()
