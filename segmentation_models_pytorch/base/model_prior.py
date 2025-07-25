import torch
import warnings

from typing import TypeVar, Type
from . import initialization as init
from .hub_mixin import SMPHubMixin
from .utils import is_torch_compiling

T = TypeVar("T", bound="SegmentationModelPrior")


class SegmentationModelPrior(torch.nn.Module, SMPHubMixin):
    """Base class for all segmentation models."""

    _is_torch_scriptable = True
    _is_torch_exportable = True
    _is_torch_compilable = True

    # if model supports shape not divisible by 2 ^ n set to False
    requires_divisible_input_shape = True

    # Fix type-hint for models, to avoid HubMixin signature
    def __new__(cls: Type[T], *args, **kwargs) -> T:
        instance = super().__new__(cls, *args, **kwargs)
        return instance

    def initialize(self):
        init.initialize_decoder(self.decoder)
        init.initialize_head(self.segmentation_head)
        if self.classification_head is not None:
            init.initialize_head(self.classification_head)

    def check_input_shape(self, x):
        """Check if the input shape is divisible by the output stride.
        If not, raise a RuntimeError.
        """
        if not self.requires_divisible_input_shape:
            return

        h, w = x.shape[-2:]
        output_stride = self.encoder.output_stride
        if h % output_stride != 0 or w % output_stride != 0:
            new_h = (
                (h // output_stride + 1) * output_stride
                if h % output_stride != 0
                else h
            )
            new_w = (
                (w // output_stride + 1) * output_stride
                if w % output_stride != 0
                else w
            )
            raise RuntimeError(
                f"Wrong input shape height={h}, width={w}. Expected image height and width "
                f"divisible by {output_stride}. Consider pad your images to shape ({new_h}, {new_w})."
            )

    def forward(self, x):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""

        if not (
            torch.jit.is_scripting() or torch.jit.is_tracing() or is_torch_compiling()
        ):
            self.check_input_shape(x)

        features = self.encoder(x)
        
        p3_query = self.query_module3(features[-4])
        p4_query = self.query_module4(features[-3])
        p5_query = self.query_module5(features[-2])

        
        p3_prior = self.ht3(p3_query)
        p4_prior = self.ht4(p4_query)
        p5_prior = self.ht5(p5_query)
        
        p3_fused, _ = self.fuse3(features[-4], p3_prior, None)
        p4_fused, _ = self.fuse4(features[-3], p4_prior, None)
        p5_fused, _ = self.fuse5(features[-2], p5_prior, None)

        features[-4] = p3_fused
        features[-3] = p4_fused
        features[-2] = p5_fused
        # img_sizes = []
        # channels = []
        # print(len(features))
        # for feat in features:
        #     # 获取特征图的高度和宽度（假设是2D特征图，形状为 [batch, channels, height, width]）
        #     _, c, h, w = feat.shape
        #     img_sizes.append((h, w))
        #     channels.append(c)
        #     print(f"特征图的大小: height={h}, width={w}, c = {c}")  # 打印每个特征图的大小
        
        # downsampling_rate = [1, 2, 4, 8, 16, 32]
        # p6 ,p5 ,p4, p3 = features[-1], features[-2], features[-3], features[-4]

        decoder_output = self.decoder(features)

        masks = self.segmentation_head(decoder_output)

        if self.classification_head is not None:
            labels = self.classification_head(features[-1])
            return masks, labels

        return masks

    @torch.inference_mode()
    def predict(self, x):
        """Inference method. Switch model to `eval` mode, call `.forward(x)` with `torch.inference_mode()`

        Args:
            x: 4D torch tensor with shape (batch_size, channels, height, width)

        Return:
            prediction: 4D torch tensor with shape (batch_size, classes, height, width)

        """
        if self.training:
            self.eval()
        x = self(x)
        return x

    def load_state_dict(self, state_dict, **kwargs):
        # for compatibility of weights for
        # timm- ported encoders with TimmUniversalEncoder
        from segmentation_models_pytorch.encoders import TimmUniversalEncoder

        if isinstance(self.encoder, TimmUniversalEncoder):
            patterns = ["regnet", "res2", "resnest", "mobilenetv3", "gernet"]
            is_deprecated_encoder = any(
                self.encoder.name.startswith(pattern) for pattern in patterns
            )
            if is_deprecated_encoder:
                keys = list(state_dict.keys())
                for key in keys:
                    new_key = key
                    if key.startswith("encoder.") and not key.startswith(
                        "encoder.model."
                    ):
                        new_key = "encoder.model." + key.removeprefix("encoder.")
                    if "gernet" in self.encoder.name:
                        new_key = new_key.replace(".stages.", ".stages_")
                    state_dict[new_key] = state_dict.pop(key)

        # To be able to load weight with mismatched sizes
        # We are going to filter mismatched sizes as well if strict=False
        strict = kwargs.get("strict", True)
        if not strict:
            mismatched_keys = []
            model_state_dict = self.state_dict()
            common_keys = set(model_state_dict.keys()) & set(state_dict.keys())
            for key in common_keys:
                if model_state_dict[key].shape != state_dict[key].shape:
                    mismatched_keys.append(
                        (key, model_state_dict[key].shape, state_dict[key].shape)
                    )
                    state_dict.pop(key)

            if mismatched_keys:
                str_keys = "\n".join(
                    [
                        f" - {key}: {s} (weights) -> {m} (model)"
                        for key, m, s in mismatched_keys
                    ]
                )
                text = f"\n\n !!!!!! Mismatched keys !!!!!!\n\nYou should TRAIN the model to use it:\n{str_keys}\n"
                warnings.warn(text, stacklevel=-1)

        return super().load_state_dict(state_dict, **kwargs)
