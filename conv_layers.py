import torch.nn as nn
import torch

INIT_WEIGHT_MEAN = -0.009
INIT_WEIGHT_STD = 0.152

class ResBlock(nn.Module):
    """
    ResBlock represents a Residual Block with a Bottleneck Transformation.

    Attributes:
        bottleneck (nn.Sequential): The main transformation pipeline of the block, consisting
                                    of a series of convolutional and normalization layers.
        use_projection (bool): Flag to determine whether a projection shortcut should be used,
                               which is necessary when the input and output dimensions differ.
        projection (nn.Sequential): A projection shortcut, used when `use_projection` is True.
        relu (nn.ReLU): The activation function applied after combining the main transformation
                        output and the shortcut path.
    """

    def __init__(
        self,
        dim_in,
        dim_out,
        temp_kernel_size,
        stride,
        dim_inner,
        num_groups=1,
        stride_1x1=False,
        inplace_relu=True,
        eps=1e-5,
        bn_mmt=0.1,
        dilation=1,
        norm_module=nn.BatchNorm3d,
        drop_connect_rate=0.2,
    ):
        """
        Args:
            dim_in (int): Number of input channels.
            dim_out (int): Number of output channels.
            temp_kernel_size (int): Temporal kernel size for the first convolution in the bottleneck.
            stride (int): Stride for the convolutions in the bottleneck.
            dim_inner (int): Number of channels for the intermediate layers in the bottleneck.
            num_groups (int, optional): Number of groups for grouped convolution. Defaults to 1.
            stride_1x1 (bool, optional): Whether to apply stride in the first 1x1 convolution.
                                        Defaults to False.
            inplace_relu (bool, optional): Whether to use in-place ReLU to save memory. Defaults to True.
            eps (float, optional): Epsilon value for batch normalization. Defaults to 1e-5.
            bn_mmt (float, optional): Momentum for batch normalization. Defaults to 0.1.
            dilation (int, optional): Dilation for convolutions. Defaults to 1.
            norm_module (nn.Module, optional): Normalization module to use. Defaults to nn.BatchNorm3d.
            drop_connect_rate (float, optional): Drop connect rate for regularization. Defaults to 0.0.
        """
        super(ResBlock, self).__init__()
        self.drop_connect_rate = drop_connect_rate

        self.use_projection = (dim_in != dim_out) or (stride != 1)

        # Bottleneck transformation pipeline
        self.bottleneck = nn.Sequential(
            # First convolutional layer (Tx1x1)
            nn.Conv3d(dim_in, dim_inner, [temp_kernel_size, 1, 1], stride=[1, stride if stride_1x1 else 1, stride if stride_1x1 else 1], padding=[temp_kernel_size // 2, 0, 0], bias=False),
            norm_module(num_features=dim_inner, eps=eps, momentum=bn_mmt),
            nn.ReLU(inplace=inplace_relu),

            # Second convolutional layer (1x3x3)
            nn.Conv3d(dim_inner, dim_inner, [1, 3, 3], stride=[1, 1 if stride_1x1 else stride, 1 if stride_1x1 else stride], padding=[0, dilation, dilation], groups=num_groups, bias=False, dilation=[1, dilation, dilation]),
            norm_module(num_features=dim_inner, eps=eps, momentum=bn_mmt),
            nn.ReLU(inplace=inplace_relu),

            # Third convolutional layer (1x1x1)
            nn.Conv3d(dim_inner, dim_out, [1, 1, 1], stride=[1, 1, 1], padding=0, bias=False),
            norm_module(num_features=dim_out, eps=eps, momentum=bn_mmt)
        )

        # Projection shortcut, used when input and output dimensions or resolutions differ
        if self.use_projection:
            self.projection = nn.Sequential(
                nn.Conv3d(dim_in, dim_out, kernel_size=1, stride=[1, stride, stride], padding=0, bias=False),
                norm_module(num_features=dim_out, eps=eps, momentum=bn_mmt)
            )

        self.relu = nn.ReLU(inplace=True)
        self._initialize_weights()

    def _initialize_weights(self):
        """
        Initializes weights of the convolutional layers using a normal distribution
        with specified mean and standard deviation.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.normal_(m.weight, mean=INIT_WEIGHT_MEAN, std=INIT_WEIGHT_STD)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _drop_connect(self, x, drop_ratio):
        """
        Applies DropConnect to the input tensor.

        DropConnect is a form of regularization where individual connections
        within the network are randomly dropped. It's similar to dropout, but
        applies to the weights rather than the activations.

        Args:
            x (torch.Tensor): The input tensor.
            drop_ratio (float): The probability of dropping a connection.

        Returns:
            torch.Tensor: The resulting tensor after applying DropConnect.
        """
        if self.training and drop_ratio > 0.0:
            mask = torch.bernoulli(torch.full([x.shape[0], 1, 1, 1, 1], 1 - drop_ratio, device=x.device, dtype=x.dtype))
            x = x * mask / (1 - drop_ratio)
        return x

    def forward(self, x):
        #print(f"Input shape: {x.shape}")
        transformed_x = self.bottleneck(x)
        #print(f"Shape after bottleneck: {transformed_x.shape}")
        
        transformed_x = self._drop_connect(transformed_x, self.drop_connect_rate)
        #remember to add the above back.

        if self.use_projection:
            x = self.projection(x)
            #print(f"Shape after projection: {x.shape}")

        return self.relu(x + transformed_x)
