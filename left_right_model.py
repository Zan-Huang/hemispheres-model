import torch
import torch.nn as nn
import torch.optim as optim
from conv_layers import ResBlock
import math
import torch.nn.functional as F
from torchvision import transforms
import torch.nn.init as init
import math

universal_dropout = 0.1
universal_drop_connect = 0.20
eps = 1e-4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class HemisphereStream(nn.Module):
    def __init__(self, num_blocks=18, slow_temporal_stride=15, fast_temporal_stride=2):
        super(HemisphereStream, self).__init__()
        self.initial_slow_conv = nn.Conv3d(3, 64, kernel_size=(1, 7, 7), stride=1, padding=0)
        self.initial_fast_conv = nn.Conv3d(3, 8, kernel_size=(1, 7, 7), stride=1, padding=0)

        self.slow_temporal_stride = slow_temporal_stride
        self.fast_temporal_stride = fast_temporal_stride

        # Slow pathway
        self.slow_blocks = nn.ModuleList([
            ResBlock(dim_in=64, dim_out=64, temp_kernel_size=3, stride=1, dim_inner=64),
            ResBlock(dim_in=64, dim_out=64, temp_kernel_size=3, stride=1, dim_inner=64),
            ResBlock(dim_in=64, dim_out=64, temp_kernel_size=3, stride=1, dim_inner=64),
            ResBlock(dim_in=64, dim_out=128, temp_kernel_size=3, stride=1, dim_inner=128),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=0),
            ResBlock(dim_in=128, dim_out=128, temp_kernel_size=3, stride=1, dim_inner=128),
            ResBlock(dim_in=128, dim_out=128, temp_kernel_size=3, stride=1, dim_inner=128),
            ResBlock(dim_in=128, dim_out=128, temp_kernel_size=3, stride=1, dim_inner=128),
            ResBlock(dim_in=128, dim_out=256, temp_kernel_size=3, stride=1, dim_inner=256),
            nn.MaxPool3d(kernel_size=(2, 4, 4), stride=(2, 1, 1), padding=0),
            ResBlock(dim_in=256, dim_out=256, temp_kernel_size=3, stride=1, dim_inner=256),
            ResBlock(dim_in=256, dim_out=256, temp_kernel_size=3, stride=1, dim_inner=256),
            ResBlock(dim_in=256, dim_out=256, temp_kernel_size=3, stride=1, dim_inner=256),
            ResBlock(dim_in=256, dim_out=512, temp_kernel_size=3, stride=1, dim_inner=512),
            ResBlock(dim_in=512, dim_out=512, temp_kernel_size=3, stride=1, dim_inner=512),
            ResBlock(dim_in=512, dim_out=512, temp_kernel_size=3, stride=1, dim_inner=512)
        ])
        # Fast pathway
        self.fast_blocks = nn.ModuleList([
            ResBlock(dim_in=8, dim_out=8, temp_kernel_size=3, stride=1, dim_inner=8),
            ResBlock(dim_in=8, dim_out=8, temp_kernel_size=3, stride=1, dim_inner=8),
            ResBlock(dim_in=8, dim_out=8, temp_kernel_size=3, stride=1, dim_inner=8),
            ResBlock(dim_in=8, dim_out=16, temp_kernel_size=3, stride=1, dim_inner=16),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=0),
            ResBlock(dim_in=16, dim_out=16, temp_kernel_size=3, stride=1, dim_inner=16),
            ResBlock(dim_in=16, dim_out=16, temp_kernel_size=3, stride=1, dim_inner=16),
            ResBlock(dim_in=16, dim_out=16, temp_kernel_size=3, stride=1, dim_inner=16),
            ResBlock(dim_in=16, dim_out=32, temp_kernel_size=3, stride=1, dim_inner=32),
            nn.MaxPool3d(kernel_size=(10, 4, 4), stride=(8, 4, 4), padding=0),
            ResBlock(dim_in=32, dim_out=32, temp_kernel_size=3, stride=1, dim_inner=32),
            ResBlock(dim_in=32, dim_out=32, temp_kernel_size=3, stride=1, dim_inner=32),
            ResBlock(dim_in=32, dim_out=32, temp_kernel_size=3, stride=1, dim_inner=32),
            ResBlock(dim_in=32, dim_out=64, temp_kernel_size=3, stride=1, dim_inner=64),
            ResBlock(dim_in=64, dim_out=64, temp_kernel_size=3, stride=1, dim_inner=64),
            ResBlock(dim_in=64, dim_out=64, temp_kernel_size=3, stride=1, dim_inner=64) 
        ])

        self.slow_linear1 = nn.Linear(512, 512)
        self.slow_linear2 = nn.Linear(512, 512)
        self.fast_linear1 = nn.Linear(64, 64)
        self.fast_linear2 = nn.Linear(64, 64)


    def forward(self, x):
        slow_input = self.initial_slow_conv(x[:, :, ::self.slow_temporal_stride, :, :])
        fast_input = self.initial_fast_conv(x[:, :, ::self.fast_temporal_stride, :, :])

        for slow_block, fast_block in zip(self.slow_blocks, self.fast_blocks):
            slow_input = slow_block(slow_input)
            fast_input = fast_block(fast_input)
            #print(f"Fast block output shape: {fast_input.shape}")
            #print(f"Slow block output shape: {slow_input.shape}")

        slow_pooled = slow_input.mean(dim=[2, 3, 4])
        fast_pooled = fast_input.mean(dim=[2, 3, 4])

        slow_pooled = self.slow_linear1(slow_pooled)
        slow_pooled = nn.ReLU()(slow_pooled)
        slow_pooled = self.slow_linear2(slow_pooled)
        slow_pooled = nn.ReLU()(slow_pooled)

        fast_pooled = self.fast_linear1(fast_pooled)
        fast_pooled = nn.ReLU()(fast_pooled)
        fast_pooled = self.fast_linear2(fast_pooled)
        fast_pooled = nn.ReLU()(fast_pooled)

        return slow_pooled, fast_pooled


class DualStream(nn.Module):
    def __init__(self):
        super(DualStream, self).__init__()
        self.shared_hemisphere_stream = HemisphereStream()

        self.slow_flat_num = 512
        self.fast_flat_num = 64

        self.dropout = nn.Dropout(universal_dropout)

        self.hemisphere_slow_reduction_mlp = nn.Sequential(
            nn.Linear(self.slow_flat_num, 2048),
            nn.BatchNorm1d(2048),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(0.1),
            nn.Linear(2048, 4000),
            nn.BatchNorm1d(4000),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(0.1),
            nn.Linear(4000, 4000),
            nn.BatchNorm1d(4000),
            nn.Linear(4000, 4000),
            nn.BatchNorm1d(4000)
        )

        self.hemisphere_fast_reduction_mlp = nn.Sequential(
            nn.Linear(self.fast_flat_num, 500),
            nn.BatchNorm1d(500),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(0.1),
            nn.Linear(500, 4000),
            nn.BatchNorm1d(4000),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(0.1),
            nn.Linear(4000, 4000),
            nn.BatchNorm1d(4000),
            nn.Linear(4000, 4000),
            nn.BatchNorm1d(4000)
        )

        self.mse_loss = nn.MSELoss()

    def forward(self, x):
        B, N, SL, C, H, W = x.shape
        x = x.view(B*N, C, SL, H, W)

        left_input = x[:, :, :, :, :W//2]
        right_input = x[:, :, :, :, W//2:]
        
        left_slow, left_fast = self.shared_hemisphere_stream(left_input)
        right_slow, right_fast = self.shared_hemisphere_stream(right_input)

        #print("Left Slow Dimensions:", left_slow.shape)
        #print("Right Slow Dimensions:", right_slow.shape)
        #print("Left Fast Dimensions:", left_fast.shape)
        #print("Right Fast Dimensions:", right_fast.shape)

        left_slow_reduced = self.hemisphere_slow_reduction_mlp(left_slow)
        right_slow_reduced = self.hemisphere_slow_reduction_mlp(right_slow)

        left_fast_reduced = self.hemisphere_fast_reduction_mlp(left_fast)
        right_fast_reduced = self.hemisphere_fast_reduction_mlp(right_fast)

        return left_slow_reduced, left_fast_reduced, right_slow_reduced, right_fast_reduced

