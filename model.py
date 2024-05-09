import torch
import torch.nn as nn
import torch.optim as optim
from conv_layers import ResBlock
from gru_layers import ConvGRU, ConvGRUCell
import math
import torch.nn.functional as F
import torch.nn.init as init

universal_dropout = 0.10
universal_drop_connect = 0.20

class DPC_RNN(nn.Module):
    def __init__(self, feature_size, hidden_size, kernel_size, num_layers, pred_steps, seq_len):
        super(DPC_RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.pred_steps = pred_steps
        self.seq_len = seq_len
        self.feature_size = feature_size

        # Initialize the multi-layer ConvGRU
        self.agg = ConvGRU(
            input_size=feature_size,
            hidden_size=hidden_size,
            kernel_size=kernel_size,
            num_layers=num_layers
        )

        self.network_pred = nn.Sequential(
                                nn.Conv2d(self.feature_size, self.feature_size, kernel_size=1, padding=0),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(self.feature_size, self.feature_size, kernel_size=1, padding=0)
                                )

        self.mask = None
        self.relu = nn.ReLU(inplace=False)
        self._initialize_weights(self.agg)
        self._initialize_weights(self.network_pred)

    def forward(self, block, B, N, C, SL, H, W):
        finalW = W
        finalH = H

        feature = F.avg_pool3d(block, ((self.seq_len, 1, 1)), stride=(1, 1, 1))
        del block
        feature_inf_all = feature.view(B, N, C, finalW, finalH)

        feature = self.relu(feature)
        feature = feature.view(B, N, C, finalW, finalH)

        feature_inf = feature_inf_all[:, N-self.pred_steps::, :].contiguous()
        del feature_inf_all

        _, hidden = self.agg(feature[:, 0:N-self.pred_steps, :])


        hidden = hidden[:, -1, :]
        future_context = F.avg_pool3d(hidden, (1, finalW, finalH), stride=1).squeeze(-1).squeeze(-1)

        pred = []
        for i in range(self.pred_steps): 
            p_tmp = self.network_pred(hidden)
            pred.append(p_tmp)

            _, hidden = self.agg(self.relu(p_tmp).unsqueeze(1), hidden_state = hidden.unsqueeze(0))
            hidden = hidden[:, -1, :]
        pred = torch.stack(pred, 1)
        del hidden

        # pred: [B, pred_step, D, last_size, last_size]
        # GT: [B, N, D, last_size, last_size]
        N = self.pred_steps

        pred = pred.permute(0, 1, 3, 4, 2).contiguous().view(B*self.pred_steps*finalH*finalW, self.feature_size)
        feature_inf = feature_inf.permute(0, 1, 3, 4, 2).contiguous().view(B*N*finalH*finalW, self.feature_size).transpose(0,1)
        score = torch.matmul(pred, feature_inf).view(B, self.pred_steps, finalH * finalW, B, N, finalH*finalW)
        del feature_inf, pred

        if self.mask is None:
            mask = torch.zeros((B, self.pred_steps, finalH*finalW, B, N, finalH*finalW), dtype=torch.bool, device=score.device).requires_grad_(False).detach()
            mask[torch.arange(B), :, :, torch.arange(B), :, :] = -3 # spatial negatives
            for k in range(B):
                mask[k, :, torch.arange(finalH*finalW), k, :, torch.arange(finalH*finalW)] = -1 # temporal negatives
            tmp = mask.permute(0, 2, 1, 3, 5, 4).contiguous().view(B*finalH*finalW, self.pred_steps, B*finalH*finalW, N)
            for j in range(B*finalH*finalW):
                tmp[j, torch.arange(self.pred_steps), j, torch.arange(N-self.pred_steps, N)] = 1

            mask = tmp.view(B, finalH * finalW, self.pred_steps, B, finalH * finalW, N).permute(0,2,1,3,5,4)

        return score, mask, future_context

    def _initialize_weights(self, module):
        for name, param in module.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.orthogonal_(param, 1)


class HemisphereStream(nn.Module):
    def __init__(self, num_blocks=9, slow_temporal_stride=16, fast_temporal_stride=2):
        super(HemisphereStream, self).__init__()
        self.initial_slow_conv = nn.Conv3d(3, 64, kernel_size=(1, 1, 1), stride=1, padding=0)
        self.initial_fast_conv = nn.Conv3d(3, 8, kernel_size=(1, 1, 1), stride=1, padding=0)

        self.slow_temporal_stride = slow_temporal_stride
        self.fast_temporal_stride = fast_temporal_stride
        
        slow_channels = [64 if i == 0 else 128 for i in range(num_blocks)]
        fast_channels = [8 if i == 0 else 16 for i in range(num_blocks)]

        # Slow pathway
        self.slow_blocks = nn.ModuleList([
            ResBlock(dim_in=slow_channels[i], dim_out=128, temp_kernel_size=3, stride=1, dim_inner=16) for i in range(num_blocks)
        ])
        # Fast pathway
        self.fast_blocks = nn.ModuleList([
            ResBlock(dim_in=fast_channels[i], dim_out=16, temp_kernel_size=3, stride=1, dim_inner=8) for i in range(num_blocks)  # Smaller inner dimension
        ])


    def forward(self, x):
        # Apply temporal stride correctly
        slow_input = self.initial_slow_conv(x[:, :, ::self.slow_temporal_stride, :, :])
        fast_input = self.initial_fast_conv(x[:, :, ::self.fast_temporal_stride, :, :])

        slow_output = slow_input
        fast_output = fast_input

        for slow_block, fast_block in zip(self.slow_blocks, self.fast_blocks):
            slow_output = slow_block(slow_output)
            fast_output = fast_block(fast_output)


        max_temporal_depth = max(slow_output.size(2), fast_output.size(2))
        max_height = max(slow_output.size(3), fast_output.size(3))
        max_width = max(slow_output.size(4), fast_output.size(4))

        slow_output = F.pad(slow_output, (0, max_width - slow_output.size(4), 0, max_height - slow_output.size(3), 0, max_temporal_depth - slow_output.size(2)))
        fast_output = F.pad(fast_output, (0, max_width - fast_output.size(4), 0, max_height - fast_output.size(3), 0, max_temporal_depth - fast_output.size(2)))
        
        # Fusion of slow and fast pathways
        output = torch.cat((slow_output, fast_output), dim=1)

        return output



class DualStream(nn.Module):
    def __init__(self):
        super(DualStream, self).__init__()
        self.left_hemisphere = HemisphereStream()
        self.right_hemisphere = HemisphereStream()
        self.dpc_rnn = DPC_RNN(feature_size=288, hidden_size=288, kernel_size=1, num_layers=1, pred_steps=3, seq_len=3) # normally 5 for SL
        # feature size is 144 * 2

    def forward(self, x):
        B, N, SL, C, H, W = x.shape
        x = x.view(B*N, C, SL, H, W)

        left_input = x[:, :, :, :, :W//2]
        right_input = x[:, :, :, :, W//2:]

        left_output = self.left_hemisphere(left_input)
        right_output = self.right_hemisphere(right_input)

        # Concatenate left and right hemisphere outputs
        concat_layer = torch.cat((left_output, right_output), dim=1)

        # Dropout can be added here if needed
        prediction, target, future_context = self.dpc_rnn(concat_layer, B, N, 288, SL, H, W//2)
        return prediction, target, concat_layer, future_context
