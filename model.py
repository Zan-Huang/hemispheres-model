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
        finalW = 64
        finalH = 32

        feature = F.avg_pool3d(block, ((SL, 1, 1)), stride=(1, 1, 1))
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

class DualStream(nn.Module):
    def __init__(self):
        super(DualStream, self).__init__()
        self.left_stream1_block1 = nn.Conv3d(in_channels=3, out_channels=64, kernel_size=(5, 7, 7), padding=(2, 3, 3), stride=(1, 2, 2))
        self.left_stream1_pool = nn.AvgPool3d(kernel_size=(1, 3, 3), padding=(0, 1, 1), stride=(1, 2, 2))
        self.left_norm = nn.BatchNorm3d(128)
        self.left_relu = nn.LeakyReLU(negative_slope=0.01)

        self.right_stream1_block1 = nn.Conv3d(in_channels=3, out_channels=64, kernel_size=(5, 7, 7), padding=(2, 3, 3), stride=(1, 2, 2))
        self.right_stream1_pool = nn.AvgPool3d(kernel_size=(1, 3, 3), padding=(0, 1, 1), stride=(1, 2, 2))
        self.right_norm = nn.BatchNorm3d(128)
        self.right_relu = nn.LeakyReLU(negative_slope=0.01)

        num_blocks = 18
        for i in range(2, num_blocks + 1):
            setattr(self, f'left_stream1_block{i}', ResBlock(dim_in=64, dim_out=64, temp_kernel_size=3, stride=1, dim_inner=16, drop_connect_rate=universal_drop_connect))
            setattr(self, f'right_stream1_block{i}', ResBlock(dim_in=64, dim_out=64, temp_kernel_size=3, stride=1, dim_inner=16, drop_connect_rate=universal_drop_connect))

        self.concat_hook_layer = nn.Identity()

        self.dpc_rnn = DPC_RNN(feature_size=128, hidden_size=128, kernel_size=1, num_layers=1, pred_steps=3, seq_len=5)

    def forward(self, x):
        num_blocks = 18
        B, N, SL, C, H, W = x.shape
        x = x.view(B*N, C, SL, H, W)

        left_input = x[:, :, :, :, :W//2]
        right_input = x[:, :, :, :, W//2:]
        
        left_output = self.left_stream1_block1(left_input)
        left_output = self.left_stream1_pool(left_output)
        left_output = self.left_relu(left_output)

        right_output = self.right_stream1_block1(right_input)
        right_output = self.right_stream1_pool(right_output)
        right_output = self.right_relu(right_output)

        for i in range(2, num_blocks + 1):
            left_output = getattr(self, f'left_stream1_block{i}')(left_output)
            right_output = getattr(self, f'right_stream1_block{i}')(right_output)
    
        concat_layer = torch.cat((left_output, right_output), dim=1)

        concat_layer = nn.Dropout(universal_dropout)(concat_layer)
        prediction, target, future_context = self.dpc_rnn(concat_layer, B, N, 128, SL, H, W)
        return prediction, target, concat_layer, future_context
