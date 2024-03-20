import torch
import torch.nn as nn
import torch.optim as optim
from conv_layers import ResBlock
from gru_layers import ConvGRU, ConvGRUCell
import math
import torch.nn.functional as F
import torch.nn.init as init

universal_dropout = 0.15
universal_drop_connect = 0.15

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
        finalW = 136
        finalH = 40

        feature = F.avg_pool3d(block, ((8, 1, 1)), stride=(1, 1, 1))
        feature_inf_all = feature.view(B, N, C, finalW, finalH)

        feature = self.relu(feature)
        feature = feature.view(B, N, C, finalW, finalH)

        feature_inf = feature_inf_all[:, N-self.pred_steps::, :].contiguous()
        del feature_inf_all

        _, hidden = self.agg(feature[:, 0:N-self.pred_steps, :])

        hidden = hidden[:, -1, :]

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

        #mask = torch.randint(low=0, high=1, size=score.shape, device=score.device)
        #print(mask)

        return score, mask

    def _initialize_weights(self, module):
        for name, param in module.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.orthogonal_(param, 1)


class DualStream(nn.Module):
    def __init__(self):
        super(DualStream, self).__init__()
        self.conv1_layer = nn.Conv3d(in_channels=3, out_channels=256, kernel_size=(5, 7, 7), padding=(2, 3, 3), stride=(1, 2, 2))
        self.conv1_pool = nn.AvgPool3d(kernel_size=(1, 3, 3), padding=(0, 1, 1), stride=(1, 2, 2))
        self.norm = nn.BatchNorm3d(256)
        self.relu = nn.LeakyReLU(negative_slope=0.01)

        self.stream1_block1 = ResBlock(dim_in=256, dim_out=128, temp_kernel_size=3, stride=1, dim_inner=32, drop_connect_rate=universal_drop_connect)
        self.stream1_block2 = ResBlock(dim_in=128, dim_out=128, temp_kernel_size=3, stride=1, dim_inner=32, drop_connect_rate=universal_drop_connect)
        self.stream1_block3 = ResBlock(dim_in=128, dim_out=128, temp_kernel_size=3, stride=1, dim_inner=32, drop_connect_rate=universal_drop_connect)
        self.stream1_block4 = ResBlock(dim_in=128, dim_out=128, temp_kernel_size=3, stride=1, dim_inner=32, drop_connect_rate=universal_drop_connect)
        #self.stream1_block5 = ResBlock(dim_in=128, dim_out=128, temp_kernel_size=3, stride=1, dim_inner=32, drop_connect_rate=universal_drop_connect)
        #self.stream1_block6 = ResBlock(dim_in=128, dim_out=128, temp_kernel_size=3, stride=1, dim_inner=32, drop_connect_rate=universal_drop_connect)
        #self.stream1_block7 = ResBlock(dim_in=128, dim_out=128, temp_kernel_size=3, stride=1, dim_inner=32, drop_connect_rate=universal_drop_connect).to('cuda:0')
        #self.stream1_block8 = ResBlock(dim_in=128, dim_out=128, temp_kernel_size=3, stride=1, dim_inner=32, drop_connect_rate=universal_drop_connect).to('cuda:0')
        #self.stream1_block9 = ResBlock(dim_in=128, dim_out=128, temp_kernel_size=3, stride=1, dim_inner=32, drop_connect_rate=universal_drop_connect).to('cuda:0')
        #self.stream1_block10 = ResBlock(dim_in=128, dim_out=128, temp_kernel_size=3, stride=1, dim_inner=32, drop_connect_rate=universal_drop_connect).to('cuda:0')

        self.stream2_block1 = ResBlock(dim_in=256, dim_out=128, temp_kernel_size=3, stride=1, dim_inner=32, drop_connect_rate=universal_drop_connect)
        self.stream2_block2 = ResBlock(dim_in=128, dim_out=128, temp_kernel_size=3, stride=1, dim_inner=32, drop_connect_rate=universal_drop_connect)
        self.stream2_block3 = ResBlock(dim_in=128, dim_out=128, temp_kernel_size=3, stride=1, dim_inner=32, drop_connect_rate=universal_drop_connect)
        self.stream2_block4 = ResBlock(dim_in=128, dim_out=128, temp_kernel_size=3, stride=1, dim_inner=32, drop_connect_rate=universal_drop_connect)
        #self.stream2_block5 = ResBlock(dim_in=128, dim_out=128, temp_kernel_size=3, stride=1, dim_inner=32, drop_connect_rate=universal_drop_connect)
        #self.stream2_block6 = ResBlock(dim_in=128, dim_out=128, temp_kernel_size=3, stride=1, dim_inner=32, drop_connect_rate=universal_drop_connect)
        #self.stream2_block7 = ResBlock(dim_in=128, dim_out=128, temp_kernel_size=3, stride=1, dim_inner=32, drop_connect_rate=universal_drop_connect).to('cuda:1')
        #self.stream2_block8 = ResBlock(dim_in=128, dim_out=128, temp_kernel_size=3, stride=1, dim_inner=32, drop_connect_rate=universal_drop_connect).to('cuda:1')
        #self.stream2_block9 = ResBlock(dim_in=128, dim_out=128, temp_kernel_size=3, stride=1, dim_inner=32, drop_connect_rate=universal_drop_connect).to('cuda:1')
        #self.stream2_block10 = ResBlock(dim_in=128, dim_out=128, temp_kernel_size=3, stride=1, dim_inner=32, drop_connect_rate=universal_drop_connect).to('cuda:1')

        
        """init.kaiming_normal_(self.conv1_layer.weight, mode='fan_out', nonlinearity='relu')
        
        # Initialize stream1 blocks with Kaiming initialization
        for block in [self.stream1_block1, self.stream1_block2]:
            for name, param in block.named_parameters():
                if 'weight' in name and param.dim() > 1:
                    init.kaiming_uniform_(param, a=math.sqrt(5))

        # Initialize stream2 blocks with Kaiming initialization
        for block in [self.stream2_block1, self.stream2_block2]:
            for name, param in block.named_parameters():
                if 'weight' in name and param.dim() > 1:
                    init.kaiming_uniform_(param, a=math.sqrt(5))"""

        self.dpc_rnn = DPC_RNN(feature_size=256, hidden_size=256, kernel_size=1, num_layers=1, pred_steps=3, seq_len=8).to('cuda:1')

    def forward(self, x):
        B, N, SL, C, H, W = x.shape
        x = x.view(B*N, C, SL, H, W)

        shared_layer = self.conv1_layer(x)

        shared_layer_pool = self.conv1_pool(shared_layer)
        shared_layer_norm = self.norm(shared_layer_pool)
        shared_layer_relu = self.relu(shared_layer_norm)

        stream1layer1 = self.stream1_block1(shared_layer_relu)
        stream1layer2 = self.stream1_block2(stream1layer1)
        stream1layer3 = self.stream1_block3(stream1layer2)
        stream1layer4 = self.stream1_block4(stream1layer3)
        #stream1layer5 = self.stream1_block5(stream1layer4)
        #stream1layer6 = self.stream1_block6(stream1layer5)
        #stream1layer7 = self.stream1_block7(stream1layer6)
        #stream1layer8 = self.stream1_block8(stream1layer7)
        #stream1layer9 = self.stream1_block9(stream1layer8)
        #stream1layer10 = self.stream1_block10(stream1layer9)

        stream2layer1 = self.stream2_block1(shared_layer_relu)
        stream2layer2 = self.stream2_block2(stream2layer1)
        stream2layer3 = self.stream2_block3(stream2layer2)
        stream2layer4 = self.stream2_block4(stream2layer3)
        #stream2layer5 = self.stream2_block5(stream2layer4)
        #stream2layer6 = self.stream2_block6(stream2layer5)
        #stream2layer7 = self.stream2_block7(stream2layer6)
        #stream2layer8 = self.stream2_block8(stream2layer7)
        #stream2layer9 = self.stream2_block9(stream2layer8)
        #stream2layer10 = self.stream2_block10(stream2layer9)

        concat_layer = torch.cat((stream1layer4, stream2layer4), dim=1)

        concat_layer = nn.Dropout(universal_dropout)(concat_layer)

        prediction, target = self.dpc_rnn(concat_layer, B, N, 256, SL, H, W)

        return prediction, target
    
    
