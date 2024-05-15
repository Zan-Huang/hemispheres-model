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
        finalW = 11
        finalH = 7

        #feature = F.avg_pool3d(block, ((self.seq_len, 1, 1)), stride=(1, 1, 1))
        feature = block
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
    def __init__(self, num_blocks=9, slow_temporal_stride=10, fast_temporal_stride=2):
        super(HemisphereStream, self).__init__()
        self.initial_slow_conv = nn.Conv3d(3, 128, kernel_size=(1, 1, 1), stride=1, padding=0)
        self.initial_fast_conv = nn.Conv3d(3, 16, kernel_size=(1, 1, 1), stride=1, padding=0)

        self.slow_temporal_stride = slow_temporal_stride
        self.fast_temporal_stride = fast_temporal_stride

        #self.hemisphere_weight = nn.Parameter(torch.randn(128 * 128 * 128))  # Example dimensions from the output of a block
        
        slow_channels = [128 if i == 0 else 128 for i in range(num_blocks)]
        fast_channels = [16 if i == 0 else 16 for i in range(num_blocks)]

        # Slow pathway
        self.slow_blocks = nn.ModuleList([
            #ResBlock(dim_in=slow_channels[i], dim_out=128, temp_kernel_size=3, stride=1, dim_inner=16) for i in range(num_blocks)
            ResBlock(dim_in=slow_channels[0], dim_out=128, temp_kernel_size=3, stride=1, dim_inner=16),
            ResBlock(dim_in=slow_channels[1], dim_out=128, temp_kernel_size=3, stride=1, dim_inner=16),
            ResBlock(dim_in=slow_channels[2], dim_out=128, temp_kernel_size=3, stride=1, dim_inner=16),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=0),
            ResBlock(dim_in=slow_channels[3], dim_out=128, temp_kernel_size=3, stride=1, dim_inner=16),
            ResBlock(dim_in=slow_channels[4], dim_out=128, temp_kernel_size=3, stride=1, dim_inner=16),
            ResBlock(dim_in=slow_channels[5], dim_out=128, temp_kernel_size=3, stride=1, dim_inner=16),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(1, 2, 2), padding=0),
            ResBlock(dim_in=slow_channels[6], dim_out=128, temp_kernel_size=3, stride=1, dim_inner=16),
            ResBlock(dim_in=slow_channels[7], dim_out=128, temp_kernel_size=3, stride=1, dim_inner=16),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(1, 2, 2), padding=0)
        ])
        # Fast pathway
        self.fast_blocks = nn.ModuleList([
            #ResBlock(dim_in=fast_channels[i], dim_out=16, temp_kernel_size=3, stride=1, dim_inner=8) for i in range(num_blocks)  # Smaller inner dimension
            ResBlock(dim_in=fast_channels[0], dim_out=16, temp_kernel_size=3, stride=1, dim_inner=8),
            ResBlock(dim_in=fast_channels[1], dim_out=16, temp_kernel_size=3, stride=1, dim_inner=8),
            ResBlock(dim_in=fast_channels[2], dim_out=16, temp_kernel_size=3, stride=1, dim_inner=8),
            nn.MaxPool3d(kernel_size=(5, 2, 2), stride=(1, 2, 2), padding=0),
            ResBlock(dim_in=fast_channels[3], dim_out=16, temp_kernel_size=3, stride=1, dim_inner=8),
            ResBlock(dim_in=fast_channels[4], dim_out=16, temp_kernel_size=3, stride=1, dim_inner=8),
            ResBlock(dim_in=fast_channels[5], dim_out=16, temp_kernel_size=3, stride=1, dim_inner=8),
            nn.MaxPool3d(kernel_size=(6, 2, 2), stride=(1, 2, 2), padding=0),
            ResBlock(dim_in=fast_channels[6], dim_out=16, temp_kernel_size=3, stride=1, dim_inner=8),
            ResBlock(dim_in=fast_channels[7], dim_out=16, temp_kernel_size=3, stride=1, dim_inner=8),
            nn.MaxPool3d(kernel_size=(6, 2, 2), stride=(1, 2, 2), padding=0)
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
        
        #print("slow output shape", slow_output.shape)
        #print("fast output shape", fast_output.shape)
        # Fusion of slow and fast pathways
        output = torch.cat((slow_output, fast_output), dim=1)

        return output



class DualStream(nn.Module):
    def __init__(self):
        super(DualStream, self).__init__()
        self.left_hemisphere = HemisphereStream()
        self.right_hemisphere = HemisphereStream()
        self.dpc_rnn = DPC_RNN(feature_size=288, hidden_size=288, kernel_size=1, num_layers=1, pred_steps=3, seq_len=1) # normally 5 for SL

        self.left_ff_layers = nn.Sequential(
            nn.Linear(144, 72),
            nn.ReLU(),
            nn.Dropout(p=universal_dropout),
            nn.Linear(72, 36),
            nn.ReLU(),
            nn.Dropout(p=universal_dropout),
            nn.Linear(36, 1)
        )
        
        self.right_ff_layers = nn.Sequential(
            nn.Linear(144, 72),
            nn.ReLU(),
            nn.Dropout(p=universal_dropout),
            nn.Linear(72, 36),
            nn.ReLU(),
            nn.Dropout(p=universal_dropout),
            nn.Linear(36, 1)
        )

        self.left_predict_right_flat = None
        self.right_predict_left_flat = None

        self.dropout = nn.Dropout(p=universal_dropout)

    def _create_predict_layer(self, input_dim, output_dim):
        layer = nn.Sequential(
            nn.Linear(input_dim, 144),
            nn.ReLU(),
            nn.Linear(144, 200),
            nn.ReLU(),
            nn.Linear(200, 144),
            nn.ReLU(),
            nn.Linear(144, output_dim)
        )
        self._initialize_weights_kaiming(layer)
        return layer

    def _initialize_weights_kaiming(self, module):
        for m in module.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        B, N, SL, C, H, W = x.shape
        x = x.view(B*N, C, SL, H, W)

        left_input = x[:, :, :, :, :W//2]
        right_input = x[:, :, :, :, W//2:]

        left_output = self.left_hemisphere(left_input)
        right_output = self.right_hemisphere(right_input)

        # Reshape left and right outputs, flattening all dimensions except the last one (feature dimension 144)
        left_output_reshaped = left_output.view(-1, 144)
        right_output_reshaped = right_output.view(-1, 144)

        weighted_left_output = self.left_ff_layers(left_output_reshaped)
        weighted_right_output = self.right_ff_layers(right_output_reshaped)

        if self.left_predict_right_flat is None:
            self.left_predict_right_flat = self._create_predict_layer(left_output_reshaped.shape[0], right_output_reshaped.shape[0]).to('cuda')
        if self.right_predict_left_flat is None:
            self.right_predict_left_flat = self._create_predict_layer(right_output_reshaped.shape[0], left_output_reshaped.shape[0]).to('cuda')

        weighted_left_predict_right_flat = self.left_predict_right_flat(weighted_left_output.T)
        weighted_right_predict_left_flat = self.right_predict_left_flat(weighted_right_output.T)

        # Calculate cosine similarity between weighted left and right outputs
        hemisphere_mse = F.mse_loss(weighted_left_output, weighted_right_output, reduction='sum')
        left_predict_right_mse = F.mse_loss(weighted_left_predict_right_flat, right_output_reshaped, reduction='sum')
        right_predict_left_mse = F.mse_loss(weighted_right_predict_left_flat, left_output_reshaped, reduction='sum')

        """print("Hemisphere Cosine Score:", hemisphere_cosine_score.item())
        print("Left Predict Right Cosine Score:", left_predict_right_cosine_score.item())
        print("Right Predict Left Cosine Score:", right_predict_left_cosine_score.item())"""

        hemisphere_mse = 1e-6 * (hemisphere_mse + right_predict_left_mse + left_predict_right_mse)
        hemisphere_mse = torch.nan_to_num(hemisphere_mse)

        # Concatenate left and right hemisphere outputs
        concat_layer = torch.cat((left_output, right_output), dim=1)
        concat_layer = self.dropout(concat_layer)

        prediction, target, future_context = self.dpc_rnn(concat_layer, B, N, 288, SL, H, W//2)
        return prediction, target, concat_layer, future_context, hemisphere_mse.unsqueeze(0)
