###########################################################################
#
# This model is ported from the following implement.
# https://github.com/crocodilegogogo/IF-ConvTransformer-UbiComp2022
#
# [NOTE]
# Due to the difference in benchmark data handling implementation, 
# some features are not ported well enough.
# In addtion, this implementation have several limits.
# 
# 1: The imu data should be converted from (acc, gyro, quaternion) to
#    (grav_angle, gyro, acc) rotated by the quaternion.
#    However, it is not implemented. 
#    This implementation uses (mag, gyro, acc) instead.
# 2: The original implementation used z-transformed data. 
#    However, it is not implemented. 
#    It is replaced by BatchNormalization for input data.
# 3: Only a few datasets can work with it.
#    If you want to enable a new dataset, 
#    please implement extract_imu_tensor_func for the dataset.
# 4: There is no definition of non-IMU data handling in the original, 
#    So, non-IMU data input is ignored.
#    This ignoring may cause of performance decrease.
#
###########################################################################

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import math
from torch.nn.utils import weight_norm
from .utils import * 

def get_config(dataset, lr_magnif=1):

    imu_num = None
    imu_channel_num = None
    if dataset.startswith("pamap2"):
        extract_imu_tensor_func = extract_imu_tensor_func_pamap2
        imu_num = 3
        imu_channel_num = 9
    #elif dataset.startswith('opportunity'):
    #    extract_imu_tensor_func = extract_imu_tensor_func_oppotunity
    #    imu_num = 5
    #elif dataset == "daphnet":
    #    extract_imu_tensor_func = extract_imu_tensor_func_daphnet
    #elif dataset.startswith("pamap2-separation"):
    #    extract_imu_tensor_func = extract_imu_tensor_func_pamap2_separation
    #elif dataset.startswith('opportunity_real-task_c'):
    #    extract_imu_tensor_func = extract_imu_tensor_func_oppotunity_task_c
    #elif dataset.startswith('opportunity-separation'):
    #    extract_imu_tensor_func = extract_imu_tensor_func_oppotunity_separation
    elif dataset.startswith('ucihar'):
        extract_imu_tensor_func = extract_imu_tensor_func_ucihar
        imu_num = 1
        imu_channel_num = 9
    #elif dataset.startswith('wisdm'):
    #    extract_imu_tensor_func = extract_imu_tensor_func_wisdm
    #elif dataset.startswith('m_health'):
    #    extract_imu_tensor_func = extract_imu_tensor_func_m_health
    #elif dataset.startswith('real_world-separation'):
    #    extract_imu_tensor_func = extract_imu_tensor_func_real_world_separation
    #elif dataset.startswith('real_world'):
    #    extract_imu_tensor_func = extract_imu_tensor_func_real_world
    #elif dataset.startswith('mighar'):
    #    extract_imu_tensor_func = extract_imu_tensor_func_mighar
    else:
        raise NotImplementedError(f"No extract_imu_tensor_func implementation for {dataset}")


    return {'input_2Dfeature_channel': 1, 
            'feature_channel': 64,
            'kernel_size': 5,
            'kernel_size_grav': 3, 
            'scale_num': 2,
            'feature_channel_out': 128,
            'multiheads': 1,
            'drop_rate': 0.2,
            'learning_rate': 0.001 * lr_magnif,
            'imu_num': imu_num, 
            'input_channel': imu_channel_num,
            'extract_imu_tensor_func': extract_imu_tensor_func,
            'regularization_rate': 0.01}

def gen_model(input_shape, n_classes, out_loss, out_activ, metrics, config):
    """
        input_shape is [total num of batches, time-series, sensors(=channel)]
    """
    return If_ConvTransformer_W(orig_input_channel=input_shape[2], data_length=input_shape[1], num_class=n_classes, out_activ=out_activ, **config)


def gen_preconfiged_model(input_shape, n_classes, out_loss, out_activ, dataset, metrics=['accuracy'], lr_magnif=1):
    config = get_config(dataset, lr_magnif)
    return gen_model(input_shape, n_classes, out_loss, out_activ, metrics, config), config


def get_optim_config(dataset, trial, lr_magnif=1):
    raise NotImplementedError("No config for optimization")


def get_dnn_framework_name():
    return 'pytorch'


class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=128):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0., d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        pe = pe.transpose(1,2)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)],requires_grad=False)
        # x = x + Variable(self.pe, requires_grad=False)
        return self.dropout(x)

class SelfAttention(nn.Module):
    def __init__(self, k, heads = 8, drop_rate = 0):
        super(SelfAttention, self).__init__()
        self.k, self.heads = k, heads
        # map k-dimentional input to k*heads dimentions
        self.tokeys    = nn.Linear(k, k * heads, bias = False)
        self.toqueries = nn.Linear(k, k * heads, bias = False)
        self.tovalues  = nn.Linear(k, k * heads, bias = False)
        # set dropout
        self.dropout_attention = nn.Dropout(drop_rate)
        # squeeze dimention to k
        self.unifyheads = nn.Linear(heads * k, k)
        
    def forward(self, x):
        
        b, t, k = x.size()
        h = self.heads
        queries = self.toqueries(x).view(b, t, h, k)
        keys    = self.tokeys(x).view(b, t, h, k)
        values  = self.tovalues(x).view(b, t, h, k)
        # squeeze head into batch dimension
        queries = queries.transpose(1, 2).contiguous().view(b * h, t, k)
        keys    = keys.transpose(1, 2).contiguous().view(b * h, t, k)
        values  = values.transpose(1, 2).contiguous().view(b * h, t, k)
        # normalize the dot products
        queries = queries / (k ** (1/4))
        keys = keys / (k ** (1/4))
        # matrix multiplication
        dot  = torch.bmm(queries, keys.transpose(1,2))
        # softmax normalization
        dot = F.softmax(dot, dim=2)
        dot = self.dropout_attention(dot)
        out = torch.bmm(dot, values).view(b, h, t, k)
        # swap h, t back, unify heads
        out = out.transpose(1, 2).contiguous().view(b, t, h*k)
        
        return self.unifyheads(out) # (b, t, k)

def conv1d(ni: int, no: int, ks: int = 1, stride: int = 1, padding: int = 0, bias: bool = False):
    """
    Create and initialize a `nn.Conv1d` layer with spectral normalization.
    """
    conv = nn.Conv1d(ni, no, ks, stride=stride, padding=padding, bias=bias)
    nn.init.kaiming_normal_(conv.weight)
    if bias:
        conv.bias.data.zero_()
    # return spectral_norm(conv)
    return conv

class SelfAttention_Branch(nn.Module):
    
    def __init__(self, n_channels: int, drop_rate = 0, div = 1):
        super(SelfAttention_Branch, self).__init__()

        self.n_channels = n_channels

        if n_channels > 1:
            self.query = conv1d(n_channels, n_channels//div)
            self.key = conv1d(n_channels, n_channels//div)
        else:
            self.query = conv1d(n_channels, n_channels)
            self.key = conv1d(n_channels, n_channels)
        self.value = conv1d(n_channels, n_channels)
        self.dropout_attention = nn.Dropout(drop_rate)
        self.gamma = nn.Parameter(torch.tensor([0.]))

    def forward(self, x):
        
        x = x.permute(0,2,1)
        size = x.size()
        x = x.view(*size[:2], -1)
        f, g, h = (self.query(x) / (self.n_channels ** (1/4))), (self.key(x) / (self.n_channels ** (1/4))), self.value(x)
        beta = F.softmax(torch.bmm(f.permute(0, 2, 1).contiguous(), g), dim=1)
        beta = self.dropout_attention(beta)
        o = self.gamma * torch.bmm(h, beta) + x
        return o.view(*size).contiguous().permute(0,2,1)

class TransformerBlock(nn.Module):
    def __init__(self, k, heads, drop_rate):
        super(TransformerBlock, self).__init__()

        self.attention = SelfAttention(k, heads = heads, drop_rate = drop_rate)
        # self.norm1 = nn.LayerNorm(k)
        self.norm1 = nn.BatchNorm1d(k)

        self.mlp = nn.Sequential(
            nn.Linear(k, 4*k),
            nn.ReLU(),
            nn.Linear(4*k, k)
        )
        # self.norm2 = nn.LayerNorm(k)
        self.norm2 = nn.BatchNorm1d(k)
        self.dropout_forward = nn.Dropout(drop_rate)

    def forward(self, x):
        
        # perform self-attention
        attended = self.attention(x)
        attended = attended + x
        attended = attended.permute(0,2,1)
        # perform layer norm
        x = self.norm1(attended).permute(0,2,1)
        # feedforward and layer norm
        feedforward = self.mlp(x)
        
        feedforward = feedforward + x
        feedforward = feedforward.permute(0,2,1)
        
        return self.dropout_forward(self.norm2(feedforward).permute(0,2,1))

class Chomp2d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp2d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :, :-self.chomp_size].contiguous()
    
class IMU_Fusion_Block(nn.Module):
    def __init__(self, input_2Dfeature_channel, input_channel, 
                 feature_channel, kernel_size_grav,
                 scale_num):
        super(IMU_Fusion_Block, self).__init__()
        
        self.scale_num         = scale_num
        self.input_channel     = input_channel
        self.tcn_grav_convs    = []
        self.tcn_gyro_convs    = []
        self.tcn_acc_convs     = []

        for i in range(self.scale_num):
            
            dilation_num_grav = i+1
            
            padding_grav      = (kernel_size_grav - 1) * dilation_num_grav
            kernel_size_gyro  = padding_grav
            kernel_size_acc   = padding_grav + 1
            
            tcn_grav = nn.Sequential(
                weight_norm(nn.Conv2d(input_2Dfeature_channel, feature_channel, 
                                      (1,kernel_size_grav), 1, (0,padding_grav), 
                                      dilation=dilation_num_grav)),
                Chomp2d(padding_grav),
                nn.ReLU(),
                )
            
            
            if kernel_size_gyro == 1:
                tcn_gyro = nn.Sequential(
                    weight_norm(nn.Conv2d(input_2Dfeature_channel, feature_channel, 
                                          (1,1), 1, (0,0), 
                                          dilation=1)),
                    nn.ReLU(),
                    )
            else:
                tcn_gyro = nn.Sequential(
                    weight_norm(nn.Conv2d(input_2Dfeature_channel, feature_channel, 
                                          (1,kernel_size_gyro), 1, (0,(kernel_size_gyro-1)*1), 
                                          dilation=1)),
                    Chomp2d((kernel_size_gyro-1)*1),
                    nn.ReLU(),
                    )
            
            tcn_acc = nn.Sequential(
                weight_norm(nn.Conv2d(input_2Dfeature_channel, feature_channel, 
                                      (1,kernel_size_acc), 1, (0,(kernel_size_acc-1)*1), 
                                      dilation=1)),
                Chomp2d((kernel_size_acc-1)*1),
                nn.ReLU(),
                )
            
            setattr(self, 'tcn_grav_convs%i' % i, tcn_grav)
            self.tcn_grav_convs.append(tcn_grav)
            setattr(self, 'tcn_gyro_convs%i' % i, tcn_gyro)
            self.tcn_gyro_convs.append(tcn_gyro)
            setattr(self, 'tcn_acc_convs%i' % i, tcn_acc)
            self.tcn_acc_convs.append(tcn_acc)
        
        self.attention = nn.Sequential(
                nn.Linear(3*feature_channel, 1),
                # nn.Tanh()
                nn.PReLU()
                )
        
    def forward(self, x):
        
        x_grav = x[:,:,0:3,:]
        x_gyro = x[:,:,3:6,:]
        x_acc  = x[:,:,6:9,:]
    
        for i in range(self.scale_num):
            
            out_grav = self.tcn_grav_convs[i](x_grav).unsqueeze(4)
            out_gyro = self.tcn_gyro_convs[i](x_gyro).unsqueeze(4)
            out_acc  = self.tcn_acc_convs[i](x_acc)
            
            if i == 0:
                out_attitude = torch.cat([out_grav, out_gyro], dim=4)
                out_dynamic  = out_acc
            else:
                out_attitude = torch.cat([out_attitude, out_grav], dim=4)
                out_attitude = torch.cat([out_attitude, out_gyro], dim=4)
                out_dynamic  = torch.cat([out_dynamic, out_acc], dim=2)

        # [MEMO]
        # print(f'{out_attitude.shape=}')         
        # out_attitude.shape=torch.Size([64, 64, 3, 128, 4])
        # The following sensor_num is 2, meaning the grav and gyro.

        # (batch_size, time_length, sensor_num*scale_num, 3(xyz), feature_chnnl)
        out_attitude = out_attitude.permute(0,3,4,2,1)
        
        # (batch_size, time_length, sensor_num*scale_num, 3(xyz)*feature_chnnl)
        out_attitude = out_attitude.reshape(out_attitude.shape[0], out_attitude.shape[1], out_attitude.shape[2], -1)
        # time-step-wise sensor attention, sensor_attn:(batch_size, time_length, sensor_num*scale_num, 1)
        sensor_attn  = self.attention(out_attitude).squeeze(3)
        sensor_attn  = F.softmax(sensor_attn, dim=2).unsqueeze(-1)
        out_attitude = sensor_attn * out_attitude
        
        # used for normalization
        norm_num     = torch.mean(sensor_attn.squeeze(-1), dim=1)
        norm_num     = torch.pow(norm_num, 2)
        norm_num     = torch.sqrt(torch.sum(norm_num, dim=1))
        norm_num     = (pow(self.scale_num,0.5)/norm_num).unsqueeze(1).unsqueeze(2).unsqueeze(3)
        
        out_attitude = out_attitude * norm_num
        
        # (batch_size, time_length, sensor_num*scale_num, 3(xyz), feature_chnnl)
        out_attitude = out_attitude.reshape(out_attitude.shape[0], out_attitude.shape[1], out_attitude.shape[2], 3, -1)
        # (batch_size, time_length, sensor_num*scale_num*3(xyz), feature_chnnl)
        out_attitude = out_attitude.reshape(out_attitude.shape[0], out_attitude.shape[1], out_attitude.shape[2]*3, -1)
        # (batch_size, feature_chnnl, sensor_num*scale_num*3(xyz), time_length)
        out_attitude = out_attitude.permute(0,3,2,1)
        
        # concatenate all the different scales
        out_attitude = torch.split(out_attitude, 6, dim=2)
        for j in range(len(out_attitude)):
            per_scale_attitude = torch.split(out_attitude[j], 3, dim=2)
            for k in range(len(per_scale_attitude)):
                if k == 0:
                    per_attitude   = per_scale_attitude[k]
                else:
                    per_attitude   = per_attitude + per_scale_attitude[k]
            if j == 0:
                all_attitude = per_attitude
            else:
                all_attitude = torch.cat([all_attitude, per_attitude], dim=2)
        out_attitude = all_attitude
        
        out          = torch.cat([out_attitude, out_dynamic], dim = 2)
        
        return out, sensor_attn

class If_ConvTransformer_W(nn.Module):
    def __init__(self, input_2Dfeature_channel, input_channel, feature_channel,
                 kernel_size, kernel_size_grav, scale_num, feature_channel_out,
                 multiheads, drop_rate, data_length, num_class,
                 extract_imu_tensor_func, imu_num, orig_input_channel,
                 out_activ, learning_rate = 0.001, regularization_rate=0.01):
        
        super(If_ConvTransformer_W, self).__init__()
       
        self.initial_learning_rate = learning_rate
        self.extract_imu_tensor_func = extract_imu_tensor_func
        self.imu_num = imu_num

        self.feature_channel  = feature_channel
        self.scale_num        = scale_num

        self.IMU_fusion_blocks     = []
        for i in range(imu_num):
            IMU_fusion_block   = IMU_Fusion_Block(input_2Dfeature_channel, input_channel, feature_channel,
                                                  kernel_size_grav, scale_num)
            setattr(self, 'IMU_fusion_blocks%i' % i, IMU_fusion_block)
            self.IMU_fusion_blocks.append(IMU_fusion_block)
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(feature_channel, feature_channel, (1,kernel_size), 1, (0,kernel_size//2)),
            nn.BatchNorm2d(feature_channel),
            nn.ReLU(),
            )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(feature_channel, feature_channel, (1,kernel_size), 1, (0,kernel_size//2)),
            nn.BatchNorm2d(feature_channel),
            nn.ReLU(),
            )
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(feature_channel, feature_channel, (1,kernel_size), 1, (0,kernel_size//2)),
            nn.BatchNorm2d(feature_channel),
            nn.ReLU(),
            )
        
        # TODO check
        # It is hardcorded in original.
        # What is it?
        # go the channel number to half?
        if input_channel  == 12:
            reduced_channel = 6
        else:
            reduced_channel = 3
        
        self.norm_conv4  = nn.LayerNorm(feature_channel)

        self.sa         = SelfAttention_Branch(feature_channel, drop_rate = drop_rate)

        self.transition = nn.Sequential(
            nn.Conv1d(feature_channel*(9-reduced_channel)*scale_num*imu_num, feature_channel_out, 1, 1),
            nn.BatchNorm1d(feature_channel_out),
            nn.ReLU()
            )

        self.position_encode = PositionalEncoding(feature_channel_out, drop_rate, data_length)
        
        self.transformer_block1 = TransformerBlock(feature_channel_out, multiheads, drop_rate)
        
        self.transformer_block2 = TransformerBlock(feature_channel_out, multiheads, drop_rate)
        
        self.global_ave_pooling = nn.AdaptiveAvgPool1d(1)
        
        self.register_buffer(
            "centers", (torch.randn(num_class, feature_channel_out).cuda())
        )
        
        self.z_trans = nn.BatchNorm1d(orig_input_channel)

        self.linear_for_out = nn.Linear(feature_channel_out, num_class)

        if out_activ == 'softmax':
            self.activation_for_out = nn.LogSoftmax(dim=1)
        else:
            raise NotImplementedError(f'Activation function {out_activ} is not implemented yet')

    def forward(self, x):
        
        
        # change axes order
        # The input is [B, time-series, sensors]
        # This model needs [B, C(=1), sensors, time-series]
        x = torch.transpose(x, 2, 1) # swap to [B, Sensors, time-series]
        # z-transform for each axis
        x = self.z_trans(x)
        x = torch.unsqueeze(x, 1) # add channel, to [B, C, Sensors, time-series]
        # TODO should be convert: acc,gyro,mag to grav_angle, gyro, acc
        # TODO: has to do rotation with quotanion

        batch_size      = x.shape[0]
        data_length     = x.shape[3]

        for i, imu_x in enumerate(self.extract_imu_tensor_func(x)):
            #x_cur_IMU, cur_sensor_attn   = self.IMU_fusion_blocks[i](x_input[:,:,i*9:(i+1)*9,:])
            x_cur_IMU, cur_sensor_attn   = self.IMU_fusion_blocks[i](imu_x)

            if i == 0:
                x        = x_cur_IMU
                out_attn = cur_sensor_attn
            else:
                x        = torch.cat((x, x_cur_IMU), 2)
                out_attn = torch.cat((out_attn, cur_sensor_attn), 2)
        
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x) # [batch_size, fea_dims, sensor_chnnl, data_len]
        
        x = x.permute(0, 3, 2, 1)
        x = self.norm_conv4(x).permute(0, 3, 2, 1)
       
        x = x.permute(0,3,2,1).reshape(batch_size*data_length, -1, self.feature_channel)
        x = self.sa(x).reshape(batch_size, data_length, -1, self.feature_channel)
        x = x.permute(0,3,2,1).reshape(batch_size, -1, data_length)
       
        x = self.transition(x)
        
        x = self.position_encode(x)
        x = x.permute(0,2,1)
        
        x = self.transformer_block1(x)
        x = self.transformer_block2(x)
        x = x.permute(0,2,1)
        
        x = self.global_ave_pooling(x).squeeze(-1)
        
#        z = x.div(
#            torch.norm(x, p=2, dim=1, keepdim=True).expand_as(x)
#        )

        x = self.linear_for_out(x)
        x = self.activation_for_out(x)
        
        return x 
#        return output, z
    

    def get_optimizer(self):
        optimizer = torch.optim.Adam(self.parameters(), lr = self.initial_learning_rate)
        return optimizer

    def get_loss_function(self):
        ce_loss = nn.CrossEntropyLoss(label_smoothing=0.1)
        def LabelSmoothingCrossEntropy(x, y):
            y = torch.argmax(y, dim=1)
            return ce_loss(x, y)
        return LabelSmoothingCrossEntropy

    def prepare_before_epoch(self, epoch):
        epoch_tau = epoch+1
        tau = max(1 - (epoch_tau - 1) / 50, 0.5)
        for m in self.modules():
            if hasattr(m, '_update_tau'):
                m._update_tau(tau)

