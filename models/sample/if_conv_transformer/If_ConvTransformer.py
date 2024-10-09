import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import math
import pandas as pd
import time
from utils.utils import *
import os
from torch.nn.utils import weight_norm
from contiguous_params import ContiguousParams


def get_config(dataset, lr_magnif=1):

    return {'input_2Dfeature_channel': 1, 
            'feature_channel': 64,
            'kernel_size': 5,
            'kernel_size_grav': 3, 
            'scale_num': 2,
            'feature_channel_out': 128,
            'multiheads': 1,
            'drop_rate': 0.2,
            'learning_rate': 0.001 * lr_magnif,
            'regularization_rate': 0.01}

def gen_model(input_shape, n_classes, out_loss, out_activ, metrics, config):
    """
        input_shape is [time-series, sensors(=channel)]
    """
    return If_ConvTransformer(input_2Dfeature_channel=1, input_channel=input_shape[1], data_length=input_shape[0], nb_classes=n_classes, **config)


def gen_preconfiged_model(input_shape, n_classes, out_loss, out_activ, dataset, metrics=['accuracy'], lr_magnif=1):
    config = get_config(dataset, lr_magnif)
    return gen_model(input_shape, n_classes, out_loss, out_activ, metrics, config), config


def get_optim_config(dataset, trial, lr_magnif=1):
    raise NotImplementedError("No config for optimization")


def get_dnn_framework_name():
    return 'pytorch'


from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
)

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
        # set dropout rate
        self.dropout_attention = nn.Dropout(drop_rate)
        # squeeze to k dimentions through linear transformation
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

class TransformerBlock(nn.Module):
    def __init__(self, k, heads, drop_rate):
        super(TransformerBlock, self).__init__()

        self.attention = SelfAttention(k, heads = heads, drop_rate = drop_rate)
        self.norm1 = nn.LayerNorm(k)

        self.mlp = nn.Sequential(
            nn.Linear(k, 4*k),
            nn.ReLU(),
            nn.Linear(4*k, k)
        )
        self.norm2 = nn.LayerNorm(k)
        self.dropout_forward = nn.Dropout(drop_rate)

    def forward(self, x):
        
        # perform self-attention
        attended = self.attention(x)
        # perform layer norm
        x = self.norm1(attended + x)
        # feedforward and layer norm
        feedforward = self.mlp(x)
        
        return self.dropout_forward(self.norm2(feedforward + x))

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

class If_ConvTransformer(nn.Module):
    def __init__(self, input_2Dfeature_channel, input_channel, feature_channel,
                 kernel_size, kernel_size_grav, scale_num, feature_channel_out,
                 multiheads, drop_rate, data_length, num_class):
        
        super(If_ConvTransformer, self).__init__()
        
        self.IMU_fusion_block = IMU_Fusion_Block(input_2Dfeature_channel, input_channel, feature_channel,
                                                 kernel_size_grav, scale_num)
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(feature_channel, feature_channel, (1,kernel_size), 1, (0,kernel_size//2)),
            nn.BatchNorm2d(feature_channel),
            nn.ReLU(),
            # nn.MaxPool2d(2)
            )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(feature_channel, feature_channel, (1,kernel_size), 1, (0,kernel_size//2)),
            nn.BatchNorm2d(feature_channel),
            nn.ReLU(),
            # nn.MaxPool2d(2)
            )
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(feature_channel, feature_channel, (1,kernel_size), 1, (0,kernel_size//2)),
            nn.BatchNorm2d(feature_channel),
            nn.ReLU(),
            # nn.MaxPool2d(2)
            )
       
        # TODO check: what is it?
        # to half?
        if input_channel  == 12:
            reduced_channel = 6
        else:
            reduced_channel = 3
        
        self.transition = nn.Sequential(
            nn.Conv1d(feature_channel*(input_channel-reduced_channel)*scale_num, feature_channel_out, 1, 1),
            nn.BatchNorm1d(feature_channel_out),
            nn.ReLU()
            )
        
        self.position_encode = PositionalEncoding(feature_channel_out, drop_rate, data_length)
        
        self.transformer_block1 = TransformerBlock(feature_channel_out, multiheads, drop_rate)
        
        self.transformer_block2 = TransformerBlock(feature_channel_out, multiheads, drop_rate)
        
        self.global_ave_pooling = nn.AdaptiveAvgPool1d(1)
        
        self.linear = nn.Linear(feature_channel_out, num_class)

    def forward(self, x):
        
        # hidden = None
        batch_size = x.shape[0]
        feature_channel = x.shape[1]
        input_channel = x.shape[2]
        data_length = x.shape[-1]

        # The input is [B, time-series, sensors]
        # This model needs [B, C(=1), sensors, time-series]
    
        x = torch.unsqueeze(x, 1) # add channel
        x = torch.transpose(x, 3, 2) # swap 

        x, out_attn = self.IMU_fusion_block(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        
        x = x.view(batch_size, -1, data_length)
        x = self.transition(x)
        
        x = self.position_encode(x)
        x = x.permute(0,2,1)
        
        x = self.transformer_block1(x)
        x = self.transformer_block2(x)
        x = x.permute(0,2,1)
        
        x = self.global_ave_pooling(x).squeeze()
        
        output = self.linear(x)
        
        return output, out_attn
    
