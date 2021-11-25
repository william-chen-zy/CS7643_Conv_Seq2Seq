from __future__ import print_function
from __future__ import absolute_import

import torch
from torch import nn
import util

class Encoder(nn.Module):
    def __init__(self,
                 nfilters,
                #  re_term,  # for weights_regularizer
                 enc_dim=527,
                 enc_dim_desc={'hidden_num':512, 'class_num': 15},
                 enc_shape=[None, 49, 54, 1], 
                 kernel_size = (2, 7), 
                 stride=(2, 1)):
        super(Encoder, self).__init__()
        self.nfilters = nfilters
        self.input_h = enc_shape[1]
        self.input_w = enc_shape[2]
        self.kernel_size = kernel_size
        self.stride = stride
        
        self.conv1 = nn.Conv2d(in_channels=1, 
                               out_channels=self.nfilters * 4,
                               kernel_size=kernel_size,
                               stride=stride,
                               #weights_regularizer=tcl.l2_regularizer(self.re_term), #specific to tf?
                               )
        self.batch_norm1 = nn.BatchNorm2d(num_features=self.nfilters * 4)
        self.relu1 = nn.LeakyReLU(negative_slope=0.2)
        
        self.conv2 = nn.Conv2d(in_channels=self.nfilters * 4, 
                               out_channels=self.nfilters * 4,
                               kernel_size=kernel_size,
                               stride=stride,
                               #weights_regularizer=tcl.l2_regularizer(self.re_term), #specific to tf?
                               )
        self.batch_norm2 = nn.BatchNorm2d(num_features=self.nfilters * 4)
        self.relu2 = nn.LeakyReLU(negative_slope=0.2)
        
        self.dropout = nn.Dropout(p=0.8)
        
        self.conv3 = nn.Conv2d(in_channels=self.nfilters * 4, 
                               out_channels=self.nfilters * 8,
                               kernel_size=kernel_size,
                               stride=stride,
                               #weights_regularizer=tcl.l2_regularizer(self.re_term), #specific to tf?
                               )
        self.batch_norm3 = nn.BatchNorm2d(num_features=self.nfilters * 8)
        self.relu3 = nn.LeakyReLU(negative_slope=0.2)
        
        self.fc = nn.Linear(self.input_h*self.input_w*self.nfilters * 8, enc_dim)
    
    def forward(self, encoder_inputs):
        
        N, H, W, C = encoder_inputs.shape
        encoder_inputs = encoder_inputs.view(N, C, H, W)

        padded_input = util.get_padding(encoder_inputs.squeeze(), 
                                         (H,W), 
                                         stride=self.stride, 
                                         kernel_size=self.kernel_size)
        padded_input = padded_input.unsqueeze(dim=1)
        out = self.conv1(padded_input)
        out = self.batch_norm1(out)
        out = self.relu1(out)
        
        padded_out = util.get_padding(out, 
                                      (H,W), 
                                      stride=self.stride, 
                                      kernel_size=self.kernel_size)
        out = self.conv2(padded_out)
        out = self.batch_norm2(out)
        out = self.relu2(out)
        
        out = self.dropout(out)
        padded_out = util.get_padding(out, 
                                      (H,W), 
                                      stride=self.stride, 
                                      kernel_size=self.kernel_size)
        out = self.conv3(padded_out)
        out = self.batch_norm3(out)
        out = self.relu3(out)

        out = torch.flatten(out, start_dim=1)
        out = self.fc(out)
        
        return out
    