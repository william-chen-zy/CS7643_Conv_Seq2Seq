import torch
from torch import nn
from torch.nn.modules.activation import ReLU
from torch.nn.modules.linear import Linear

class Discriminator(nn.Module):
    def __init__(self, 
                 nfilters, 
                 d_encoder,
                 enc_dim_desc={'hidden_num':512, 'class_num': 15}
                 ):
        super().__init__()
        self.nfilters = nfilters
        self.d_encoder = d_encoder
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(enc_dim_desc['hidden_num']+enc_dim_desc['class_num'], nfilters*8)
        self.fc2 = nn.Linear(nfilters*8, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, seq_data, class_data):
        
        out,_ = self.d_encoder.forward(seq_data)
        out = self.relu(out)
        out = torch.cat([out, class_data], dim=1)
        # print('out: ',out.device)
        # print(out)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        
        return out