import torch
from torch import nn

class Decoder(nn.Module):
    def __init__(self,
                 st_encoder,
                #  re_term,  # for weights_regularizer
                 out_dim=54,
                 enc_dim_desc={'hidden_num':512, 'class_num': 15}
                 ):
        super().__init__()
        self.st_encoder = st_encoder
        self.fc1 = nn.Linear(enc_dim_desc['hidden_num']*2, enc_dim_desc['hidden_num'])
        self.fc2 = nn.Linear(enc_dim_desc['hidden_num'], out_dim)
        self.relu = nn.LeakyReLU()
        
    def forward(self, encoder_hidden, decoder_inputs):
        
        st_hidden = self.st_encoder(decoder_inputs)
        
        out = torch.cat([encoder_hidden, st_hidden],dim=1)
        
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        
        return out.unsqueeze(dim=1).unsqueeze(dim=3)