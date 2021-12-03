import torch
from torch import nn
from Encoder import Encoder

class Discriminator(nn.Module):
    def __init__(self,nfilters):
        super(Discriminator, self).__init__()
        # pass
        self.nfilters = nfilters
        self.encoder = Encoder(nfilters,
                                 enc_shape=[None, 75, 54, 1],
                                 enc_dim=512,
                                 enc_dim_desc={'hidden_num': 512})
        self.relu = nn.ReLU()
    
    def forward(self, data_input,input_class):
        dec_in_enc = self.encoder.forward(data_input)

        dec_in_enc = self.relu(dec_in_enc)

        y = torch.cat((dec_in_enc, input_class), 1)
        in_dim = y.shape[-1]
        linear1 = nn.Linear(in_dim, self.nfilters * 8)
        h0 = linear1(y)
        linear2 = nn.Linear(h0.shape[-1], 1)
        out = linear2(h0)
        return out

    