import torch
from torch import nn
from Encoder import Encoder

class Decoder(nn.Module):
    def __init__(self,nfilters,
                 #re_term,
                 input_dim, encoded_dim,encoded_desc,name_scope,weight_decay):
        super(Decoder, self).__init__()
        self.input_dim = input_dim
        self.encoded_dim = encoded_dim
        self.encoded_desc = encoded_desc
        self.name_scope = name_scope
        self.weight_decay = weight_decay
        self.nfilters = nfilters
        # self.re_term = re_term

        self.relu = nn.LeakyReLU(negative_slope=0.2)
        self.encoder = Encoder(self.nfilters,
                               enc_shape=[None, 20, 70, 1],
                               enc_dim=512,
                               enc_dim_desc={'hidden_num': 512}) # Todo: Param tuning
        self.drop = nn.Dropout(p=0.5)

    def forward(self, decoder_hidden, dec_in):
        dec_in_enc = self.encoder.forward(dec_in)

        y = torch.cat((decoder_hidden, dec_in_enc), 1)

        in_dim = y.shape[-1]
        linear1 = nn.Linear(in_dim, 512)
        out = self.relu(linear1(y))

        out = self.drop(out)
        linear2 = nn.Linear(out.shape[-1], 70)
        h0 = self.relu(linear2(y))

        h0 = torch.unsqueeze(torch.unsqueeze(h0, 1), 3)

        return h0

    