from typing import ForwardRef
import torch
from torch import nn

class ConvSeq2Seq(nn.Module):
    
    def __init__(self, encoder, decoder, window_length, device, concat_input_output=True):
        super().__init__()
        self.device = device
        self.encoder = encoder.to(device)
        self.decoder = decoder.to(device)
        self.window_length = window_length
        self.length_input = 50
        self.length_output = 25
        self.sampling = 0.95
        self.concat_input_output = concat_input_output
        
        self.enc_hidden_num = self.encoder.enc_dim_desc['hidden_num']
        try:
            self.enc_class_num = self.encoder.enc_dim_desc['class_num']
        except:
            self.encoder.enc_dim_desc['class_num'] = 0 # test 
            self.enc_class_num = 0
        
        # self.softmax = nn.Softmax(dim=1)
    
    def forward(self, encoder_data, discriminator_data):
        hlayer, hlogits = self.encoder.forward(encoder_data)
        # if self.enc_class_num != 0:
        #     # print("encoder_output: ", encoder_output.shape)
        #     hlayer, hlogits = torch.split(encoder_output,
        #                                [self.enc_hidden_num, self.enc_class_num],
        #                                dim=1)
        #     hlogits = self.softmax(hlogits)
        # else:
        #     hlayer = encoder_output
    
        predicted_res = []
        windowLength = self.window_length
        seqStart = self.length_input - windowLength

        
        dec_in0 = discriminator_data[:, seqStart:(windowLength + seqStart), :, :]
        for it in range(self.length_output):    # sampling from both predicted frame and groundtruth frame during training
            if self.sampling == 0:
                dec_in0 = discriminator_data[:, (seqStart+it):(windowLength + it +seqStart), :, :]
            dec_out = self.decoder.forward(hlayer, dec_in0)
            
            last_input = torch.split(dec_in0, [windowLength - 1, 1], dim=1)
            final_out = dec_out + last_input[1]  
            if self.sampling > 0:
                new_gt = discriminator_data[:, self.length_input + it + 1 : self.length_input + it + 2, :, :]
                dec_in0 = torch.split(dec_in0, [1, windowLength - 1], dim=1)
                dec_in0 = torch.cat([dec_in0[1], self.sampling * final_out + (1 - self.sampling) * new_gt], axis=1)

            predicted_res += [final_out]
        
        predicted_res = torch.cat(predicted_res, axis=1)
        
        if self.concat_input_output:
                generated_sample = torch.cat(
                    [encoder_data, discriminator_data[:, 50:51, :, :], predicted_res], axis=1)
        else:
            generated_sample = predicted_res
        return predicted_res, hlogits, generated_sample