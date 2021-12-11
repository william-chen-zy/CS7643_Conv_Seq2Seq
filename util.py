import torch
from torch import nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from math import floor, ceil
import timeit
import os
import shutil
import h5py
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials

import VAE.DataLoader as DataLoader
import VAE.Encoder as Encoder
import VAE.Decoder as Decoder
import VAE.ConvSeq2Seq as ConvSeq2Seq
import VAE.Discriminator as Discriminator

# def get_padding(input_tensor, input_size: tuple, stride: tuple, kernel_size: tuple, padding='same', pad_mode='constant', c=0):
#     h_old, w_old = input_size
#     s_h, s_w = stride
#     k_h, k_w = kernel_size
#     if padding == 'same':
#         h_new, w_new = h_old, w_old
#         p_h = ((h_old - 1) * s_h - h_old + k_h)/2
#         p_w = ((w_old - 1) * s_w - w_old + k_w)/2
#     return nn.functional.pad(input_tensor,
#                              (floor(p_w), ceil(p_w),
#                               floor(p_h), ceil(p_h)),
#                              pad_mode, c)
    
def get_padding(input_tensor, input_size: tuple, stride: tuple, kernel_size: tuple, padding='same', pad_mode='constant', c=0):
    h_old, w_old = input_size
    s_h, s_w = stride
    k_h, k_w = kernel_size
    if padding == 'same':
        h_new = ceil(h_old/s_h)
        w_new  = ceil(w_old/s_w)

        if (h_old % s_h == 0):
            pad_along_height = max(k_h - s_h, 0)
        else:
            pad_along_height = max(k_h - (h_old % s_h), 0)
        if (w_old % s_w == 0):
            pad_along_width = max(k_w - s_w, 0)
        else:
            pad_along_width = max(k_w - (w_old % s_w), 0)
        
        pad_top = pad_along_height // 2
        pad_bottom = pad_along_height - pad_top
        pad_left = pad_along_width // 2
        pad_right = pad_along_width - pad_left

    return nn.functional.pad(input_tensor,
                             (pad_left, pad_right,
                              pad_top, pad_bottom),
                             pad_mode, c)
    
def train(dloader, 
          generator, 
          discriminator, 
        #   G_criterion, 
        #   D_criterion, 
        #   optimizerG, 
        #   optimizerD,
          device='cuda',
          batch = 16,
          lr=5e-5,
          lr_decay_steps = 10000,
          lr_decay = 0.99,
          L2_lambda = 0.001,
          iterations = 20000,
          display = 100,
          tensorboard=True,
          save_model=True,
          model_name='default_setting'):
    
    if os.path.exists('logs/' + model_name):
        shutil.rmtree('logs/' + model_name)
    if tensorboard:
        writer = SummaryWriter('logs/' + model_name)
    else:
        writer = None
    
    start=timeit.default_timer()
    
    G_criterion = nn.MSELoss()
    D_criterion = nn.BCEWithLogitsLoss()
    
    # Setup Adam optimizers for both G and D
    optimizerD = optim.Adam(discriminator.parameters(), lr=lr, weight_decay=L2_lambda)
    optimizerG = optim.Adam(generator.parameters(), lr=lr, weight_decay=L2_lambda)
    
    D_LOSS = []
    G_LOSS = []
    d_loss_in_training =[]
    g_loss_in_training =[]
    
    # previous_loss_discriminator = 9999
    # previous_loss_generator = 9999
    
    for i in range(iterations):
        generator.train()
        discriminator.train()
        
        # print("****************train discriminator*****************")
        encoder_data, discriminator_data, yhat = dloader.get_train_batch(batch)
        encoder_data = torch.from_numpy(encoder_data).float().to(device)
        discriminator_data = torch.from_numpy(discriminator_data).float().to(device) 
        yhat = torch.from_numpy(yhat).float().to(device)

        for param in discriminator.parameters():
            param.grad = None

        d_logits_real = discriminator.forward(discriminator_data, yhat)
        d_loss_real = D_criterion(d_logits_real, torch.ones_like(d_logits_real))

        d_loss_real.backward()

        predicted_seq, predicted_action, generated_sample = generator.forward(encoder_data, discriminator_data)
        d_logits_fake = discriminator.forward(generated_sample, yhat)
        d_loss_fake = D_criterion(d_logits_fake, torch.zeros_like(d_logits_fake))

        d_loss_fake.backward()

        loss_discriminator = d_loss_real + d_loss_fake

        optimizerD.step()
        
        # print("****************train generator*****************")
        encoder_data, discriminator_data, yhat = dloader.get_train_batch(batch)
        encoder_data = torch.from_numpy(encoder_data).float().to(device)
        discriminator_data = torch.from_numpy(discriminator_data).float().to(device)
        yhat = torch.from_numpy(yhat).float().to(device)

        expected_seq = discriminator_data[:, 50:, :, :]

        for param in generator.parameters():
            param.grad = None

        predicted_seq, predicted_action, generated_sample = generator.forward(encoder_data, discriminator_data)
        ReconstructError = G_criterion(predicted_seq, expected_seq)

        ReconstructError.backward(retain_graph=True)

        d_logits_fake = discriminator.forward(generated_sample, yhat)
        g_loss = D_criterion(d_logits_fake, torch.ones_like(d_logits_fake)) * torch.tensor(0.01)
        
        g_loss.backward()

        loss_generator = ReconstructError + g_loss
        
        optimizerG.step()
        
        # if (float(loss_discriminator)>previous_loss_discriminator*5) or (float(loss_generator)>previous_loss_generator*5):
        #     break
        # else:
        D_LOSS += [float(loss_discriminator)]
        G_LOSS += [float(loss_generator)]
            # previous_loss_discriminator = float(loss_discriminator)
            # previous_loss_generator = float(loss_generator)
        
        if tensorboard:
            writer.add_scalars('Train',{'Discriminator':loss_discriminator, 'Generator': loss_generator}, global_step=i)
            writer.add_scalar('Train/Learning rate', lr, global_step=i)

        if (i >= lr_decay_steps) and (i % lr_decay_steps == 0):
            lr = lr * lr_decay
            for g in optimizerG.param_groups:
                g['lr'] = lr
            for g in optimizerD.param_groups:
                g['lr'] = lr
                
        # Output training stats    
        if i%display==0:
            time_elasped = timeit.default_timer() - start
            print('Iterations %d loss_d %f, loss_g %f, lr %f, time %f' % (i, np.mean(D_LOSS), 
                                                                          np.mean(G_LOSS), 
                                                                          lr, time_elasped))
            D_LOSS=[]
            G_LOSS=[]
            for action in dloader.actions:
                encoder_data, discriminator_data, yhat = dloader.get_test_batch(action)
                test(encoder_data, discriminator_data, action, generator, global_step=i, 
                     device=device, tensorboard=tensorboard, tb_writer=writer)
        if save_model:
            if (i > 0) and (i % 2000 == 0):
                torch.save(generator.state_dict(), 'model/generator_{}_{}_steps.pt'.format(model_name, i))
                torch.save(discriminator.state_dict(), 'model/discriminator_{}_{}_steps.pt'.format(model_name, i))
        
def test(encoder_data, 
         discriminator_data,
         action, 
         generator, 
         global_step=None,
         device='cuda',
         tensorboard=True,
         tb_writer=None):
    
    G_criterion = nn.MSELoss()

    encoder_data = torch.from_numpy(encoder_data).float().to(device)
    discriminator_data = torch.from_numpy(discriminator_data).float().to(device)

    expected_seq = discriminator_data[:, 50:, :, :]

    predicted_seq, predicted_action, generated_sample = generator.forward(encoder_data, discriminator_data)
    ReconstructError = G_criterion(predicted_seq, expected_seq)
    if tensorboard:
        tb_writer.add_scalar('TestErrors/'+action,ReconstructError, global_step)

def InferenceSample(dloader, 
                    generator, 
                    iter=20000, 
                    one_hot=False, 
                    device='cuda', 
                    model_name=None,
                    model_tuning=False):
             
        one_hot=one_hot
        srnn_gts_expmap = dloader.get_srnn_gts(one_hot, to_euler=False)
        srnn_gts_euler=dloader.get_srnn_gts(one_hot, to_euler=True)

        SAMPLES_FNAME = "./samples/{}_{}_steps.h5".format(model_name,iter)

        # try:
        #     os.remove(SAMPLES_FNAME)
        # except OSError:
        #     pass
        
        step_time = []
        accumulated_error = 0
        for action in dloader.actions:

            start_time = timeit.default_timer()
            encoder_input,decoder_expect_output, _ = dloader.get_test_batch(action)
            encoder_input = torch.from_numpy(encoder_input).float().to(device)
            decoder_expect_output = torch.from_numpy(decoder_expect_output).float().to(device)

            predicted_seq, predicted_action, generated_sample = generator.forward(encoder_input, decoder_expect_output)
            
            time=timeit.default_timer()-start_time
            step_time.append(time)
            # print('generated_sample: ', generated_sample.shape)
            # print('predicted_seq: ', predicted_seq.shape)
            # print('srnn_gts_expmap: ', len(srnn_gts_expmap[action]))
            # print('srnn_gts_euler: ', len(srnn_gts_euler[action]))
            action_error = dloader.compute_test_error(action, predicted_seq.to('cpu').detach().numpy(), 
                                                      srnn_gts_expmap, srnn_gts_euler, one_hot, SAMPLES_FNAME, 
                                                      model_tuning=model_tuning)
            accumulated_error += sum(action_error)
        if model_tuning:
            return accumulated_error
        else:
            print (np.mean(step_time))
            


def optimize(dloader, params):

    lr = params['lr']
    L2_lambda = params['L2_lambda']
    lt_encoder_filters = params['lt_encoder_filters']
    st_encoder_filters = params['st_encoder_filters']
    d_encoder_filters = params['d_encoder_filters']
    discriminator_output_filters = params['discriminator_output_filters']
    kernel_height = params['kernel_height']
    kernel_width = params['kernel_width']
    stride_vert = params['stride_vert']
    stride_hori = params['stride_hori']
    
    device = params['device']
    
    textstr = '\n'.join((
                        r'='*60,
                        r'Using device: {}'.format(device),
                        r'Training on following parameters: ',
                        r'lr=%.8f' % (lr, ),
                        r'L2_lambda=%.2f' % (L2_lambda, ),
                        r'lt_encoder_filters size=%.2f' % (lt_encoder_filters, ),
                        r'st_encoder_filters size=%.2f' % (st_encoder_filters, ),
                        r'd_encoder_filters size=%.2f' % (d_encoder_filters, ),
                        r'discriminator_output_filters size=%.2f' % (discriminator_output_filters, ),
                        r'kernel size=({},{})'.format(kernel_height, kernel_width),
                        r'stride=({},{})'.format(stride_hori, stride_vert)))
    print(textstr)
    

    lt_encoder = Encoder.Encoder(lt_encoder_filters,enc_shape=[None, 49, 54, 1], 
                                 enc_dim_desc={ 'hidden_num': 512,'class_num': 15 }, 
                                 stride=(stride_hori,stride_vert), 
                                 kernel_size=(kernel_height, kernel_width))

    st_encoder = Encoder.Encoder(st_encoder_filters,enc_shape=[None, 20, 54, 1], 
                                 enc_dim_desc={ 'hidden_num': 512}, 
                                 stride=(stride_hori,stride_vert), 
                                 kernel_size=(kernel_height, kernel_width))
    decoder = Decoder.Decoder(st_encoder)

    generator = ConvSeq2Seq.ConvSeq2Seq(lt_encoder, decoder, window_length=20, device=device)

    d_encoder = Encoder.Encoder(d_encoder_filters, enc_shape=[None, 75,54,1], 
                                enc_dim_desc={ 'hidden_num': 512}, 
                                stride=(stride_hori,stride_vert), 
                                kernel_size=(kernel_height, kernel_width))
    discriminator = Discriminator.Discriminator(discriminator_output_filters, d_encoder).to(device)


    # # create loss function for both G and D
    # G_criterion = nn.MSELoss()
    # D_criterion = nn.BCEWithLogitsLoss()
    # # Setup Adam optimizers for both G and D
    # optimizerD = optim.Adam(discriminator.parameters(), lr=lr, weight_decay=0.001)
    # optimizerG = optim.Adam(generator.parameters(), lr=lr, weight_decay=0.001)
    
    
    train(dloader, 
          generator, 
          discriminator, 
        #   G_criterion, 
        #   D_criterion, 
        #   optimizerG, 
        #   optimizerD,
          device=device,
          batch = 16,
          lr=lr,
          lr_decay_steps = 10000,
          lr_decay = 0.99,
          L2_lambda = L2_lambda,
          iterations = 1000,
          display = 100,
          tensorboard=False,
          save_model=False,
          model_name='default_setting_1000_itr')
    
    error = InferenceSample(dloader, generator, model_name='default_setting_1000_itr', model_tuning=True)
    del dloader
    del generator
    del discriminator
    torch.cuda.empty_cache()
    print('Test error=%.5f' % (error, ))
    return {'loss': error, 
            'status': STATUS_OK, 
            'params': params}