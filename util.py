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
          G_criterion, 
          D_criterion, 
          optimizerG, 
          optimizerD,
          device='cuda',
          batch = 16,
          lr=5e-5,
          lr_decay_steps = 10000,
          lr_decay = 0.99,
          L2_lambda = 0.001,
          iterations = 20000,
          display = 100,
          model_name='default_setting'):
    
    if os.path.exists('logs/' + model_name):
        shutil.rmtree('logs/' + model_name)
    writer = SummaryWriter('logs/' + model_name)
    
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
        
        D_LOSS += [float(loss_discriminator)]
        G_LOSS += [float(loss_generator)]
        
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
                     device=device, tb_writer=writer)

        if (i > 0) and (i % 2000 == 0):
            torch.save(generator.state_dict(), 'model/generator_{}_{}_steps.pt'.format(model_name, i))
            torch.save(discriminator.state_dict(), 'model/discriminator_{}_{}_steps.pt'.format(model_name, i))
        
def test(encoder_data, 
         discriminator_data,
         action, 
         generator, 
         global_step=None,
         device='cuda',
         tb_writer=None):
    
    G_criterion = nn.MSELoss()

    encoder_data = torch.from_numpy(encoder_data).float().to(device)
    discriminator_data = torch.from_numpy(discriminator_data).float().to(device)

    expected_seq = discriminator_data[:, 50:, :, :]

    predicted_seq, predicted_action, generated_sample = generator.forward(encoder_data, discriminator_data)
    ReconstructError = G_criterion(predicted_seq, expected_seq)
    tb_writer.add_scalar('TestErrors/'+action,ReconstructError, global_step)

def InferenceSample(dloader, 
                    generator, 
                    iter=20000, 
                    one_hot=False, 
                    device='cuda', 
                    model_name=None):
             
        one_hot=one_hot
        srnn_gts_expmap = dloader.get_srnn_gts(one_hot, to_euler=False)
        srnn_gts_euler=dloader.get_srnn_gts(one_hot, to_euler=True)

        SAMPLES_FNAME = "./samples/{}_{}_steps.h5".format(model_name,iter)

        # try:
        #     os.remove(SAMPLES_FNAME)
        # except OSError:
        #     pass
        
        step_time=[]
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
            dloader.compute_test_error(action, predicted_seq.to('cpu').detach().numpy(), srnn_gts_expmap, srnn_gts_euler, one_hot, SAMPLES_FNAME)
        print (np.mean(step_time))