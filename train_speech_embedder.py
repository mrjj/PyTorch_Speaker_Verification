#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  5 21:49:16 2018

@author: harry
"""

import os
import random
import time
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
# from tensorboard_logger import configure, log_value
from tensorboardX import SummaryWriter

from hparam import hparam as hp
from data_load import SpeakerDatasetTIMIT, SpeakerDatasetTIMITPreprocessed
from speech_embedder_net import SpeechEmbedder, GE2ELoss, get_centroids, get_cossim

def train(model_path, tensorboard_writer):
    device = torch.device(hp.device)
    
    if hp.data.data_preprocessed:
        train_dataset = SpeakerDatasetTIMITPreprocessed()
    else:
        train_dataset = SpeakerDatasetTIMIT()
    train_loader = DataLoader(train_dataset, batch_size=hp.train.N, shuffle=True, num_workers=hp.train.num_workers, drop_last=True) 

    if hp.data.data_preprocessed:
        test_dataset = SpeakerDatasetTIMITPreprocessed(test_mode=True)
    else:
        test_dataset = SpeakerDatasetTIMIT(test_mode=True)
    test_loader = DataLoader(test_dataset, batch_size=hp.test.N, shuffle=True, num_workers=hp.test.num_workers, drop_last=True)

    embedder_net = SpeechEmbedder().to(device)
    start_epoch = 0
    if hp.train.restore:
        try:
            start_epoch = int(os.path.basename(model_path).strip().split('_')[2])
        except:
            print('==> Cannot determine start epoch')
        embedder_net.load_state_dict(torch.load(model_path))

    ge2e_loss = GE2ELoss(device)

    #Both net and loss have trainable parameters
    optimizer = torch.optim.SGD([
                    {'params': embedder_net.parameters()},
                    {'params': ge2e_loss.parameters()}
                ], lr=hp.train.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=hp.train.epochs, eta_min=4e-08)
    
    os.makedirs(hp.train.checkpoint_dir, exist_ok=True)
    
    embedder_net.train()
    iteration = 0
    best_EER = float('inf')
    for e in range(start_epoch, hp.train.epochs):
        print('==> epochs', e)

        scheduler.step(e)
        current_lr = optimizer.state_dict()['param_groups'][0]['lr']
        print('==> Learning rate: {lr:.6f}'.format(lr=current_lr))

        # # Note(xin): step decay
        # if e != 0 and e % hp.train.lr_decay == 0:
        #     optim_state = optimizer.state_dict()
        #     optim_state['param_groups'][0]['lr'] = optim_state['param_groups'][0]['lr'] * 0.5
        #     optimizer.load_state_dict(optim_state)
        #     print('Learning rate decayed to: {lr:.6f}'.format(lr=optim_state['param_groups'][0]['lr']))

        tensorboard_writer.add_scalars('learning_rate', {'learning_rate': current_lr}, e + 1)

        total_loss = 0
        for batch_id, mel_db_batch in enumerate(train_loader): 
            mel_db_batch = mel_db_batch.to(device)
            
            mel_db_batch = torch.reshape(mel_db_batch, (hp.train.N*hp.train.M, mel_db_batch.size(2), mel_db_batch.size(3)))
            perm = random.sample(range(0, hp.train.N*hp.train.M), hp.train.N*hp.train.M)
            unperm = list(perm)
            for i,j in enumerate(perm):
                unperm[j] = i
            mel_db_batch = mel_db_batch[perm]
            #gradient accumulates
            optimizer.zero_grad()
            
            embeddings = embedder_net(mel_db_batch)
            embeddings = embeddings[unperm]
            embeddings = torch.reshape(embeddings, (hp.train.N, hp.train.M, embeddings.size(1)))
            
            #get loss, call backward, step optimizer
            loss = ge2e_loss(embeddings) #wants (Speaker, Utterances, embedding)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(embedder_net.parameters(), 3.0)
            torch.nn.utils.clip_grad_norm_(ge2e_loss.parameters(), 1.0)
            optimizer.step()
            
            total_loss = total_loss + loss
            iteration += 1
            if (batch_id + 1) % hp.train.log_interval == 0:
                mesg = "{0}\tEpoch:{1}[{2}/{3}],Iteration:{4}\tLoss:{5:.4f}\tTLoss:{6:.4f}\t\n".format(time.ctime(), e+1,
                        batch_id+1, len(train_dataset)//hp.train.N, iteration,loss, total_loss / (batch_id + 1))
                print(mesg)
                # log_value('loss', loss, iteration)
                # log_value('avg_loss', total_loss / (batch_id + 1), iteration)
                tensorboard_writer.add_scalars(
                    'train', 
                    {
                        'loss': loss,
                        'avg_loss': total_loss / (batch_id + 1)
                    },
                    iteration
                )

                if hp.train.log_file is not None:
                    with open(hp.train.log_file,'a') as f:
                        f.write(mesg)

        embedder_net.eval()
        avg_EER = test(model_path=None, embedder_net=embedder_net, test_loader=test_loader)
        tensorboard_writer.add_scalars('eval', {'avg_eer': avg_EER}, iteration)
                    
        if hp.train.checkpoint_dir is not None and (e + 1) % hp.train.checkpoint_interval == 0:
            # embedder_net.eval()
            # ckpt_model_filename = "ckpt_epoch_" + str(e+1) + "_batch_id_" + str(batch_id+1) + ".pth"
            ckpt_model_filename = "ckpt_epoch_{}_batch_id_{}_avg_eer_{}.pth".format(e+1, batch_id+1, avg_EER)
            ckpt_model_path = os.path.join(hp.train.checkpoint_dir, ckpt_model_filename)
            torch.save(embedder_net.state_dict(), ckpt_model_path)
            # embedder_net.to(device).train()


        if avg_EER < best_EER:
            best_EER = min(avg_EER, best_EER)

            # embedder_net.eval()
            ckpt_model_filename = "best_epoch_{}_batch_id_{}_avg_eer_{}.pth".format(e+1, batch_id+1, avg_EER)
            ckpt_model_path = os.path.join(hp.train.checkpoint_dir, ckpt_model_filename)
            torch.save(embedder_net.state_dict(), ckpt_model_path)
            # embedder_net.to(device).train()

        embedder_net.to(device).train()
        
            # avg_EER = test(model_path)
            # if avg_EER < best_EER:
            #     embedder_net.eval().cpu()
            #     ckpt_model_filename = "best.pth"
            #     ckpt_model_path = os.path.join(hp.train.checkpoint_dir, ckpt_model_filename)
            #     torch.save(embedder_net.state_dict(), ckpt_model_path)
            #     embedder_net.to(device).train()

    #save model
    embedder_net.eval()
    # save_model_filename = "final_epoch_" + str(e + 1) + "_batch_id_" + str(batch_id + 1) + ".model"
    save_model_filename = "final_epoch_{}_batch_id_{}_avg_eer_{}.pth".format(e+1, batch_id+1, avg_EER)
    save_model_path = os.path.join(hp.train.checkpoint_dir, save_model_filename)
    torch.save(embedder_net.state_dict(), save_model_path)
    
    print("\nDone, trained model saved at", save_model_path)

def test(model_path, embedder_net=None, test_loader=None):
    device = torch.device(hp.device)

    if not embedder_net:
        if hp.data.data_preprocessed:
            test_dataset = SpeakerDatasetTIMITPreprocessed(test_mode=True)
        else:
            test_dataset = SpeakerDatasetTIMIT(test_mode=True)
        test_loader = DataLoader(test_dataset, batch_size=hp.test.N, shuffle=True, num_workers=hp.test.num_workers, drop_last=True)

        if not model_path:
            raise Exception('==> Model path needs to be specified')

        embedder_net = SpeechEmbedder().to(device)
        embedder_net.load_state_dict(torch.load(model_path))
        embedder_net.eval()
    
    avg_EER = 0
    for e in range(hp.test.epochs):
        print('==> epochs', e)

        batch_avg_EER = 0
        for batch_id, mel_db_batch in tqdm(enumerate(test_loader)):
            mel_db_batch = mel_db_batch.to(device)

            assert hp.test.M % 2 == 0
            enrollment_batch, verification_batch = torch.split(mel_db_batch, int(mel_db_batch.size(1)/2), dim=1)
            
            enrollment_batch = torch.reshape(enrollment_batch, (int(hp.test.N*hp.test.M/2), enrollment_batch.size(2), enrollment_batch.size(3)))
            verification_batch = torch.reshape(verification_batch, (int(hp.test.N*hp.test.M/2), verification_batch.size(2), verification_batch.size(3)))
            
            perm = random.sample(range(0,verification_batch.size(0)), verification_batch.size(0))
            unperm = list(perm)
            for i,j in enumerate(perm):
                unperm[j] = i

            verification_batch = verification_batch[perm]
            enrollment_embeddings = embedder_net(enrollment_batch)
            verification_embeddings = embedder_net(verification_batch)
            verification_embeddings = verification_embeddings[unperm]
            
            enrollment_embeddings = torch.reshape(enrollment_embeddings, (hp.test.N, int(hp.test.M/2), enrollment_embeddings.size(1)))
            verification_embeddings = torch.reshape(verification_embeddings, (hp.test.N, int(hp.test.M/2), verification_embeddings.size(1)))

            enrollment_centroids = get_centroids(enrollment_embeddings)

            sim_matrix = get_cossim(verification_embeddings, enrollment_centroids)

            # calculating EER
            diff = 1; EER=0; EER_thresh = 0; EER_FAR=0; EER_FRR=0
            
            for thres in [0.01*i+0.5 for i in range(50)]:
                sim_matrix_thresh = sim_matrix>thres
                
                FAR = (sum([sim_matrix_thresh[i].float().sum()-sim_matrix_thresh[i,:,i].float().sum() for i in range(int(hp.test.N))])
                /(hp.test.N-1.0)/(float(hp.test.M/2))/hp.test.N)
    
                FRR = (sum([hp.test.M/2-sim_matrix_thresh[i,:,i].float().sum() for i in range(int(hp.test.N))])
                /(float(hp.test.M/2))/hp.test.N)
                
                # Save threshold when FAR = FRR (=EER)
                if diff> abs(FAR-FRR):
                    diff = abs(FAR-FRR)
                    EER = (FAR+FRR)/2
                    EER_thresh = thres
                    EER_FAR = FAR
                    EER_FRR = FRR
            batch_avg_EER += EER
            print("\nEER : %0.2f (thres:%0.2f, FAR:%0.2f, FRR:%0.2f)"%(EER,EER_thresh,EER_FAR,EER_FRR))
        avg_EER += batch_avg_EER/(batch_id+1)
    avg_EER = avg_EER / hp.test.epochs
    print("\n EER across {0} epochs: {1:.4f}".format(hp.test.epochs, avg_EER))
    return avg_EER
        
if __name__=="__main__":
    # configure(hp.train.checkpoint_dir, flush_secs=5)
    tensorboard_writer = SummaryWriter(hp.train.checkpoint_dir)

    if hp.training:
        train(hp.model.model_path, tensorboard_writer)
    else:
        test(hp.model.model_path)