import os
from itertools import cycle

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from dataloader import CustomDataset
from networks import SRModel
from torchvision.utils import save_image, make_grid

from datetime import datetime
from tensorboardX import SummaryWriter

from embiggen import *


class Solver(object):
    def __init__(self, FLAGS):
        self.use_cuda = FLAGS.cuda and torch.cuda.is_available()
        self.gpu = FLAGS.gpu
        
        self.train = FLAGS.train
        self.batch_size = FLAGS.batch_size
        self.data_path = FLAGS.data_path
        
        self.learning_rate = FLAGS.initial_learning_rate
        self.beta1 = FLAGS.beta_1
        self.beta2 = FLAGS.beta_2
        
        self.load_path = FLAGS.load_path

        self.epoch = 0
        self.end_epoch = FLAGS.end_epoch
        
        
        self.model = SRModel().double()
        
        if not self.train:
            if self.use_cuda:
                self.model.load_state_dict(torch.load(os.path.join('checkpoints', self.load_path)))
            else:
                self.model.load_state_dict(torch.load(os.path.join('checkpoints', self.load_path), map_location='cpu'))
        
        if self.use_cuda:
            torch.cuda.set_device(self.gpu)
            self.model.cuda()

        self.optim = optim.Adam(list(self.model.parameters()),lr=self.learning_rate, betas=(self.beta1, self.beta2))
        
        
        date_time = datetime.now().strftime("%b%d_%H-%M-%S")
        self.save_folder = os.path.join('checkpoints', date_time)
        
        if (not os.path.exists(self.save_folder)) and (self.train == True):
            os.makedirs(self.save_folder, exist_ok=True)
        
        if self.train:
            self.writer = SummaryWriter()
            settings = ''
            for arg in vars(FLAGS):
                settings += str(arg)+'='+str(getattr(FLAGS, arg))+'    '
            self.writer.add_text('Settings',settings)
        
        
        print("Loading data...")
        self.train_set = CustomDataset(root=self.data_path,train=True)
#        self.test_set = CustomDataset(root=self.data_path,train=False)
        self.loader = cycle(DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True, num_workers=0, drop_last=True))
        
    
    def Train(self):
        
        X = torch.DoubleTensor(self.batch_size, 2, 128, 128)  #Stores input batch
        hr = torch.DoubleTensor(self.batch_size, 1, 384, 384) #Stores ground truth batch
        if self.use_cuda: 
            X = X.cuda()
            hr = hr.cuda()
        
        for self.epoch in range(self.end_epoch):
            print('')
            print('Epoch #' + str(self.epoch) + '..................................................................')
            
            for iteration in range(int(len(self.train_set)/self.batch_size)):
                
                batch_lr, batch_hr, batch_paths = next(self.loader)
                
                self.optim.zero_grad()
                
                X.copy_(torch.DoubleTensor(batch_lr))
                hr.copy_(torch.DoubleTensor(batch_hr).unsqueeze(1))
                
                sr = self.model(X)
                
                loss = torch.sum((sr - hr).pow(2)) / sr.data.nelement()
                loss.backward()
                self.optim.step()
                
                score = score_images(sr.data.squeeze().clamp(0,1).cpu().numpy(), batch_paths)
                
                
                #Display steps
                if (iteration + 1) % 10 == 0:
                        print('')
                        print('Epoch #' + str(self.epoch))
                        print('Iteration #' + str(iteration))
                        print('')
                        print('Reconstruction loss: ' + str(loss.data.storage().tolist()[0]))
                        print('Score : ' + str(score))
                        print('..........')
                    
                if iteration == 0:
                    with torch.no_grad():
                        self.Reconstruction(hr, sr, path='./images/reconstruction/'+str(self.epoch)+'.png')

                iterations = self.epoch * int(len(self.train_set) / self.batch_size) + iteration
                self.writer.add_scalar('Loss', loss.data.storage().tolist()[0], iterations)
                self.writer.add_scalar('Batch score', score, iterations)
            
            
            # save model every epochs
            torch.save(self.model.state_dict(), os.path.join(self.save_folder, 'model'+str(self.epoch)))
                
                
    #Evaluate the loaded model on the whole training set (as test set ground truth is not available)
    def EvaluateScore(self):    
        self.model.eval()
        
        all_scores = []
        
        for lr, hr, path in tqdm(self.train_set):
            lr = torch.DoubleTensor(lr).unsqueeze(0)
            if self.use_cuda: lr = lr.cuda()
            
            sr = self.model(lr)
            sr = sr.data.squeeze().clamp(0,1).cpu().numpy()
            all_scores.append((score_image_fast(sr, path), path))
        
        S = 0
        for (s,_) in all_scores:
            S += s
        S /= len(all_scores)
        
        print("Average score over whole train set: " + str(S))
            
        
    #Generates a grid with Super Resolved and High Resolution images side by side to visualize training
    def Reconstruction(self, hr, sr, path='./exemples/reconstruction.png', nrow=4):
        recmatrix = torch.FloatTensor(12, 1, 384, 384)
        for i in range(12):
            if i%2 == 0:
                recmatrix[i] = hr[i//2]
            else:
                recmatrix[i] = sr[i//2]
        save_image(recmatrix, path, nrow=nrow)
        self.writer.add_image('Reconstructions', make_grid(recmatrix, nrow=nrow), self.epoch)