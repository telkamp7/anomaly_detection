import numpy as np
import pandas as pd

import torch
from torch import nn, Tensor
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import lightning as L

from argparse import Namespace

class TSDataset(Dataset):
    def __init__(self, split: str, dataset_root: str = "./", cont_vars=None, cat_vars=None, lbl_as_feat=True, ):
        """
        split: 'train' if we want to get data from the training examples, 'test' for
        test examples, or 'both' to merge the training and test sets and return samples
        from either.
        cont_vars: List of continuous variables to return as features. If None, returns
        all continuous variables available.
        cat_vars: Same as above, but for categorical variables.
        lbl_as_feat: Set to True when training a VAE -- the labels (temperature values)
        will be included as another dimension of the data. Set to False when training
        a model to predict temperatures.
        """
        super().__init__()
        assert split in ['train', 'test', 'both']
        self.lbl_as_feat = lbl_as_feat
        if split == 'train':
            self.df = pd.read_csv(dataset_root/'train.csv')
        elif split == 'test':
            self.df = pd.read_csv(dataset_root/'test.csv')
        else:
            df1 = pd.read_csv(dataset_root/'train.csv')
            df2 = pd.read_csv(dataset_root/'test.csv')
            self.df = pd.concat((df1, df2), ignore_index=True)
        
        # Select continuous variables to use
        if cont_vars:
            self.cont_vars = cont_vars
            # If we want to use 'value' as a feature, ensure it is returned
            if self.lbl_as_feat:
                try:
                    assert 'value' in self.cont_vars
                except AssertionError:
                    self.cont_vars.insert(0, 'value')
            # If not, ensure it not returned as a feature
            else:
                try:
                    assert 'value' not in self.cont_vars
                except AssertionError:
                    self.cont_vars.remove('value')
                    
        else:  # if no list provided, use all available
            self.cont_vars = ['Demand', 'correction', 'correctedDemand', 'FRCE', 'LFCInput', 'aFRRactivation', 'correctionEcho']
        
        # Select categorical variables to use
        if cat_vars:
            self.cat_vars = cat_vars
        else:  # if no list provided, use all available
            self.cat_vars = ['participationCMO', 'participationIN', 'controlArea']
        
        # Finally, make two Numpy arrays for continuous and categorical
        # variables, respectively:
        if self.lbl_as_feat:
            self.cont = self.df[self.cont_vars].copy().to_numpy(dtype=np.float32)
        else:
            self.cont = self.df[self.cont_vars].copy().to_numpy(dtype=np.float32)
            self.lbl = self.df['value'].copy().to_numpy(dtype=np.float32)
        self.cat = self.df[self.cat_vars].copy().to_numpy(dtype=np.int64)
            
    def __getitem__(self, idx):
        if self.lbl_as_feat:  # for VAE training
            return torch.tensor(self.cont[idx]), torch.tensor(self.cat[idx])
        else:  # for supervised prediction
            return torch.tensor(self.cont[idx]), torch.tensor(self.cat[idx]), torch.tensor(self.lbl[idx])
    
    def __len__(self):
        return self.df.shape[0]

class Layer(nn.Module):
    '''
    A single fully connected layer with optional batch 
    normalisation and activation.
    '''
    def __init__(self, in_dim, out_dim, bn = True):
        super().__init__()
        layers = [nn.Linear(in_dim, out_dim)]
        if bn: 
            layers.append(nn.BatchNorm1d(out_dim))
        layers.append(nn.LeakyReLU(0.1, inplace=True))
        self.block = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.block(x)

class Encoder(nn.Module):
    '''
    The encoder part of our VAE. Takes a data sample and returns the
    mean and the log-variance of the vector's distribution.
    '''
    def __init__(self, **hparams):
        super().__init__()
        self.hparams = Namespace(**hparams)
        self.embeds = nn.ModuleList(
             [
                nn.Embedding(
                    n_cats, emb_size) for (n_cats, emb_size) \
                        in self.hparams.embedding_sizes
                
            ]
        )
        # The input to the first layer is the concatenation 
        # of all embedding vectors and continuous values
        in_dim = sum(emb.embedding_dim for emb in self.embeds) \
            + len(self.hparams.cont_vars)
        layer_dims = [in_dim] \
            + [int(s) for s in self.hparams.layer_sizes.split(',')]
        bn = self.hparams.batch_norm
        self.layers = nn.Sequential(
            *[Layer(layer_dims[i], layer_dims[i + 1], bn) \
                for i in range(len(layer_dims) - 1)],
        )
        self.mu = nn.Linear(layer_dims[-1], self.hparams.latent_dim)
        self.logvar = nn.Linear(
            layer_dims[-1], 
            self.hparams.latent_dim,
        )
    
    def forward(self, x_cont, x_cat):
        x_embed = [
            e(x_cat[:, i]) for i, e in enumerate(self.embeds)
        ]        
        x_embed = torch.cat(x_embed, dim=1)
        x = torch.cat((x_embed, x_cont), dim=1)
        h = self.layers(x)
        mu_ = self.mu(h)
        logvar_ = self.logvar(h)
        
        # we return the concatenated input vector for use in loss 
        return mu_, logvar_, x

class Decoder(nn.Module):
    '''
    The decoder part of our VAE. Takes a latent vector (sampled from
    the distribution learned by the encoder) and converts it back 
    to a reconstructed data sample.
    '''
    def __init__(self, **hparams):
        super().__init__()
        self.hparams = Namespace(**hparams)
        hidden_dims = [self.hparams.latent_dim] \
            + [
                   int(s) for s in \
                       reversed(self.hparams.layer_sizes.split(','))
        ]
        out_dim = sum(
            emb_size for _, emb_size in self.hparams.embedding_sizes
        ) + len(self.hparams.cont_vars) 
        bn = self.hparams.batch_norm
        self.layers = nn.Sequential(
            *[Layer(hidden_dims[i], hidden_dims[i + 1], bn) \
                for i in range(len(hidden_dims) - 1)],
        )
        self.reconstructed = nn.Linear(hidden_dims[-1], out_dim)
        
    def forward(self, z):
        h = self.layers(z)
        return self.reconstructed(h)

class VAE(L.LightningModule):
    def __init__(self, **hparams):
        super().__init__()
        self.save_hyperparameters()
        self.encoder = Encoder(**hparams)
        self.decoder = Decoder(**hparams)
        
    def reparameterize(self, mu, logvar):
        '''
        The reparameterisation trick allows us to backpropagate
        through the encoder.
        '''
        if self.training:
            std = torch.exp(logvar / 2.)
            eps = torch.randn_like(std) * self.hparams.stdev
            return eps * std + mu
        else:
            return mu
        
    def forward(self, batch):
        x_cont, x_cat = batch
        assert x_cat.dtype == torch.int64
        mu, logvar, x = self.encoder(x_cont, x_cat)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar, x
        
    def loss_function(self, obs, recon, mu, logvar):
        recon_loss = F.smooth_l1_loss(recon, obs, reduction='mean')
        kld = -0.5 * torch.mean(1 + logvar - mu ** 2 - logvar.exp())
        return recon_loss, kld
                               
    def training_step(self, batch, batch_idx): 
        ''' 
        Executed with each batch of data during training
        '''
        recon, mu, logvar, x = self.forward(batch)
        
        # The loss function compares the concatenated input vector
        # including embeddings to the reconstructed vector
        recon_loss, kld = self.loss_function(x, recon, mu, logvar)
        loss = recon_loss + self.hparams.kld_beta * kld
        # We log some values to monitor the training process
        self.log(
            'total_loss', loss.mean(dim=0), 
            on_step=True, prog_bar=True, logger=True,
        )
        self.log(
            'recon_loss', recon_loss.mean(dim=0), 
            on_step=True, prog_bar=True, logger=True,
        )
        self.log(
            'kld', kld.mean(dim=0), 
            on_step=True, prog_bar=True, logger=True,
        )
        return loss
    
    def test_step(self, batch, batch_idx):       
        ''' 
        Executed with each batch of data during testing
        '''
        recon, mu, logvar, x = self.forward(batch)
        recon_loss, kld = self.loss_function(x, recon, mu, logvar)
        loss = recon_loss + self.hparams.kld_beta * kld
        self.log('test_loss', loss)
        return loss
        
    def configure_optimizers(self):
        # Define the Adam optimiser:
        opt = torch.optim.AdamW(
            self.parameters(), lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay, eps=1e-4,
        )
        # Set up a cosine annealing schedule for the learning rate
        sch = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            opt, T_0=25, T_mult=1, eta_min=1e-9, last_epoch=-1,
        )
        return [opt], [sch]
    # The next two methods create the training and test data loaders 
    # based on the custom Dataset class.
    def train_dataloader(self):
        dataset = TSDataset(
            'train', cont_vars=self.hparams.cont_vars, 
            cat_vars = self.hparams.cat_vars, lbl_as_feat=True,
        )
        return DataLoader(
            dataset, batch_size=self.hparams.batch_size,
            num_workers=2, pin_memory=True, shuffle=True,
        )
    
    def test_dataloader(self):
        dataset = TSDataset(
            'test', cont_vars=self.hparams.cont_vars,
            cat_vars=self.hparams.cat_vars, lbl_as_feat=True,
        )
        return DataLoader(
            dataset, batch_size=self.hparams.batch_size, 
            num_workers=2, pin_memory=True, 
        )