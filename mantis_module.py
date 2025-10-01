import torch
import torch.nn as nn
import numpy as np
import torch.multiprocessing as mp
import pandas as pd
import matplotlib.pyplot as plt
import os
from torch.utils.data import Dataset, DataLoader
import h5py
from torch.func import vmap
from sklearn.metrics import roc_curve, auc, classification_report, confusion_matrix
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import optuna
import multiprocessing as mp
import random
import math


class MANTIS_PRO(nn.Module):
    def __init__(self, system_param: dict, opt_method: dict):
        super(MANTIS_PRO, self).__init__()
        self.system_param = system_param
        self.opt_method = opt_method
        self.datatype = torch.float32
        self.M = system_param['M']
        self.P = system_param['P']
        self.num_site = system_param['num_site']
        self.device = opt_method['device']
        self.encoding = opt_method['encoding']

        # MANTIS parameters size (L,M,P)
        if opt_method['set_initial']:
            self.θ = nn.Parameter(torch.zeros((self.num_site, self.M, self.P), dtype = self.datatype).to(self.device)) #torch.DoubleTensor(self.num_site, self.M).uniform_(-torch.pi/2, torch.pi/2).to(self.device))
            self.coef = nn.Parameter(torch.ones(self.M, self.P, dtype = self.datatype).to(self.device))
        else:
            self.θ = nn.Parameter(torch.Tensor(self.num_site, self.M, self.P).uniform_(-1, 1).to(self.device))
            self.coef = nn.Parameter(torch.Tensor(self.M, self.P).uniform_(-1, 1).to(self.device))
    
    def log_loss(self, dataset):
        p_range = torch.arange(self.P, dtype=self.θ.dtype, device=self.θ.device)
        scaling_factor = torch.pi / (2**(p_range + 1))
        input_ds = dataset['input'].to(self.device)
        x = scaling_factor.view(1, 1, self.P)*(input_ds[:,:,None]) #size (N,L,P)
        θ_x = self.θ.unsqueeze(0) + x[:,:,None,:] #size (N,L,M,P)
        Δθ = self.θ.view(1, self.num_site, self.M, 1, self.P, 1) - self.θ.view(1, self.num_site, 1, self.M, 1, self.P) #size (N,L,M,M,P,P)
        Δx = x.view(input_ds.shape[0], self.num_site, 1, 1, self.P, 1) - x.view(input_ds.shape[0], self.num_site, 1, 1, 1, self.P) #size (N,L,M,M,P,P)
        coef_list = self.coef # size (M,P)
        coef_tensor = torch.einsum('mp,nq->mnpq', coef_list, coef_list) # size (M,M,P,P)
        prob = torch.sum(coef_list.unsqueeze(0) * torch.prod(torch.cos(θ_x), dim = 1), dim = (1,2))**2
        normalization = torch.sum(coef_tensor.unsqueeze(0) * torch.prod(torch.cos(Δθ + Δx), dim = 1), dim = (1,2,3,4))
        prob_norm = prob/normalization
        var_theta_m = torch.var(self.θ, dim=1)
        reg_theta_m_term = torch.mean(var_theta_m)
        if self.P != 1:
            var_theta_p = torch.var(self.θ, dim=2)
            reg_theta_p_term = torch.mean(var_theta_p)
        else:
            reg_theta_p_term = 0
        reg_c_term = torch.var(self.coef)
        # reg_term_mp = (self.θ[1:,:,:] - self.θ[:-1,:,:])**2
        # reg_term = torch.sum(reg_term_mp)
        loss = torch.mean(-torch.log(prob_norm + 1e-20)) \
                        + self.opt_method['reg_param']['c'] * (reg_c_term) \
                        + self.opt_method['reg_param']['theta_m'] * (reg_theta_m_term) \
                        + self.opt_method['reg_param']['theta_p'] * (reg_theta_p_term)
        return loss
        
        
    
    def norm_loss(self, dataset):
        p_range = torch.arange(self.P, dtype=self.θ.dtype, device=self.θ.device)
        scaling_factor = torch.pi / (2**(p_range + 1))
        input_ds = dataset['input'].to(self.device)
        x = scaling_factor.view(1, 1, self.P)*(input_ds[:,:,None]) #size (N,L,P)
        # θ_x = self.θ.unsqueeze(0) + x[:,:,None,:] #size (N,L,M,P)
        Δθ = self.θ.view(1, self.num_site, self.M, 1, self.P, 1) - self.θ.view(1, self.num_site, 1, self.M, 1, self.P) #size (N,L,M,M,P,P)
        Δx = x.view(input_ds.shape[0], self.num_site, 1, 1, self.P, 1) - x.view(input_ds.shape[0], self.num_site, 1, 1, 1, self.P) #size (N,L,M,M,P,P)
        coef_list = self.coef # size (M,P)
        coef_tensor = torch.einsum('mp,nq->mnpq', coef_list, coef_list) # size (M,M,P,P)
        normalization = torch.sum(coef_tensor.unsqueeze(0) * torch.prod(torch.cos(Δθ + Δx), dim = 1), dim = (1,2,3,4))
        frobenius_norm = torch.sum(coef_tensor.unsqueeze(0) * torch.prod(torch.cos(Δθ), dim = 1))
        λ = 0.3
        loss = torch.mean((torch.log(normalization + 1e-20)-1)**2) + λ * torch.nn.functional.relu(torch.log(frobenius_norm))
        return loss
    
    # def marginal_prob(self, input_ds, bitstring = torch.empty(0)):
    #     p_range = torch.arange(self.P, dtype=self.θ.dtype, device=self.θ.device)
    #     scaling_factor = torch.pi / (2**(p_range + 1))
    #     x = scaling_factor.view(1, 1, 1, self.P)*(input_ds.to(self.device)[:,:,None,None]) #size (N,L,M,P)
    #     θ_x = self.θ.unsqueeze(0) + x #size (N,L,M,P)
    #     Δθ = self.θ[None,:,:, None, :,None] - self.θ[None,:,None,:, :,None] #size (N,L,M,M,P,P)
    #     Δx = x[:,:,:,None,None,:] - x[:,:,:,None,:,None] #size (N,L,M,M,P,P)
    #     coef_list = self.coef
    #     coef_tensor = coef_list[None,:] * coef_list[:,None] # size (M,M)
    #     site_A = torch.where(bitstring[:,:,None,None,None,None], torch.einsum('nlp,nmq->nlmpq',l_term_m,l_term_m), torch.einsum('nlp,nmq->nlmpq',torch.cos(θ_x[:,l]),torch.cos(θ_x[:,l])))

    def local_0_loss(self, dataset):
        p_range = torch.arange(self.P, dtype=self.θ.dtype, device=self.θ.device)
        scaling_factor = torch.pi / (2**(p_range + 1))
        input_ds = dataset['input'].to(self.device)
        x = scaling_factor.view(1, 1, self.P)*(input_ds[:,:,None]) #size (N,L,P)
        θ_x = self.θ.unsqueeze(0) + x[:,:,None,:] #size (N,L,M,P)
        Δθ = self.θ.view(1, self.num_site, self.M, 1, self.P, 1) - self.θ.view(1, self.num_site, 1, self.M, 1, self.P) #size (N,L,M,M,P,P)
        Δx = x.view(input_ds.shape[0], self.num_site, 1, 1, self.P, 1) - x.view(input_ds.shape[0], self.num_site, 1, 1, 1, self.P) #size (N,L,M,M,P,P)
        coef_list = self.coef
        coef_tensor = torch.einsum('mp,nq->mnpq', coef_list, coef_list) # size (M,M,P,P)
        l_term_m = torch.cos(θ_x) # (N,L,M,P)
        l_term = torch.einsum('nlop,nlmq->nlompq',l_term_m,l_term_m) # (N,L,M,M,P,P)
        local_0_mn = torch.sum(l_term/(torch.cos(Δθ + Δx) + 1e-10), dim = 1) # (N,M,M,P,P)
        inner_product = torch.prod(torch.cos(Δθ + Δx), dim = 1) # (N,M,M,P,P)
        local_0 = torch.sum(coef_tensor.unsqueeze(0) * local_0_mn * inner_product, dim = (1,2,3,4))
        normalize = torch.sum(coef_tensor.unsqueeze(0) * inner_product, dim = (1,2,3,4))
        loss = torch.mean((1 - local_0/(normalize*self.num_site))**2)
        return loss
    
    def pauli_z_loss(self, dataset):
        p_range = torch.arange(self.P, dtype=self.θ.dtype, device=self.θ.device)
        scaling_factor = torch.pi / (2**(p_range + 1))
        input_ds = dataset['input'].to(self.device)
        x = scaling_factor.view(1, 1, self.P)*(input_ds[:,:,None]) #size (N,L,P)
        θ_x = self.θ.unsqueeze(0) + x[:,:,None,:] #size (N,L,M,P)
        coef_list = self.coef
        coef_tensor = torch.einsum('mp,nq->mnpq', coef_list, coef_list) # size (M,M,P,P)
        cos_θ_x_p = torch.cos(θ_x.view(input_ds.shape[0], self.num_site, self.M, 1, self.P, 1) + θ_x.view(input_ds.shape[0], self.num_site, 1, self.M, 1, self.P))
        cos_θ_x_m = torch.cos(θ_x.view(input_ds.shape[0], self.num_site, self.M, 1, self.P, 1) - θ_x.view(input_ds.shape[0], self.num_site, 1, self.M, 1, self.P))
        pauli_z = torch.sum(cos_θ_x_p/(cos_θ_x_m + 1e-10), dim = 1)
        normalize_mn = coef_tensor.unsqueeze(0) * torch.prod(cos_θ_x_m, dim = 1) #(N,M,M,P,P)
        pauli_z_mn = normalize_mn * pauli_z #(N,M,M,P,P)
        pauli_z_mean = torch.sum(pauli_z_mn, dim = (1,2,3,4))/self.num_site
        normalize = torch.sum(normalize_mn, dim = (1,2,3,4))
        var_theta_m = torch.var(self.θ, dim=1)
        reg_theta_m_term = torch.mean(var_theta_m) #torch.norm(self.θ)**2
        if self.P != 1:
            var_theta_p = torch.var(self.θ, dim=2)
            reg_theta_p_term = torch.mean(var_theta_p)
        else:
            reg_theta_p_term = 0
        reg_c_term = torch.var(self.coef)
        # reg_term_mp = (self.θ[1:,:,:] - self.θ[:-1,:,:])**2
        # reg_term = torch.sum(reg_term_mp)
        loss = torch.mean((1 - pauli_z_mean/normalize)**2) \
                        + self.opt_method['reg_param']['c'] * (reg_c_term) \
                        + self.opt_method['reg_param']['theta_m'] * (reg_theta_m_term) \
                        + self.opt_method['reg_param']['theta_p'] * (reg_theta_p_term)
        return loss



    def measurement(self, input_ds):
        p_range = torch.arange(self.P, dtype=self.θ.dtype, device=self.θ.device)
        scaling_factor = torch.pi / (2**(p_range + 1))
        x = scaling_factor.view(1, 1, 1, self.P)*(input_ds.to(self.device)[:,:,None,None]) #size (N,L,M,P)
        θ_x = self.θ.unsqueeze(0) + x #size (N,L,M,P)
        # first_p_term = torch.ones(input_ds.shape[0],self.M,self.M,self.P,self.P).to(self.datatype).to(self.device) # (N,M,M,P,P)
        Δθ = self.θ[None,:,:, None, :,None] - self.θ[None,:,None,:, :,None] #size (N,L,M,M,P,P)
        Δx = x[:,:,:,None,None,:] - x[:,:,:,None,:,None] #size (N,L,M,M,P,P)
        normalize_trigo = Δθ + Δx
        coef_list = self.coef
        coef_tensor = torch.einsum('mp,nq->mnpq', coef_list, coef_list) # size (M,M,P,P)
        first_p_term = coef_tensor[None,:,:,None,None]
        measurement_result = []
        for l in range(self.num_site):
            l_term_m = torch.sin(θ_x[:,l]) # (N,M,P)
            l_term = torch.einsum('nlp,nmq->nlmpq',l_term_m,l_term_m) # (N,M,M,P,P)
            prob_mn = first_p_term * l_term * torch.prod(torch.cos(normalize_trigo[l+1:]), dim = 1) # (N,M,M,P,P)
            normalize_mn = first_p_term * torch.prod(torch.cos(normalize_trigo[l:]), dim = 1) # (N,M,M,P,P)
            prob_l = torch.sum(prob_mn, dim = (1,2,3,4))/torch.sum(normalize_mn, dim = (1,2,3,4)) # (N)
            random_prob = torch.rand(input_ds.shape[0]).to(self.datatype).to(self.device)
            measurement_result_bool = random_prob < prob_l # True = accept 1, else = accept 0
            measurement_result_l = measurement_result_bool.int()
            measurement_result.append(measurement_result_l)
            l_term = torch.where(measurement_result_bool.view(-1,1,1,1,1), torch.einsum('nlp,nmq->nlmpq',l_term_m,l_term_m), torch.einsum('nlp,nmq->nlmpq',torch.cos(θ_x[:,l]),torch.cos(θ_x[:,l])))
            first_p_term *= l_term
        measurement_result = torch.stack(measurement_result).T # (N,L)
        return measurement_result.detach(), torch.sum(first_p_term, dim = (1,2,3,4))/torch.sum(torch.prod(torch.cos(normalize_trigo), dim=1), dim = (1,2,3,4))
    
    def hamming_dist(self, input_ds):
        num_sampling = self.opt_method['num_measure']
        hamming_distances = []
        for n in range(num_sampling):
            m, _ = self.measurement(input_ds)
            hamming_distance = torch.sum(m, dim = 1)/self.num_site
            hamming_distances.append(hamming_distance)
            if (n + 1) % 100 == 0:
                print(f'{n+1} measurements')
        mean_hamming = torch.mean(torch.stack(hamming_distances).float(), dim = 0)
        return mean_hamming
    
    def hamming_loss(self, input_ds):
        num_sampling = self.opt_method['num_measure']
        hamming_loss = torch.zeros(1, device=self.device)
        hamming_dist = torch.zeros(1, device=self.device)
        for n in range(num_sampling):
            m, P_m = self.measurement(input_ds)
            hamming_distance = torch.sum(m, dim = 1)/m.shape[1]
            hamming_loss += torch.mean(hamming_distance * torch.log(P_m))
            hamming_dist += torch.mean(hamming_distance)
            if (n + 1) % 100 == 0:
                print(f'{n+1} measurements')
        mean_hamming = hamming_loss/num_sampling
        mean_hamming_dist = hamming_dist/num_sampling
        # print(f'Mean Hamming distance {mean_hamming}')
        # print(f'Hamming Distance: {mean_hamming_dist[0].item():.8f}')
        return mean_hamming, mean_hamming_dist


    def loss_fn(self, dataset):
        '''
        Compute loss function

        Args:
            input (torch.Tensor): dataset tensor with size (N,L)
            target (torch.Tensor): 1D tensor with 0 or 1 with size N
        '''
        if self.opt_method['loss_fn'] != 'hamming':
            loss = getattr(self, f"{self.opt_method['loss_fn']}_loss")(dataset)
            return loss
        elif self.opt_method['loss_fn'] == 'hamming':
            loss, hamming_dist = self.hamming_loss(dataset)
            return loss, hamming_dist
    
    def amplitude_test(self,input_ds):
        input_ds = input_ds.to(self.θ.device)
        p_range = torch.arange(self.P, dtype=self.θ.dtype, device=self.θ.device)
        scaling_factor = torch.pi / (2**(p_range + 1))
        x = scaling_factor.view(1, 1, self.P)*(input_ds[:,:,None]) #size (N,L,P)
        θ_x = self.θ.unsqueeze(0) + x[:,:,None,:] #size (N,L,M,P)
        Δθ = self.θ.view(1, self.num_site, self.M, 1, self.P, 1) - self.θ.view(1, self.num_site, 1, self.M, 1, self.P) #size (N,L,M,M,P,P)
        Δx = x.view(input_ds.shape[0], self.num_site, 1, 1, self.P, 1) - x.view(input_ds.shape[0], self.num_site, 1, 1, 1, self.P) #size (N,L,M,M,P,P)
        coef_list = self.coef
        coef_tensor = torch.einsum('mp,nq->mnpq', coef_list, coef_list) # size (M,M,P,P)
        prob = torch.sum(coef_list.unsqueeze(0) * torch.prod(torch.cos(θ_x), dim = 1), dim = (1,2))**2
        normalization = torch.sum(coef_tensor.unsqueeze(0) * torch.prod(torch.cos(Δθ + Δx), dim = 1), dim = (1,2,3,4))
        amplitude_0 = prob/normalization
        return amplitude_0
    
    def local_0_test(self, input_ds):
        input_ds = input_ds.to(self.θ.device)
        p_range = torch.arange(self.P, dtype=self.θ.dtype, device=self.θ.device)
        scaling_factor = torch.pi / (2**(p_range + 1))
        x = scaling_factor.view(1, 1, self.P)*(input_ds[:,:,None]) #size (N,L,P)
        θ_x = self.θ.unsqueeze(0) + x[:,:,None,:] #size (N,L,M,P)
        Δθ = self.θ.view(1, self.num_site, self.M, 1, self.P, 1) - self.θ.view(1, self.num_site, 1, self.M, 1, self.P) #size (N,L,M,M,P,P)
        Δx = x.view(input_ds.shape[0], self.num_site, 1, 1, self.P, 1) - x.view(input_ds.shape[0], self.num_site, 1, 1, 1, self.P) #size (N,L,M,M,P,P)
        coef_list = self.coef
        coef_tensor = torch.einsum('mp,nq->mnpq', coef_list, coef_list) # size (M,M,P,P)
        l_term_m = torch.cos(θ_x) # (N,L,M,P)
        l_term = torch.einsum('nlop,nlmq->nlompq',l_term_m,l_term_m) # (N,L,M,M,P,P)
        local_0_mn = torch.sum(l_term/(torch.cos(Δθ + Δx) + 1e-10), dim = 1) # (N,M,M,P,P)
        inner_product = torch.prod(torch.cos(Δθ + Δx), dim = 1) # (N,M,M,P,P)
        local_0 = torch.sum(coef_tensor.unsqueeze(0) * local_0_mn * inner_product, dim = (1,2,3,4))
        normalize = torch.sum(coef_tensor.unsqueeze(0) * inner_product, dim = (1,2,3,4))
        loss = local_0/(normalize*self.num_site)
        return loss
    
    def pauli_z_test(self, input_ds):
        input_ds = input_ds.to(self.θ.device)
        p_range = torch.arange(self.P, dtype=self.θ.dtype, device=self.θ.device)
        scaling_factor = torch.pi / (2**(p_range + 1))
        x = scaling_factor.view(1, 1, self.P)*(input_ds[:,:,None]) #size (N,L,P)
        θ_x = self.θ.unsqueeze(0) + x[:,:,None,:] #size (N,L,M,P)
        coef_list = self.coef
        coef_tensor = torch.einsum('mp,nq->mnpq', coef_list, coef_list) # size (M,M,P,P)
        cos_θ_x_p = torch.cos(θ_x.view(input_ds.shape[0], self.num_site, self.M, 1, self.P, 1) + θ_x.view(input_ds.shape[0], self.num_site, 1, self.M, 1, self.P))
        cos_θ_x_m = torch.cos(θ_x.view(input_ds.shape[0], self.num_site, self.M, 1, self.P, 1) - θ_x.view(input_ds.shape[0], self.num_site, 1, self.M, 1, self.P))
        pauli_z = torch.sum(cos_θ_x_p/(cos_θ_x_m + 1e-10), dim = 1)
        normalize_mn = coef_tensor.unsqueeze(0) * torch.prod(cos_θ_x_m, dim = 1) #(N,M,M,P,P)
        pauli_z_mn = normalize_mn * pauli_z #(N,M,M,P,P)
        pauli_z_mean = torch.sum(pauli_z_mn, dim = (1,2,3,4))/self.num_site
        normalize = torch.sum(normalize_mn, dim = (1,2,3,4))
        loss = pauli_z_mean/normalize
        return loss

    def norm_test(self, input_ds):
        input_ds = input_ds.to(self.θ.device)
        p_range = torch.arange(self.P, dtype=self.θ.dtype, device=self.θ.device)
        scaling_factor = torch.pi / (2**(p_range + 1))
        x = scaling_factor.view(1, 1, self.P)*(input_ds[:,:,None]) #size (N,L,P)
        # θ_x = self.θ.unsqueeze(0) + x[:,:,None,:] #size (N,L,M,P)
        Δθ = self.θ.view(1, self.num_site, self.M, 1, self.P, 1) - self.θ.view(1, self.num_site, 1, self.M, 1, self.P) #size (N,L,M,M,P,P)
        Δx = x.view(input_ds.shape[0], self.num_site, 1, 1, self.P, 1) - x.view(input_ds.shape[0], self.num_site, 1, 1, 1, self.P) #size (N,L,M,M,P,P)
        coef_list = self.coef # size (M,P)
        coef_tensor = torch.einsum('mp,nq->mnpq', coef_list, coef_list) # size (M,M,P,P)
        normalization = torch.sum(coef_tensor.unsqueeze(0) * torch.prod(torch.cos(Δθ + Δx), dim = 1), dim = (1,2,3,4))
        loss = normalization
        return loss
    
    def forward(self, input_ds):
        if self.opt_method['test_fn'] != 'hamming':
            prediction = getattr(self, f"{self.opt_method['test_fn']}_test")(input_ds)
        return prediction







class MANTIS(nn.Module):
    def __init__(self, system_param: dict, opt_method: dict):
        super(MANTIS, self).__init__()
        self.system_param = system_param
        self.opt_method = opt_method
        self.datatype = torch.float32
        self.M = system_param['M']
        self.num_site = system_param['num_site']
        self.device = opt_method['device']
        self.encoding = opt_method['encoding']

        # MANTIS parameters size (L,M)
        if opt_method['set_initial']:
            self.θ = nn.Parameter(torch.zeros((self.num_site, self.M), dtype = self.datatype).to(self.device)) #torch.DoubleTensor(self.num_site, self.M).uniform_(-torch.pi/2, torch.pi/2).to(self.device))
            self.coef = nn.Parameter(torch.ones(self.M, dtype = self.datatype).to(self.device))
        else:
            self.θ = nn.Parameter(0.01 * torch.Tensor(self.num_site, self.M).uniform_(-torch.pi/2, torch.pi/2).to(self.device))
            self.coef = nn.Parameter(1 + 0.01 * torch.Tensor(self.M).uniform_(-1, 1).to(self.device))

    def MPO(self):
        θ = self.θ  # shape (num_site, M)
        cosθ = torch.cos(θ)  # (num_site, M)
        sinθ = torch.sin(θ)  # (num_site, M)

        # Build rotation matrices with shape (num_site, M, 2, 2)
        R = torch.stack([
            torch.stack([ cosθ, -sinθ], dim=-1),
            torch.stack([ sinθ,  cosθ], dim=-1)
        ], dim=-2)  # shape: (num_site, M, 2, 2)

        # Permute to match desired output shape: (M, num_site, 2, 2)
        MPO_list = R.permute(1, 0, 2, 3)
        return MPO_list
    
    def input_to_MPS(self,input):
        # size (N,L,2)
        if self.opt_method['encoding'] == 'fourier':
            m_vals = torch.arange(1,self.M+1, device=self.device)
            scale = m_vals * torch.pi/2
            X_expanded = input.unsqueeze(1).to(self.device)           # shape -> (N, 1, L)
            angle = X_expanded * scale.view(1, self.M, 1)
            input_embeded = torch.stack([torch.cos(angle),torch.sin(angle)], dim = -1).to(self.device)
        else:
            scale = torch.pi/2 * torch.ones(self.M, device=self.device)
            X_expanded = input.unsqueeze(1).to(self.device)         # shape -> (N, 1, L)
            angle = X_expanded * scale.view(1, self.M, 1)
            input_embeded = torch.stack([torch.cos(angle),torch.sin(angle)], dim = -1).to(self.device)
        return input_embeded

    def output_MPS(self,input):
        input_MPS = self.input_to_MPS(input).to(self.datatype)
        MPO_list = self.MPO().to(self.datatype)
        coef_list = self.coef.to(self.datatype)
        # print(f'input: {input_MPS.shape}, theta: {MPO_list.shape}')
        output = torch.einsum('m,mlpq,nmlq->nmlp',coef_list,MPO_list,input_MPS)
        return output # (N,M,L,2)
    
    def inner_product(self, MPS, MPS2=torch.empty(0)):
        '''
        For MPS: (M,L,2)
        '''
        if MPS2.numel() == 0:
            if not self.opt_method['L-root']:
                return torch.sum(torch.prod(torch.einsum('mlp,nlp->mnl',MPS,MPS), dim=-1))
            else:
                return torch.sum(torch.exp(torch.sum(torch.log(torch.abs(torch.einsum('mlp,nlp->mnl',MPS,MPS) + 1e-10)), dim=-1)/(self.num_site)))
        else:
            return torch.sum(torch.prod(torch.einsum('mlp,nlp->mnl',MPS,MPS2), dim=-1))

    def marginal_prob_single_bit(self, MPS, position: int, bit_value: int):
        """
        Computes the probability of observing a specific bit at a single position.

        Args:
            MPS (torch.Tensor): The Matrix Product State tensor of shape (N, M, L, 2).
            position (int): The single position 'l' to compute the probability for.
            bit_value (torch.Tensor): A 1D tensor of shape (N,) with the bit value (0 or 1)
                                    at the given position for each sample in the batch.
        """
        # Get the batch size N from the MPS tensor
        N = MPS.shape[0]

        # Create a copy of the MPS to avoid modifying the original tensor
        new_MPS = MPS.clone()

        ### --- Key Change: Zero out the opposite bit state at the specified position --- ###

        # 1. Create an index for each sample in the batch, e.g., [0, 1, ..., N-1]
        #    We ensure the index tensor is on the same device as the model's data.
        n_idx = torch.arange(N, device=MPS.device)

        # 2. Determine the component to zero out. If `bit_value` is 0, we zero out
        #    component 1, and if `bit_value` is 1, we zero out component 0.
        component_to_zero = 1 - bit_value

        # 3. Use advanced indexing to efficiently set the selected elements to 0.
        #    For each sample 'n' in the batch, this performs the operation:
        #    new_MPS[n, :, position, component_to_zero[n]] = 0
        #    The ':' ensures this is applied across all 'M' components.
        new_MPS[n_idx, :, position, component_to_zero] = 0

        # Calculate the probability of this conditioned state via the inner product
        prob_l = torch.abs(vmap(self.inner_product)(new_MPS))

        return prob_l

    def marginal_prob(self, MPS, bitstring = torch.empty(0)):
        '''
        MPS: (N,M,L,2)
        bitstring: (N,l)
        '''
        N = MPS.shape[0]
        M = self.M
        if bitstring.numel() > 0:
            new_MPS = MPS.clone()
            l = bitstring.shape[1]
            # Extract positions and components to zero
            positions = torch.arange(l).unsqueeze(0)  # shape (N,l,)
            b = bitstring          # shape (N,l,)
            not_b = 1 - b         # shape (N,l,)

            # Expand batch indices
            n_idx = torch.arange(N)[:, None, None, None]  # shape (N,1,1,1)
            m_idx = torch.arange(M)[None, :, None, None]  # shape (N,1,M,1)
            p_idx = positions[:, None, :, None]        # shape (N,1,1,l)
            c_idx = not_b[:, None, :,None]            # shape (N,1,1,l)

            # Broadcast to shape (N, M, l)
            n_idx = n_idx.expand(N, M, l, 2)
            m_idx = m_idx.expand(N, M, l, 2)
            p_idx = p_idx.expand(N, M, l, 2)
            c_idx = c_idx.expand(N, M, l, 2)

            # Set the selected elements to 0
            new_MPS[n_idx, m_idx, p_idx, c_idx] = 0
            prob_l = torch.abs(vmap(self.inner_product)(new_MPS))
        else:
            prob_l = torch.abs(vmap(self.inner_product)(MPS))
        # neg_mask = prob_l < 0
        # neg_indices = neg_mask.nonzero(as_tuple=False)
        # neg_values = prob_l[neg_mask]
        # if neg_indices.numel() > 0:  # if any negative values
        #     raise ValueError(
        #         f"Tensor contains negative values at the following positions:\n"
        #         f"indices:\n{neg_indices}\n"
        #         f"values:\n{neg_values}"
        #     )
        return prob_l
    
    def zipper_MPS(self,MPS):
        N = MPS.shape[0]
        measurement_result = torch.empty(N, 0, dtype=torch.int, device=self.device)
        for _ in range(self.num_site):
            measurement_l = torch.cat([measurement_result,torch.ones(N, 1, dtype=torch.int, device=self.device)], dim = 1)
            prob_1 = self.marginal_prob(MPS, measurement_l)/self.marginal_prob(MPS, measurement_result)
            random_prob = torch.rand(N).to(self.datatype).to(self.device)
            measurement_result_bool = random_prob < prob_1 # True = accept 1, else = accept 0
            measurement_result_l = measurement_result_bool.int()
            measurement_result = torch.cat((measurement_result,measurement_result_l.unsqueeze(-1)), dim = 1)
        normalize_const = self.marginal_prob(MPS)
        return measurement_result.detach(), self.marginal_prob(MPS, measurement_result)/normalize_const # size (N,L)

    def hamming_loss_MPS(self,dataset):
        input_ds = dataset['input']
        target_ds = dataset['target'].to(self.device)
        MPS = self.output_MPS(input_ds)
        num_sampling = self.opt_method['num_measure']
        hamming_loss = torch.zeros(1, device=self.device)
        hamming_dist = torch.zeros(1, device=self.device)
        for n in range(num_sampling):
            m, P_m = self.zipper_MPS(MPS)
            # print(f'measurement {n + 1}, results {m}')
            hamming_distance = torch.abs(target_ds - torch.sum(m, dim = 1)/m.shape[1])
            # print(P_m)
            hamming_loss += torch.mean(hamming_distance * torch.log(P_m))
            hamming_dist += torch.mean(hamming_distance)
            # print(f'{m}, Hamming: {hamming_distance}')
            if (n + 1) % 100 == 0:
                print(f'{n+1} measurements')
        mean_hamming = hamming_loss/num_sampling
        mean_hamming_dist = hamming_dist/num_sampling
        # print(f'Mean Hamming distance {mean_hamming}')
        print(f'Hamming Distance: {mean_hamming_dist[0].item():.8f}')
        return mean_hamming, mean_hamming_dist
    
    def hamming_test_MPS(self,dataset):
        input_ds = dataset['input']
        MPS = self.output_MPS(input_ds)
        num_sampling = 1000
        hamming_dist = torch.zeros(len(input_ds), device=self.device)
        for n in range(num_sampling):
            m, _ = self.zipper_MPS(MPS)
            # print(f'measurement {n + 1}, results {m}')
            hamming_distance = torch.sum(m, dim = 1)/m.shape[1]
            # print(P_m)
            hamming_dist += hamming_distance
            # print(f'{m}, Hamming: {hamming_distance}')
            if (n + 1) % 100 == 0:
                print(f'{n+1} measurements')
        mean_hamming_dist = hamming_dist/num_sampling
        return mean_hamming_dist
    
    def mse_loss_MPS(self,dataset):
        input_ds = dataset['input']
        MPS = self.output_MPS(input_ds)
        N = MPS.shape[0]
        mse_loss = 1 + torch.mean(torch.abs(self.marginal_prob(MPS))) - 2*torch.mean(torch.sqrt(torch.abs(self.marginal_prob(MPS,bitstring=torch.zeros(N,self.num_site,dtype=int)))))
        return mse_loss
    
    def log_mse_loss_MPS(self,dataset):
        input_ds = dataset['input']
        MPS = self.output_MPS(input_ds)
        N = MPS.shape[0]
        mse_loss = (1 + torch.mean(self.marginal_prob(MPS)) - 2*torch.mean(torch.sqrt(self.marginal_prob(MPS,bitstring=torch.zeros(N,self.num_site,dtype=int)))))
        return torch.log(mse_loss)

    def log_loss_MPS(self,dataset):
        input_ds = dataset['input']
        MPS = self.output_MPS(input_ds)
        N = MPS.shape[0]
        regularized = True
        normalization = self.marginal_prob(MPS)
        if not regularized:
            ϵ = 1e-10
            antilog = self.marginal_prob(MPS,bitstring=torch.zeros(N,self.num_site,dtype=int))/normalization + ϵ**2
            neg_mask = antilog < 0
            neg_indices = neg_mask.nonzero(as_tuple=False)
            neg_values = antilog[neg_mask]
            if neg_indices.numel() > 0:  # if any negative values
                raise ValueError(
                    f"Tensor contains negative values at the following positions:\n"
                    f"indices:\n{neg_indices}\n"
                    f"values:\n{neg_values}"
                )
            log_loss = torch.mean((-torch.log(antilog))**2)
        else:
            ϵ = 1e-10
            λ = 0.01
            normalize_trigo = self.θ[:,:, None] - self.θ[:,None,:] #size (L,M,M)
            coef_list = self.coef
            coef_tensor = coef_list[None,:] * coef_list[:,None]
            normalize_MPO = torch.sum(coef_tensor * torch.prod(torch.cos(normalize_trigo), dim = 0))
            antilog = self.marginal_prob(MPS,bitstring=torch.zeros(N,self.num_site,dtype=int))/normalization + ϵ**2
            # print(f'MPO trace {normalize_MPO}, loss {torch.mean(-torch.log(antilog))}')
            log_loss = torch.mean((-torch.log(antilog))**2) + λ * torch.log(2 * torch.nn.functional.relu(normalize_MPO))
        return log_loss

    
    def local_0_loss_MPS(self,dataset):
        input_ds = dataset['input']
        MPS = self.output_MPS(input_ds)
        N = MPS.shape[0]
        prob0local = torch.stack(
            [self.marginal_prob_single_bit(MPS, l, 0)
            for l in range(self.num_site)],
            dim=-1  # stack over last axis to get shape (L, N)
        ).T
        neg_mask = prob0local.isnan()
        neg_indices = neg_mask.nonzero(as_tuple=False)
        neg_values = prob0local[neg_mask]
        if neg_indices.numel() > 0:  # if any negative values
            raise ValueError(
                f"Tensor contains negative values at the following positions:\n"
                f"indices:\n{neg_indices}\n"
                f"values:\n{neg_values}"
            )
        normalize_const = self.marginal_prob(MPS)
        prob1local = torch.ones_like(prob0local) - prob0local/normalize_const
        neg_mask = prob1local.isnan()
        neg_indices = neg_mask.nonzero(as_tuple=False)
        neg_values = prob1local[neg_mask]
        if neg_indices.numel() > 0:  # if any negative values
            raise ValueError(
                f"Tensor contains negative values at the following positions:\n"
                f"indices:\n{neg_indices}\n"
                f"values:\n{neg_values}"
            )
        reg_param = 1e-2
        loss = torch.mean((1-torch.mean(prob0local/normalize_const, dim=-1))**2)
        # loss = torch.mean((1-torch.mean(2*prob0local/normalize_const-1, dim=-1))**2) #+ reg_param*torch.mean(torch.log(prob1local + 1e-10))
        # loss = torch.mean(torch.log(prob1local + 1e-10))
        return loss
    
    def recon_loss_MPS(self,dataset):
        input_ds = dataset['input']
        MPS_output = self.output_MPS(input_ds)
        normalize_const = self.marginal_prob(MPS_output)
        print(normalize_const)
        MPS_target = self.input_to_MPS(input_ds)[:,None,:]
        reconstruct_score_n = torch.abs(torch.vmap(self.inner_product)(MPS_target,MPS_output)/normalize_const)
        print(reconstruct_score_n)
        reconstruct_score = torch.mean(reconstruct_score_n)
        loss = 1 - reconstruct_score
        return loss
    
    def recon_test_MPS(self,input_ds):
        MPS_output = self.output_MPS(input_ds)
        MPS_target = self.input_to_MPS(input_ds)[:,None,:]
        normalize_const = self.marginal_prob(MPS_output)
        reconstruct_score_mnl = torch.vmap(self.inner_product)(MPS_target,MPS_output)
        reconstruct_score = 1 - torch.abs(reconstruct_score_mnl)/normalize_const
        return reconstruct_score




    def amplitude_test_MPS(self,input_ds):
        MPS = self.output_MPS(input_ds)
        N = MPS.shape[0]
        amplitude_0 = self.marginal_prob(MPS,bitstring=torch.zeros(N,self.num_site,dtype=int))/self.marginal_prob(MPS)
        return amplitude_0
    
    def local_amplitude_test_MPS(self,input_ds):
        MPS = self.output_MPS(input_ds)
        N = MPS.shape[0]
        prob0local = torch.stack(
            [self.marginal_prob_single_bit(MPS, l, 0)
            for l in range(self.num_site)],
            dim=-1  # stack over last axis to get shape (N, L)
        ).T
        normalize_const = self.marginal_prob(MPS)
        loss = torch.mean(prob0local/normalize_const, dim = 0)
        #loss = torch.mean(2*prob0local/normalize_const-1, dim = 0)
        return loss


    def loss_fn(self, dataset):
        '''
        Compute loss function

        Args:
            input (torch.Tensor): dataset tensor with size (N,L)
            target (torch.Tensor): 1D tensor with 0 or 1 with size N
        '''

        # compute θ[m,l] + input[n,l]
        input, target = dataset['input'].to(self.device), dataset['target'].to(self.device)
        trigo_arg = self.θ.unsqueeze(0) + 0.5*torch.pi*input.unsqueeze(-1) #size (N,L,M)
        normalize_trigo = self.θ[:,:, None] - self.θ[:,None,:] #size (L,M,M)
        coef_list = self.coef
        coef_tensor = coef_list[None,:] * coef_list[:,None] # size (M,M)
        # target_view = target.view(-1,1,1)
        #trigo_term = torch.where(target_view == 0, torch.sin(trigo_arg), torch.cos(trigo_arg)) #size (N,L,M)
        trigo_term = torch.cos(trigo_arg)
        # Fubini-Study
        if self.opt_method['loss_fn'] == 'fubini':
            if not self.opt_method['L-root']:
                normalize = torch.sqrt(torch.sum(coef_tensor * torch.prod(torch.cos(normalize_trigo), dim = 0)))
                inner_product = torch.sum(torch.prod(trigo_term, dim = 1), dim = 1)
                # print(f'inner product {inner_product/normalize}')
                # print(f'out-of-range: {torch.sum(torch.abs(inner_product/normalize) > 1)}')
                loss = torch.sum((torch.arccos(inner_product/(normalize + 1e-8)))**2)/input.shape[0]
            else:
                normalize = torch.sqrt(torch.sum(coef_tensor * torch.prod(torch.abs(torch.cos(normalize_trigo))**(1/input.shape[1]), dim = 0)))
                inner_product = torch.sum(torch.prod(torch.abs(trigo_term)**(1/input.shape[1]), dim = 1), dim = 1)
                # print(f'inner product {inner_product/normalize}')
                # print(f'out-of-range: {torch.sum(torch.abs(inner_product/normalize) > 1)}')
                loss = torch.sum((torch.arccos(inner_product/(normalize + 1e-8)))**2)/input.shape[0]
        elif self.opt_method['loss_fn'] == 'mse':
            if self.opt_method['L-root']:
                normalize = torch.sum(coef_tensor * torch.prod(torch.abs(torch.cos(normalize_trigo))**(1/input.shape[1]), dim = 0))
                overlap = torch.sum(coef_list * torch.prod(torch.abs(trigo_term)**(1/input.shape[1]), dim = 1))
                # print(f'normalize = {normalize}, overlap = {overlap}')
                loss = 1 + normalize - 2*overlap/(input.shape[0])
            else:
                normalize = torch.sum(coef_tensor * torch.prod(torch.cos(normalize_trigo), dim = 0))
                loss_b = 1 + normalize - 2*torch.sum(coef_list * torch.prod(trigo_term, dim = 1))/(input.shape[0])
                loss = loss_b
        elif self.opt_method['loss_fn'] == 'cross':
            normalize = torch.sqrt(torch.sum(coef_tensor.unsqueeze(0) * torch.prod(torch.cos(normalize_trigo), dim = 0)))
            p = (torch.sum(coef_tensor * torch.prod(trigo_term, dim = 1), dim = 1)/normalize)**2
            cross_entropy = - (1 - target) * torch.log(p + 1e-8) - target * torch.log(1 - p + 1e-8)
            loss = torch.sum(cross_entropy)/(input.shape[0])
        elif self.opt_method['loss_fn'] == 'hamming':
            num_sampling = self.opt_method['num_measure']
            hamming_loss = torch.zeros(1, device=self.device)
            hamming_dist = torch.zeros(1, device=self.device)
            for n in range(num_sampling):
                m, P_m = self.measurement(input)
                # print(f'measurement {n + 1}, results {m}')
                hamming_distance = torch.sum(m, dim = 1)/input.shape[1].to(self.datatype)
                # print(P_m)
                hamming_loss += torch.mean(hamming_distance.detach() * torch.log(P_m))
                hamming_dist += torch.mean(hamming_distance.detach())
                # print(f'{m}, Hamming: {hamming_distance}')
                if (n + 1) % 100 == 0:
                    print(f'{n+1} measurements')
            loss = hamming_loss/num_sampling
            print(f'Mean Hamming loss {loss}')
            print(f'Mean Hamming distance {hamming_dist/num_sampling}')
        elif self.opt_method['loss_fn'] == 'log':
            overlap_m = torch.prod(trigo_term, dim = 1)
            normalize_mn = torch.prod(torch.cos(normalize_trigo), dim = 0)
            normalization = torch.sum(coef_tensor * normalize_mn)
            # print(f'normalization value {normalization}')
            overlap = torch.sum(coef_list[None,:] * overlap_m, dim = 1)**2
            ϵ = 1e-10
            log_overlap = torch.log(overlap/normalization + ϵ**2)
            # print(log_overlap)
            loss = -torch.mean(log_overlap)

        elif self.opt_method['loss_fn'] == 'log_local':
            I_mn = torch.cos(normalize_trigo)
            normalize = torch.sum(coef_tensor * torch.prod(I_mn, dim = 0))
            sin_theta = torch.sin(self.θ.unsqueeze(0) + 0.5*torch.pi*input.unsqueeze(-1))
            sin_term = sin_theta[:,:,:, None] * sin_theta[:,:,None,:]
            hamming_mn = torch.prod(I_mn, dim = 0).unsqueeze(0) * sin_term/(I_mn.unsqueeze(0))
            local_inner = torch.log(torch.sum(coef_tensor * hamming_mn, dim = (-2,-1))/normalize)
            loss = -torch.sum(local_inner)/(input.shape[0] * self.num_site)
        elif self.opt_method['loss_fn'] == 'mse_tensor':
            print('We use full tensor')
            MPS_input = self.to_MPS(input)
            MPS_output = torch.vmap(self.MPO())(MPS_input)
            print(MPS_output.shape)
        
        return loss

    def forward(self, input):
        """
        Compute loss function for an individual input

        Args:
            input (torch.Tensor): dataset tensor with size L

        Returns:
            torch.Tensor: the loss function
        
        """
        normalize_trigo = self.θ[:,:, None] - self.θ[:,None,:]
        normalize = torch.sqrt(torch.sum(torch.prod(torch.cos(normalize_trigo), dim = 0)))
        print(f'Normalization constant {normalize}')
        trigo_arg = self.θ.unsqueeze(0) + 0.5*torch.pi*input.unsqueeze(-1).to(self.device)
        # print(trigo_arg.shape)
        inner_product = torch.sum(torch.prod(torch.cos(trigo_arg), dim = 1), dim = 1)
        print(f'result max {max(inner_product)/normalize}, min {min(inner_product)/normalize}')
        # Fubini-Study
        # loss = (inner_product/normalize) * input.shape[0]
        # Grover
        loss = inner_product**2/normalize
        return loss
    
    def measurement(self, input):
        trigo_arg = self.θ.unsqueeze(0).to(self.device) + 0.5*torch.pi*input.unsqueeze(-1).to(self.device)
        first_p_term = torch.ones(input.shape[0],self.M,self.M).to(self.datatype).to(self.device)
        normalize_trigo = self.θ[:,:, None] - self.θ[:,None,:]
        measurement_result = []
        for l in range(self.num_site):
            l_term_m = torch.sin(trigo_arg[:,l]) # (N,M)
            l_term = l_term_m[:,:,None] * l_term_m[:,None,:] # (N,M,M)
            prob_mn = first_p_term * l_term * torch.prod(torch.cos(normalize_trigo[l+1:]), dim = 0).unsqueeze(0)
            normalize_mn = first_p_term * torch.prod(torch.cos(normalize_trigo[l:]), dim = 0).unsqueeze(0)
            prob_l = torch.sum(prob_mn, dim = (1,2))/torch.sum(normalize_mn, dim = (1,2)) # (N)
            # print(f'Prob at l = {l} = {prob_l}')
            random_prob = torch.rand(input.shape[0]).to(self.datatype).to(self.device)
            measurement_result_bool = random_prob < prob_l # True = accept 1, else = accept 0
            measurement_result_l = measurement_result_bool.int()
            measurement_result.append(measurement_result_l)
            l_term = torch.where(measurement_result_bool.view(-1,1,1), torch.sin(trigo_arg[:,l])[:,:,None] * torch.sin(trigo_arg[:,l])[:,None,:], torch.cos(trigo_arg[:,l])[:,:,None] * torch.cos(trigo_arg[:,l])[:,None,:])
            first_p_term *= l_term
        measurement_result = torch.stack(measurement_result).T # (N,L)
        return measurement_result.detach(), torch.sum(first_p_term, dim = (1,2))/torch.sum(torch.prod(torch.cos(normalize_trigo), dim=0))
    
    def hamming_dist(self, input):
        num_sampling = self.opt_method['num_measure']
        hamming_distances = []
        for n in range(num_sampling):
            m, _ = self.measurement(input)
            # print(f'P_m check {torch.sum(P_m < torch.ones(len(P_m), device = self.device))}')
            # print(f'measurement {n + 1}, results {m}')
            hamming_distance = torch.sum(m, dim = 1)/self.num_site
            # print(f'{m}, Hamming: {hamming_distance}')
            hamming_distances.append(hamming_distance)
            if (n + 1) % 100 == 0:
                print(f'{n+1} measurements')
        mean_hamming = torch.mean(torch.stack(hamming_distances).float(), dim = 0)
        return mean_hamming
    
    def hamming_loss(self, input):
        num_sampling = self.opt_method['num_measure']
        hamming_loss = torch.zeros(1, device=self.device)
        for n in range(num_sampling):
            m, P_m = self.measurement(input)
            # print(f'measurement {n + 1}, results {m}')
            hamming_distance = torch.sum(m, dim = 1)/input.shape[1]
            # print(P_m)
            hamming_loss += torch.mean(hamming_distance * torch.log(P_m))
            # print(f'{m}, Hamming: {hamming_distance}')
            if (n + 1) % 100 == 0:
                print(f'{n+1} measurements')
        mean_hamming = hamming_loss/num_sampling
        print(f'Mean Hamming distance {mean_hamming}')
        return mean_hamming



        





def get_rawdata(filename):
    if filename == "creditcard":
        csv_rawdata = pd.read_csv(filename)
        column = ["Time"]
        column.extend([f"V{i}" for i in range(1,29)])
        column.extend(["Amount"])
        input_ds = torch.tensor(csv_rawdata[column].values)
        target_ds = torch.tensor(csv_rawdata["Class"].values, dtype=torch.int)
    elif filename == "wine":
        column_names = [
                            'Class', 'Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash',
                            'Magnesium', 'Total phenols', 'Flavanoids', 'Nonflavanoid phenols',
                            'Proanthocyanins', 'Color intensity', 'Hue',
                            'OD280/OD315 of diluted wines', 'Proline'
                        ]
        csv_rawdata = pd.read_csv('wine.data', header=None, names=column_names)
        csv_rawdata['Class'] = csv_rawdata['Class'].replace({1: 0, 2: 0, 3: 1})
        feature_columns = csv_rawdata.columns[1:] # Select all columns except the first one
        input_ds = torch.tensor(csv_rawdata[feature_columns].values, dtype=torch.float32)
        target_ds = torch.tensor(csv_rawdata["Class"].values, dtype=torch.int)
    elif filename == 'lymphography':
        csv_rawdata = pd.read_csv('lymphography.data', header=None)
        print(csv_rawdata.shape)

        # Reassign class labels: 1 and 4 -> 1, 2 and 3 -> 0
        csv_rawdata.loc[:, 0] = csv_rawdata[0].replace({1: 1, 2: 0, 3: 0, 4: 1})

        # Split into features and target
        input_ds = torch.tensor(csv_rawdata.iloc[:, 1:].values, dtype=torch.float32)
        target_ds = torch.tensor(csv_rawdata.iloc[:, 0].values, dtype=torch.long)
    elif filename == 'glass':
        # Load the dataset (glass.data has no header)
        csv_rawdata = pd.read_csv("glass.data", header=None)

        # Column 0 = ID, columns 1:-1 = features, column -1 = original class
        features = csv_rawdata.iloc[:, 1:-1]
        labels = csv_rawdata.iloc[:, -1]

        # Map: only class 6 → 1, all others → 0
        labels = (labels == 6).astype(int)

        # Convert to tensors
        input_ds = torch.tensor(features.values, dtype=torch.float32)
        target_ds = torch.tensor(labels.values, dtype=torch.long)
    
    elif filename == 'thyroid':
        csv_rawdata = pd.read_csv('thyroid/ann-train.data', header=None, sep=r'\s+')
        selected_cols = [0, 16, 17, 18, 19, 20, 21]
        df = csv_rawdata[selected_cols].copy()

        # Replace class labels (assuming 1 is positive, 2 and 3 are negative)
        target_idx = 21
        df[target_idx] = df[target_idx].replace({1: 1, 2: 0, 3: 0})

        # Create input and target tensors
        input_columns = selected_cols[:-1]  # all but the last column
        input_ds = torch.tensor(df[input_columns].values, dtype=torch.float32)
        target_ds = torch.tensor(df[target_idx].values, dtype=torch.int)  # Use int64 for classification

        print("Input shape:", input_ds.shape)
        print("Target shape:", target_ds.shape)
    elif filename == 'satellite':
        csv_rawdata = pd.read_csv('satellite/sat.trn', header=None, sep=r'\s+')
        df = csv_rawdata.copy()

        csv_rawdata_2 = pd.read_csv('satellite/sat.tst', header=None, sep=r'\s+')
        df = pd.concat([df, csv_rawdata_2], ignore_index=True)

        # class_counts = df[df.columns[-1]].value_counts()
        # print("Class counts:\n", class_counts)
        target_idx = df.shape[1] - 1
        df[target_idx] = df[target_idx].replace({1: 0, 7: 0, 3: 0, 5: 1, 2: 1, 4: 1})

        # Create input and target tensors
        input_ds = torch.tensor(df.iloc[:,:target_idx].values, dtype=torch.float32)
        target_ds = torch.tensor(df[target_idx].values, dtype=torch.int64)  # Use int64 for classification

        print("Input shape:", input_ds.shape)
        print("Target shape:", target_ds.shape)

    return {"input": input_ds, "target": target_ds}


# def remove_outliers_iqr(dataset):
#     input_r, target_r = dataset['input'], dataset['target']
#     mask_total = torch.tensor([True] * input_r.shape[0])
#     for col in range(input_r.shape[1]):
#         Q1 = input_r[:,col].quantile(0.75)
#         Q3 = input_r[:,col].quantile(0.25)
#         print(f'Col {col}, Max: {max(input_r[:,col])}, Min: {min(input_r[:,col])}')
#         IQR = Q3 - Q1
#         print(f'Upper bound: {Q1 - 1.5 * IQR}, Lower bound: {Q3 + 1.5 * IQR}')
#         mask = (input_r[:,col] <= Q1 - 1.5 * IQR) & (input_r[:,col] >= Q3 + 1.5 * IQR)
#         print(mask)
#         mask_total &= mask
#         print(f'Number of outliers: {sum(mask_total)}')
#     dataset['input'] = input_r[mask_total]
#     dataset['target'] = target_r[mask_total]
#     print(f'Number of data now: {len(dataset['input'])}')
#     return dataset

def preprocessing(dataset, method: str = 'quantile'):
    """
    Normalizes a 2D tensor along a specified dimension.

    Args:
        X (torch.Tensor): The input 2D tensor of shape (N, L).
        method (str): The normalization method.
                      Options: 'min-max', 'z-score', 'l2', 'quantile'.
        dim (int): The dimension to normalize along.
                   dim=0 normalizes each feature (column) across all samples.
                   dim=1 normalizes each sample (row) across all its features.

    Returns:
        torch.Tensor: The normalized tensor.
    """

    X = dataset['input']
    if X.dim() != 2:
        raise ValueError("Input tensor must be 2-dimensional (N, L)")
    N, D = X.shape
    epsilon = 1e-8
    dim = 0
    # print(input)
    # Sort each column and get sorted values + indices
    if method == 'min-max':
        dim = 0
        # Min-Max normalization (scales to [0, 1])
        min_val, _ = torch.min(X, dim=dim, keepdim=True)
        max_val, _ = torch.max(X, dim=dim, keepdim=True)
        range_val = max_val - min_val
        X_qn = (X - min_val) / (range_val + epsilon)


    elif method == 'z-score':
        # Z-score standardization (mean=0, std=1)
        mean_val = torch.mean(X, dim=dim, keepdim=True)
        std_val = torch.std(X, dim=dim, keepdim=True)
        X_qn = (X - mean_val) / (std_val + epsilon)


    elif method == 'l2':
        # L2 normalization (scales each vector to have unit norm)
        # This is typically always done instance-wise (dim=1)
        l2_norm = torch.linalg.norm(X, ord=2, dim=dim, keepdim=True)
        X_qn = X / (l2_norm + epsilon)


    elif method == 'quantile':
        sorted_idx = torch.argsort(X, dim = 0)
        rank = torch.empty_like(sorted_idx)
        rank.scatter_(0, sorted_idx, torch.arange(X.shape[0]).unsqueeze(1).expand_as(X))
        # print(rank)
        def y(alpha, x):
            return np.log(1+alpha*x)/np.log(1+alpha)
        # X_qn = y(N, rank/N)
        X_qn = rank/N
        # X_qn_n = torch.cat((time_input.unsqueeze(1), X_qn), dim = 1)
        print(X_qn.shape)

    # print(f'max {torch.max(X_qn)}, min {torch.min(X_qn)}')
    elif method == 'robust':
        # Scales data based on quantiles, robust to outliers
        # Typically uses the Interquartile Range (IQR)
        q1 = torch.quantile(X, 0.25, dim=dim, keepdim=True)
        q3 = torch.quantile(X, 0.75, dim=dim, keepdim=True)
        median = torch.quantile(X, 0.5, dim=dim, keepdim=True)
        iqr = q3 - q1
        X_qn = (X - median) / (iqr + epsilon)
    else:
        raise ValueError("Invalid normalization method. Choose from 'min-max', 'z-score', 'l2', 'quantile', 'robust'.")
    normalized_dataset = {'input': X_qn, 'target': dataset['target']}
    # with PdfPages(f'normalized_data_{method}.pdf') as pdf:
    #     for i in range(4):
    #         # 1. Create a new figure for the plot
    #         plt.figure()
            
    #         # 2. Generate your plot
    #         plt.scatter(X[:,i], X_qn[:,i])
    #         plt.xlabel(rf'$V_{i}$')
    #         plt.ylabel(rf'normalized $V_{i}$')
            
    #         # 3. Save the current figure to the PDF object
    #         pdf.savefig()
            
    #         # 4. Close the figure to free up memory
    #         plt.close()
    return normalized_dataset

def random_selection(dataset, num_random):
    num_data = len(dataset['input'])
    indices = torch.randperm(num_data)
    selected_idx = indices[:num_random]
    remaining_idx = indices[num_random:]
    selected = {k:v[selected_idx] for (k,v) in dataset.items()}
    remaining = {k:v[remaining_idx] for (k,v) in dataset.items()}
    return selected, remaining


def get_block_sweep(num_site, block_size):
    """
    Generates a tensor of indices for a block sweep pattern.

    Args:
        num_site (int): The total number of sites (L).
        block_size (int): The size of each block.

    Returns:
        torch.Tensor: A tensor of indices representing the block sweep.
                      Each row contains the indices of a block.
    """
    if block_size > num_site or block_size <= 0:
        raise ValueError("Block size must be positive and not larger than the number of sites.")
    if num_site < 1:
        raise ValueError("Number of sites must be at least 1.")

    indices_list = []
    L = num_site

    # Forward sweep
    for i in range(0, L - block_size + 1, block_size//2):
        indices_list.append(list(range(i, i + block_size)))

    # Backward sweep (excluding the last block if it overlaps with the start)
    if L > block_size:
        for i in range(L - block_size - block_size//2, -1, - block_size//2):
            indices_list.append(list(range(i, i + block_size)))

    return torch.tensor(indices_list, dtype=torch.int)

def manual_dataloader(dataset, batch_size=100, shuffle=True):
    inputs = dataset["input"]
    targets = dataset["target"]
    size = len(inputs)

    indices = np.arange(size)
    if shuffle:
        np.random.shuffle(indices)

    for start_idx in range(0, size, batch_size):
        end_idx = min(start_idx + batch_size, size)
        batch_indices = indices[start_idx:end_idx]
        yield {
            "input": inputs[batch_indices],
            "target": targets[batch_indices]
        }

def new_training(training_set, system_param, opt_method):
    model = MANTIS_PRO(system_param,opt_method).to(opt_method['device'])
    # MPO_trained = training_MPS(training_set,system_param,opt_method)
    # for name, param in model.named_parameters():
    #     print(f"Name: {name}")
    #     print(f"Shape: {param.shape}")
    #     print(f"Values: \n{param.data}\n")
    #     print(f"Require grad?: {param.requires_grad}")
    #     print("=========================")
    # print(f"No. of training set: {len(training_set['input'])}")
    num_epochs = opt_method['num_epochs']

    # print(f'We use {opt_method['loss_fn']} optimizer')
    # h5_filename = f"history/param_history_{opt_method['detail_name']}.h5"
    # if os.path.exists(h5_filename):
    #     os.remove(h5_filename)
    # h5f = h5py.File(h5_filename, "w")
    num_batches = int(np.ceil(len(training_set['input'])/opt_method['num_batch']))
    # grad_theta_log = h5f.create_dataset("grad theta", shape=(num_epochs * num_batches ,) + model.θ.shape, dtype='f')
    # grad_coef_log = h5f.create_dataset("grad coef", shape=(num_epochs * num_batches, ) + model.coef.shape, dtype='f')
    

    optimizer = torch.optim.AdamW(model.parameters(), lr=opt_method['lr'])
    if opt_method['loss_fn'] == 'hamming':
        # loss_log = h5f.create_dataset("loss", shape=(num_epochs * num_batches, 2, 1), dtype='f')
        for epoch in range(num_epochs):
            loss = torch.zeros(1, dtype = torch.float32).to(opt_method['device'])
            hamming = torch.zeros(1, dtype = torch.float32).to(opt_method['device'])
            count = 0
            for batch in manual_dataloader(training_set, batch_size=opt_method['num_batch']):
                optimizer.zero_grad()
                ind_h5f = (epoch * num_batches + count)
                count += 1
                # print(f'Batch {count}')
                loss_batch, hamming_batch = model.hamming_loss_MPS(batch)
                loss_batch.backward()
                # grad_theta_log[ind_h5f] = model.θ.grad.detach().cpu().numpy()
                # grad_coef_log[ind_h5f] = model.coef.grad.detach().cpu().numpy()
                # theta_log[ind_h5f] = model.θ.detach().cpu().numpy()
                # coef_log[ind_h5f] = model.coef.detach().cpu().numpy()
                # loss_log[ind_h5f] = [hamming_batch.detach().cpu().numpy(),loss_batch.detach().cpu().numpy()]
                optimizer.step()
                loss += loss_batch * len(batch['input'])
                hamming += hamming_batch * len(batch['input'])
            epoch_loss = loss / len(training_set['input'])
            epoch_hamming = hamming / len(training_set['input'])
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss.item():.8f}')
    else:
        # loss_log = h5f.create_dataset("loss", shape=(num_epochs * num_batches), dtype='f')
        print(f'{num_epochs} epoches in total')
        for epoch in range(num_epochs):
            loss = torch.zeros(1, dtype = torch.float32).to(opt_method['device'])
            count = 0
            for batch in manual_dataloader(training_set, batch_size=opt_method['num_batch']):
                optimizer.zero_grad()
                ind_h5f = (epoch * num_batches + count)
                count += 1
                loss_batch = model.loss_fn(batch)
                loss_batch.backward()
                # grad_theta_log[ind_h5f] = model.θ.grad.detach().cpu().numpy()
                # grad_coef_log[ind_h5f] = model.coef.grad.detach().cpu().numpy()
                # theta_log[ind_h5f] = model.θ.detach().cpu().numpy()
                # coef_log[ind_h5f] = model.coef.detach().cpu().numpy()
                # loss_log[ind_h5f] = loss_batch.detach().cpu().numpy()
                optimizer.step()
                loss += loss_batch * len(batch['input'])
                
            epoch_loss = loss / len(training_set['input'])
            # theta_log[epoch] = model.θ.detach().cpu().numpy()
            # coef_log[epoch] = model.coef.detach().cpu().numpy()
            # loss_log[epoch] = epoch_loss.detach().cpu().numpy()
            # if (epoch+1)%100 == 0 :
            #     print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss.item():.8f}')
    # h5f.create_dataset("theta", shape=model.θ.shape, dtype='f', data=model.θ.detach().cpu().numpy())
    # h5f.create_dataset("coef", shape=model.coef.shape, dtype='f', data=model.coef.detach().cpu().numpy())
    # h5f.close()
    return model, epoch_loss.item()



def training(dataset, system_param, opt_method):
    print(f"No. of training set: {len(dataset['input'])}")
    num_epochs = opt_method['num_epochs']
    model = MANTIS(system_param, opt_method).to(opt_method['device'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=opt_method['lr'])
    print(f'We use {opt_method['loss_fn']} optimizer')
    h5_filename = f"theta_history_{system_param['M']}_{opt_method['loss_fn']}.h5"
    if os.path.exists(h5_filename):
        os.remove(h5_filename)
    h5f = h5py.File(h5_filename, "w")
    theta_log = h5f.create_dataset("theta", shape=(num_epochs, system_param['num_site'], system_param['M']), dtype='f')
    loss_log = h5f.create_dataset("loss", shape=(num_epochs), dtype='f')
    coef_log = h5f.create_dataset("coef", shape=(num_epochs, system_param['M']), dtype='f')
    if opt_method["opt_mode"] == "global":
        print('We use Global optimizer')
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            loss = model.loss_fn(dataset)
            # print(f'loss {loss}')
            loss.backward()
            optimizer.step()
            if (epoch + 1) % 50 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss:.8f}')
                theta_log[epoch//50] = model.θ.detach().cpu().numpy()
                loss_log[epoch//50] = loss.detach().cpu().numpy()
            # if torch.isnan(loss):
            #     break
    elif opt_method["opt_mode"] == "global_dl":
        print('We use DataLoader')
        for epoch in range(num_epochs):
            loss = torch.zeros(1, dtype = torch.float32).to(opt_method['device'])
            count = 1
            for batch in manual_dataloader(dataset, batch_size=opt_method['num_batch'], shuffle=True):
                # print(f'Round {count}')
                optimizer.zero_grad()
                loss_batch = model.loss_fn(batch)
                # print(f'batch: {count}, loss: {loss_batch}')
                loss_batch.backward()
                optimizer.step()
                loss += loss_batch * len(batch['input'])
                count += 1
            epoch_loss = loss / len(dataset['input'])
            # if (epoch + 1) % 50 == 0:
            theta_log[epoch] = model.θ.detach().cpu().numpy()
            loss_log[epoch] = loss.detach().cpu().numpy()
            coef_log[epoch] = model.coef.detach().cpu().numpy()
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss.item():.8f}')
            # optimizer.zero_grad()
            # epoch_loss.backward(retain_graph=True)
            # optimizer.step()
            # if torch.isnan(epoch_loss):
            #     break
    elif opt_method["opt_mode"] == "local":
        num_sweeps = opt_method["num_sweeps"]
        block_sweep = get_block_sweep(system_param['num_site'], opt_method['sweep_size'])
        print(block_sweep)
        for sweep in range(num_sweeps):
            # Forward sweep (left-to-right)
            print(f"Sweep {sweep + 1}")
            for indices in block_sweep:
                for epoch in range(num_epochs):
                    old_params = model.θ.clone()
                    optimizer.zero_grad()
                    loss = model.loss_fn(dataset)
                    loss.backward(retain_graph=True)
                    optimizer.step()
                    with torch.no_grad():
                        mask = torch.ones(system_param['num_site'], dtype=torch.bool)
                        mask[indices] = False
                        # Use the mask to select the rows from old_params and update model.θ
                        model.θ[mask] = old_params[mask]
                    # print(model.θ)
                    if (epoch + 1) % 50 == 0:
                        print(f'Epoch [{epoch+1}/{num_epochs}]/Sweep {sweep + 1}, Loss: {loss:.8f}')
    h5f.close()
    return model

def training_MPS(dataset, system_param, opt_method):
    print(f"No. of training set: {len(dataset['input'])}")
    
    model = MANTIS(system_param,opt_method)
    # MPO_trained = training_MPS(training_set,system_param,opt_method)
    for name, param in model.named_parameters():
        print(f"Name: {name}")
        print(f"Shape: {param.shape}")
        print(f"Values: \n{param.data}\n")
        print(f"Require grad?: {param.requires_grad}")
        print("=========================")
    print(f"No. of training set: {len(dataset['input'])}")
    num_epochs = opt_method['num_epochs']
    model = MANTIS(system_param, opt_method).to(opt_method['device'])

    print(f'We use {opt_method['loss_fn']} optimizer')
    h5_filename = f"grad_theta_history_{system_param['M']}_{opt_method['loss_fn']}_{opt_method["encoding"]}.h5"
    if os.path.exists(h5_filename):
        os.remove(h5_filename)
    h5f = h5py.File(h5_filename, "w")
    num_batches = int(np.ceil(len(dataset['input'])/100))
    grad_theta_log = h5f.create_dataset("grad theta", shape=(num_epochs * num_batches ,) + model.θ.shape, dtype='f')
    grad_coef_log = h5f.create_dataset("grad coef", shape=(num_epochs * num_batches, ) + model.coef.shape, dtype='f')
    theta_log = h5f.create_dataset("theta", shape=(num_epochs * num_batches ,) + model.θ.shape, dtype='f')
    coef_log = h5f.create_dataset("coef", shape=(num_epochs * num_batches, ) + model.coef.shape, dtype='f')

    optimizer = torch.optim.AdamW(model.parameters(), lr=opt_method['lr'])
    if opt_method['loss_fn'] == 'hamming':
        loss_log = h5f.create_dataset("loss", shape=(num_epochs * num_batches, 2, 1), dtype='f')
        for epoch in range(num_epochs):
            loss = torch.zeros(1, dtype = torch.float32).to(opt_method['device'])
            hamming = torch.zeros(1, dtype = torch.float32).to(opt_method['device'])
            count = 0
            for batch in manual_dataloader(dataset,batch_size=opt_method['num_batch']):
                optimizer.zero_grad()
                ind_h5f = (epoch * num_batches + count)
                count += 1
                # print(f'Batch {count}')
                loss_batch, hamming_batch = model.hamming_loss_MPS(batch)
                loss_batch.backward()
                grad_theta_log[ind_h5f] = model.θ.grad.detach().cpu().numpy()
                grad_coef_log[ind_h5f] = model.coef.grad.detach().cpu().numpy()
                theta_log[ind_h5f] = model.θ.detach().cpu().numpy()
                coef_log[ind_h5f] = model.coef.detach().cpu().numpy()
                loss_log[ind_h5f] = [hamming_batch.detach().cpu().numpy(),loss_batch.detach().cpu().numpy()]
                optimizer.step()
                loss += loss_batch * len(batch['input'])
                hamming += hamming_batch * len(batch['input'])
            epoch_loss = loss / len(dataset['input'])
            epoch_hamming = hamming / len(dataset['input'])
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss.item():.8f}')
    else:
        loss_log = h5f.create_dataset("loss", shape=(num_epochs * num_batches), dtype='f')
        for epoch in range(num_epochs):
            loss = torch.zeros(1, dtype = torch.float32).to(opt_method['device'])
            count = 0
            for batch in manual_dataloader(dataset,batch_size=opt_method['num_batch']):
                optimizer.zero_grad()
                ind_h5f = (epoch * num_batches + count)
                count += 1
                # print(f'Batch {count}')
                if opt_method['loss_fn'] == 'log':
                    loss_batch = model.log_loss_MPS(batch)
                elif opt_method['loss_fn'] == 'local_0':
                    loss_batch = model.local_0_loss_MPS(batch)
                elif opt_method['loss_fn'] == 'recon':
                    loss_batch = model.recon_loss_MPS(batch)
                loss_batch.backward()
                grad_theta_log[ind_h5f] = model.θ.grad.detach().cpu().numpy()
                grad_coef_log[ind_h5f] = model.coef.grad.detach().cpu().numpy()
                theta_log[ind_h5f] = model.θ.detach().cpu().numpy()
                coef_log[ind_h5f] = model.coef.detach().cpu().numpy()
                loss_log[ind_h5f] = loss_batch.detach().cpu().numpy()
                optimizer.step()
                loss += loss_batch * len(batch['input'])
                
            epoch_loss = loss / len(dataset['input'])
            # theta_log[epoch] = model.θ.detach().cpu().numpy()
            # coef_log[epoch] = model.coef.detach().cpu().numpy()
            # loss_log[epoch] = epoch_loss.detach().cpu().numpy()
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss.item():.8f}')
    h5f.close()
    return model

def training_reg(dataset, system_param, opt_method):
    λ1_range = torch.tensor([1]).to(opt_method['device']) #torch.logspace(-2,2,9)
    λ2_range = torch.logspace(start=-4, end=4, steps=9)
    mask_normal = dataset["target"] == 0
    training_positive_set = {k:v[mask_normal] for (k,v) in dataset.items()}
    mask_anomalous = dataset["target"] == 1
    training_negative_set = {k:v[mask_anomalous] for (k,v) in dataset.items()}
    print(f'Data size: {training_positive_set['target'].shape[0]} positive, {training_negative_set['target'].shape[0]} negative')
    num_epochs = opt_method['num_epochs']
    best_loss = torch.tensor(torch.inf, dtype = torch.float32).to(opt_method['device'])
    print(f'We use {opt_method['loss_fn']}')
    for λ1 in λ1_range:
        for λ2 in λ2_range:
            print(f'(λ1,λ2) = {(λ1.item(),λ2.item())}')
            model = MANTIS(system_param, opt_method).to(opt_method['device'])
            optimizer = torch.optim.AdamW(model.parameters(), lr=opt_method['lr'])
            if opt_method["opt_mode"] == "global":
                for epoch in range(num_epochs):
                    optimizer.zero_grad()
                    loss = λ1 * model.loss_fn(training_positive_set) + λ2 * model.loss_fn(training_negative_set)
                    loss.backward()
                    optimizer.step()
                    if (epoch + 1) % 50 == 0:
                        print(f'Epoch [{epoch+1}/{num_epochs}], (λ1,λ2) = {(λ1.item(),λ2.item())}, Loss: {loss:.8f}')
                    if torch.isnan(loss):
                        break
            if loss < best_loss:
                best_loss = loss
                best_reg = (λ1,λ2)
    
    #For best (λ1,λ2)
    print(f'Best (λ1,λ2): {best_reg}')
    λ1,λ2 = best_reg
    model = MANTIS(system_param, opt_method).to(opt_method['device'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=opt_method['lr'])
    if opt_method["opt_mode"] == "global":
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            loss = λ1 * model.loss_fn(training_positive_set) + λ2 * model.loss_fn(training_negative_set)
            loss.backward()
            optimizer.step()
            if (epoch + 1) % 50 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], (λ1,λ2) = {(λ1,λ2)}, Loss: {loss:.8f}')
            if torch.isnan(loss):
                break
    return model

def find_optimal_threshold(model, validation_set, acceptable_false_positive_rate=0.05):
    """
    Finds the optimal fidelity threshold from a validation set of normal data.

    In fraud detection, the "False Positive Rate" is the percentage of legitimate
    transactions that are incorrectly flagged as fraud.

    Args:
        model (MANTIS): The trained model.
        validation_set (dict): A dataset containing ONLY normal (Class 0) data.
        acceptable_false_positive_rate (float): The fraction of normal transactions
                                                you are willing to misclassify.

    Returns:
        float: The calculated fidelity threshold.
    """
    print("--- Finding Optimal Threshold from Validation Set ---")
    if torch.any(validation_set['target'] != 0):
        raise ValueError("Validation set for threshold finding must only contain normal data (Class 0).")
    # Get fidelity scores for the normal validation data
    # fidelity_scores = torch.abs(model.local_amplitude_test_MPS(validation_set['input']))
    fidelity_scores = model(validation_set['input']).detach()
    threshold = torch.quantile(fidelity_scores.cpu(), acceptable_false_positive_rate).item()

    # Set the calculated threshold as a new attribute on the model object
    model.threshold = threshold

    print(f"To achieve a False Positive Rate of ~{acceptable_false_positive_rate*100}%, "
          f"the fidelity threshold has been set to: {threshold:.4f}")
    print(f"This threshold has been stored in `model.threshold`.")
    
    # y_scores = 1 - fidelity_scores.detach().cpu().numpy()
    # y_true = validation_set['target'].cpu().numpy()

    # # For scikit-learn, the score should be the probability of the POSITIVE class (fraud=1).
    # # Since a low fidelity score means fraud, we use (1 - fidelity) as the score.
    

    # # Calculate ROC curve
    # fpr, tpr, thresholds_roc = roc_curve(y_true, y_scores)

    # # Find the first threshold that gives a TPR (fraud recall) >= our target
    # try:
    #     optimal_idx = np.min(np.where(tpr >= fraud_recall_target))
    #     # The threshold from roc_curve is for the (1 - fidelity) score
    #     optimal_threshold_for_score = thresholds_roc[optimal_idx]
    #     # We convert it back to a threshold for our original fidelity score
    #     threshold = 1 - optimal_threshold_for_score
    # except (ValueError, IndexError):
    #     print(f"\nWarning: Could not meet the desired fraud recall target of {fraud_recall_target*100}%.")
    #     print("Using the threshold that gives the highest possible recall instead.")
    #     optimal_idx = np.argmax(tpr)
    #     threshold = 1 - thresholds_roc[optimal_idx]

    # Set the calculated threshold as a new attribute on the model object
    model.threshold = threshold

    # print(f"To achieve a Fraud Recall of ~{tpr[optimal_idx]*100:.1f}%, "
    #       f"the fidelity threshold has been set to: {threshold:.4f}")
    # print(f"This will result in a False Positive Rate of: {fpr[optimal_idx]*100:.1f}%")
    # print(f"This threshold has been stored in `model.threshold`.")
@torch.no_grad()
def testing(dataset, model):
    # if not hasattr(model, 'threshold') or model.threshold is None:
    #     raise ValueError("Model threshold has not been set. Please run `find_optimal_threshold` on the model first.")

    # threshold = model.threshold
    testing = model(dataset['input']).detach()
    # print(f'testing lenght: {len(testing)}')
    target = dataset['target'].cpu().numpy()
    # print(f'Test results: {testing}')
    y_scores_for_roc = 1 - testing.cpu().numpy()
    fpr, tpr, thresholds = roc_curve(target, y_scores_for_roc)
    roc_auc = auc(fpr, tpr)
    target_tpr = 0.95
    # print(f"--- Finding threshold for TPR >= {target_tpr} ---")

    # Find the first index of the threshold that achieves a TPR of at least our target
    # Note: [0] is used to select the first such occurrence
    try:
        idx_tpr = np.where(tpr >= target_tpr)[0][0]
        threshold = thresholds[idx_tpr]
        actual_fpr_for_tpr = fpr[idx_tpr]
        
        # print(f"✅ Threshold to achieve TPR >= {target_tpr}: {threshold:.4f}")
        # print(f"   This results in an FPR of: {actual_fpr_for_tpr:.4f}")

    except IndexError:
        print(f"⚠️ Could not find a threshold to achieve a TPR of {target_tpr}.")
    

    print("\n" + "="*50 + "\n")
    # plt.figure()
    # plt.scatter(range(len(testing)),testing.detach().cpu().numpy(),c=target, alpha=0.5)
    # plt.xlabel('# of data')
    # plt.ylabel(r'p(0...0)')
    # plt.savefig(f'testing_({model.M},{model.P})_{model.opt_method['loss_fn']}_{model.opt_method["encoding"]}.pdf')

    mask = dataset['target'] == 0
    # plt.figure()
    # bins = np.arange(0, 1.02, 0.02)
    # plt.hist(testing.detach().cpu().numpy()[mask], density=True, alpha=0.5, bins = bins)
    mask_not = dataset['target'] == 1
    # if torch.any(mask_not):
    #     plt.hist(testing.detach().cpu().numpy()[mask_not], density=True, alpha=0.5, bins = bins)
    # plt.xlabel(r'p(0...0)')
    # plt.ylabel('Per cent')
    # plt.savefig(f'hist/hist_({model.M},{model.P})_{model.opt_method['loss_fn']}_{model.opt_method["encoding"]}.pdf')

    predictions = (testing.cpu().numpy() < threshold).astype(int)

    # Get the classification report as a string
    # report_str = classification_report(target, predictions, target_names=['Normal (Class 0)', 'Fraud (Class 1)'])
    # print("\nClassification Report at Given Threshold:")
    # print(report_str)
    # # 3. PLOT RESULTS AND SAVE TO A SINGLE PDF
    # report_filename = f'analysis_report/analysis_report_{model.opt_method['detail_name']}.pdf'
    # with PdfPages(report_filename) as pdf:
    #     # Page 1: Classification Report Texะ
    #     fig_text, ax_text = plt.subplots(figsize=(8.5, 11))
    #     ax_text.axis('off')
    #     ax_text.text(0.05, 0.95, report_str, va='top', ha='left', fontsize=10, fontfamily='monospace')
    #     ax_text.set_title('Classification Report', fontsize=16)
    #     pdf.savefig(fig_text)
    #     plt.close(fig_text)
    #     # Plot 1: Confusion Matrix
    #     fig1, ax1 = plt.subplots(figsize=(8, 6))
    #     cm = confusion_matrix(target, predictions)
    #     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
    #                 xticklabels=['Predicted Normal', 'Predicted Fraud'],
    #                 yticklabels=['Actual Normal', 'Actual Fraud'])
    #     ax1.set_title(f'Confusion Matrix (Threshold = {threshold:.2f})')
    #     ax1.set_ylabel('Actual Label')
    #     ax1.set_xlabel('Predicted Label')
    #     pdf.savefig(fig1)
    #     plt.close(fig1)

    #     # Plot 2: Score Distribution Histogram
    #     fig2, ax2 = plt.subplots(figsize=(10, 6))
    #     sns.histplot(data={'scores': testing.cpu().numpy(), 'label': target}, x='scores', hue='label',
    #                  bins=50, stat='density', common_norm=False, palette=['steelblue', 'darkorange'], ax=ax2)
    #     ax2.axvline(threshold, color='r', linestyle='--', label=f'Decision Threshold = {threshold:.2f}')
    #     ax2.set_title('Score Distribution for Normal vs. Fraud')
    #     ax2.set_xlabel('Fidelity Score with Normal Class')
    #     # Manually set legend labels for clarity
    #     handles, _ = ax2.get_legend_handles_labels()
    #     ax2.legend(handles, ['Normal', 'Fraud'], title='Class')
    #     pdf.savefig(fig2)
    #     plt.close(fig2)

    #     # Plot 3: ROC Curve for overall model context
    #     fig3, ax3 = plt.subplots(figsize=(8, 8))
        
        
        
    #     print(f"\nFor context, the overall AUC of the model is: {roc_auc:.4f}")
    #     ax3.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    #     ax3.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    #     ax3.set_xlim([0.0, 1.0])
    #     ax3.set_ylim([0.0, 1.05])
    #     ax3.set_xlabel('False Positive Rate')
    #     ax3.set_ylabel('True Positive Rate (Recall)')
    #     ax3.set_title('Receiver Operating Characteristic (ROC) Curve')
    #     ax3.legend(loc="lower right")
    #     ax3.grid(True)
    #     pdf.savefig(fig3)
    #     plt.close(fig3)

    # print(f"\nAll analysis plots have been saved to '{report_filename}'.")



    # For normal
    normal_check = testing[mask] >= threshold
    normal_acc = torch.sum(normal_check)/len(testing[mask])
    # For anomalous
    anomaly_check = testing[mask_not] < threshold
    anomaly_acc = torch.sum(anomaly_check)/len(testing[mask_not])

    # print(f'Normal accuracy: {normal_acc*100}%')
    # print(f'Anomalous accuracy: {anomaly_acc*100}%')
    
    # Overall
    prediction = torch.tensor([0 if x >= threshold else 1 for x in testing])
    error = torch.sum((prediction + target)%2)/len(testing)
    accuracy = 1 - error
    # print(f'Accuracy: {accuracy*100}%')
    return {'roc_auc': roc_auc.item(), 'acc': accuracy.item()}

def testing_MPS(dataset, model):
    # Check if the threshold has been set on the model
    if not hasattr(model, 'threshold') or model.threshold is None:
        raise ValueError("Model threshold has not been set. Please run `find_optimal_threshold` on the model first.")
    model.opt_method['L-root'] = False
    threshold = model.threshold
    if not model.opt_method['test_fn'] == 'hamming':
        testing = torch.abs(model.amplitude_test_MPS(dataset['input'])).detach()
    else:
        testing = model.hamming_test_MPS(dataset)
    print(f'testing lenght: {len(testing)}')
    target = dataset['target'].cpu().numpy()
    print(f'Test results: {testing}')
    plt.figure()
    plt.scatter(range(len(testing)),testing.detach().cpu().numpy(),c=target, alpha=0.5)
    plt.xlabel('# of data')
    plt.ylabel(r'p(0...0)')
    plt.savefig(f'testing_{model.M}_{model.opt_method['loss_fn']}_{model.opt_method["encoding"]}.pdf')

    mask = dataset['target'] == 0
    plt.figure()
    bins = np.arange(0, 1.02, 0.02)
    plt.hist(testing.detach().cpu().numpy()[mask], density=True, alpha=0.5, bins = bins)
    mask_not = dataset['target'] == 1
    if torch.any(mask_not):
        plt.hist(testing.detach().cpu().numpy()[mask_not], density=True, alpha=0.5, bins = bins)
    plt.xlabel(r'p(0...0)')
    plt.ylabel('Per cent')
    plt.savefig(f'hist_{model.M}_{model.opt_method['loss_fn']}_{model.opt_method["encoding"]}.pdf')

    predictions = (testing.cpu().numpy() < threshold).astype(int)

    # Get the classification report as a string
    report_str = classification_report(target, predictions, target_names=['Normal (Class 0)', 'Fraud (Class 1)'])
    print("\nClassification Report at Given Threshold:")
    print(report_str)
    # 3. PLOT RESULTS AND SAVE TO A SINGLE PDF
    report_filename = f'analysis_report_{model.M}_{model.opt_method['loss_fn']}_{model.opt_method["encoding"]}.pdf'
    with PdfPages(report_filename) as pdf:
        # Page 1: Classification Report Text
        fig_text, ax_text = plt.subplots(figsize=(8.5, 11))
        ax_text.axis('off')
        ax_text.text(0.05, 0.95, report_str, va='top', ha='left', fontsize=10, fontfamily='monospace')
        ax_text.set_title('Classification Report', fontsize=16)
        pdf.savefig(fig_text)
        plt.close(fig_text)
        # Plot 1: Confusion Matrix
        fig1, ax1 = plt.subplots(figsize=(8, 6))
        cm = confusion_matrix(target, predictions)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
                    xticklabels=['Predicted Normal', 'Predicted Fraud'],
                    yticklabels=['Actual Normal', 'Actual Fraud'])
        ax1.set_title(f'Confusion Matrix (Threshold = {threshold:.2f})')
        ax1.set_ylabel('Actual Label')
        ax1.set_xlabel('Predicted Label')
        pdf.savefig(fig1)
        plt.close(fig1)

        # Plot 2: Score Distribution Histogram
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        sns.histplot(data={'scores': testing.cpu().numpy(), 'label': target}, x='scores', hue='label',
                     bins=50, stat='density', common_norm=False, palette=['steelblue', 'darkorange'], ax=ax2)
        ax2.axvline(threshold, color='r', linestyle='--', label=f'Decision Threshold = {threshold:.2f}')
        ax2.set_title('Score Distribution for Normal vs. Fraud')
        ax2.set_xlabel('Fidelity Score with Normal Class')
        # Manually set legend labels for clarity
        handles, _ = ax2.get_legend_handles_labels()
        ax2.legend(handles, ['Normal', 'Fraud'], title='Class')
        pdf.savefig(fig2)
        plt.close(fig2)

        # Plot 3: ROC Curve for overall model context
        fig3, ax3 = plt.subplots(figsize=(8, 8))
        y_scores_for_roc = 1 - testing.cpu().numpy()
        fpr, tpr, _ = roc_curve(target, y_scores_for_roc)
        roc_auc = auc(fpr, tpr)
        print(f"\nFor context, the overall AUC of the model is: {roc_auc:.4f}")
        ax3.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        ax3.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax3.set_xlim([0.0, 1.0])
        ax3.set_ylim([0.0, 1.05])
        ax3.set_xlabel('False Positive Rate')
        ax3.set_ylabel('True Positive Rate (Recall)')
        ax3.set_title('Receiver Operating Characteristic (ROC) Curve')
        ax3.legend(loc="lower right")
        ax3.grid(True)
        pdf.savefig(fig3)
        plt.close(fig3)

    print(f"\nAll analysis plots have been saved to '{report_filename}'.")



    # For normal
    normal_check = testing[mask] >= threshold
    normal_acc = torch.sum(normal_check)/len(testing[mask])
    # For anomalous
    anomaly_check = testing[mask_not] < threshold
    anomaly_acc = torch.sum(anomaly_check)/len(testing[mask_not])

    print(f'Normal accuracy: {normal_acc*100}%')
    print(f'Anomalous accuracy: {anomaly_acc*100}%')
    
    # Overall
    prediction = torch.tensor([0 if x >= threshold else 1 for x in testing])
    error = torch.sum((prediction + target)%2)/len(testing)
    accuracy = 1 - error
    print(f'Accuracy: {accuracy*100}%')


def testing_sampling_MPS(dataset, model):
    hamming_dist_list = model.hamming_test_MPS(dataset)
    target = dataset['target']
    print(f'Test results: {hamming_dist_list}')
    np.savetxt(f'Testing_{model.M}_{model.opt_method['loss_fn']}.csv',hamming_dist_list.detach().cpu().numpy(),delimiter=",")
    bins = np.arange(0, 1.02, 0.02)
    plt.figure()
    plt.scatter(range(len(hamming_dist_list)),hamming_dist_list.detach().cpu().numpy(),c=target, alpha=0.5)
    plt.xlabel('# of data')
    plt.ylabel(r'Mean Hamming distance')
    plt.savefig(f'testing_sampling_{model.M}_{model.opt_method['loss_fn']}.pdf')
    mask = dataset['target'] == 0
    plt.figure()
    plt.hist(hamming_dist_list.detach().cpu().numpy()[mask], density=True, alpha=0.5, bins = bins)
    bin_range = (max(hamming_dist_list.detach().cpu().numpy()[mask]) - min(hamming_dist_list.detach().cpu().numpy()[mask]))/10
    mask_not = dataset['target'] == 1
    if torch.any(mask_not):
        plt.hist(hamming_dist_list.detach().cpu().numpy()[mask_not], density=True, alpha=0.5, bins=bins) #bins=int((max(testing.detach().cpu().numpy()[mask_not]) - min(testing.detach().cpu().numpy()[mask_not]))/bin_range))
    plt.xlabel(r'Mean Hamming distance')
    plt.ylabel('Per cent')
    plt.savefig(f'hist_sampling_{model.M}_{model.opt_method['loss_fn']}.pdf')

    # For normal
    # normal_check = testing[mask] <= 0.1
    # normal_acc = torch.sum(normal_check)/len(testing[mask])
    # # For anomalous
    # anomaly_check = testing[mask_not] > 0.1
    # anomaly_acc = torch.sum(anomaly_check)/len(testing[mask_not])

    # print(f'Normal accuracy: {normal_acc*100}%')
    # print(f'Anomalous accuracy: {anomaly_acc*100}%')
    
    # # Overall
    # prediction = torch.tensor([0 if x <= 0.1 else 1 for x in testing])
    # error = torch.sum((prediction + target)%2)/len(testing)
    # accuracy = 1 - error
    # print(f'Accuracy: {accuracy*100}%')


def testing_sampling(dataset, model):
    model.opt_method['num_measure'] = 1000
    testing = model.hamming_dist(dataset['input'])
    print(f'testing lenght: {len(testing)}')
    target = dataset['target']
    print(f'Test results: {testing}')
    np.savetxt(f'Testing_{model.M}_{model.opt_method['loss_fn']}.csv',testing.detach().cpu().numpy(),delimiter=",")
    bins = np.arange(0, 1.02, 0.02)
    plt.figure()
    plt.scatter(range(len(testing)),testing.detach().cpu().numpy(),c=target, alpha=0.5)
    plt.xlabel('# of data')
    plt.ylabel(r'Mean Hamming distance')
    plt.savefig(f'testing_sampling_{model.M}_{model.opt_method['loss_fn']}.pdf')
    mask = dataset['target'] == 0
    plt.figure()
    plt.hist(testing.detach().cpu().numpy()[mask], density=True, alpha=0.5, bins = bins)
    bin_range = (max(testing.detach().cpu().numpy()[mask]) - min(testing.detach().cpu().numpy()[mask]))/10
    mask_not = dataset['target'] == 1
    if torch.any(mask_not):
        plt.hist(testing.detach().cpu().numpy()[mask_not], density=True, alpha=0.5, bins=bins) #bins=int((max(testing.detach().cpu().numpy()[mask_not]) - min(testing.detach().cpu().numpy()[mask_not]))/bin_range))
    plt.xlabel(r'Mean Hamming distance')
    plt.ylabel('Per cent')
    plt.savefig(f'hist_sampling_{model.M}_{model.opt_method['loss_fn']}.pdf')

    # For normal
    normal_check = testing[mask] <= 0.1
    normal_acc = torch.sum(normal_check)/len(testing[mask])
    # For anomalous
    anomaly_check = testing[mask_not] > 0.1
    anomaly_acc = torch.sum(anomaly_check)/len(testing[mask_not])

    print(f'Normal accuracy: {normal_acc*100}%')
    print(f'Anomalous accuracy: {anomaly_acc*100}%')
    
    # Overall
    prediction = torch.tensor([0 if x <= 0.1 else 1 for x in testing])
    error = torch.sum((prediction + target)%2)/len(testing)
    accuracy = 1 - error
    print(f'Accuracy: {accuracy*100}%')

# def measurement(MPO_result, opt_method):
    

