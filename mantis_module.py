import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
import h5py
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import average_precision_score, precision_recall_curve

class MANTIS(nn.Module):
    def __init__(self, system_param: dict, opt_method: dict):
        super(MANTIS, self).__init__()
        self.system_param = system_param
        self.opt_method = opt_method
        self.datatype = torch.float32
        self.M = system_param['M']
        self.P = system_param['P']
        self.num_site = system_param['num_site']
        self.device = opt_method['device']
        self.encoding = opt_method['encoding']
        self.delta_P = system_param['delta_P']
        # MANTIS parameters size (L,M,P)
        if opt_method['set_initial']:
            self.θ = nn.Parameter(torch.zeros((self.num_site, self.M, self.P), dtype = self.datatype).to(self.device))
            self.coef = nn.Parameter(torch.ones(self.M, self.P, dtype = self.datatype).to(self.device))
        else:
            self.θ = nn.Parameter(torch.Tensor(self.num_site, self.M, self.P).uniform_(-1, 1).to(self.device))
            self.coef = nn.Parameter(torch.Tensor(self.M, self.P).uniform_(-1, 1).to(self.device))
    
    def log_loss(self, dataset):
        p_range = torch.arange(self.P, dtype=self.θ.dtype, device=self.θ.device)
        scaling_factor = torch.pi / (2**(self.delta_P*(p_range + 1)))
        input_ds = dataset['input'].to(self.device)
        target_ds = dataset['target'].to(self.device)
        anomaly_score = torch.where(target_ds == 0, torch.zeros_like(target_ds).to(self.device), 20*torch.ones_like(target_ds).to(self.device))
        x = scaling_factor.view(1, 1, self.P)*(input_ds[:,:,None]) #size (N,L,P)
        θ_x = self.θ.unsqueeze(0) + x[:,:,None,:] #size (N,L,M,P)
        Δθ = self.θ.view(1, self.num_site, self.M, 1, self.P, 1) - self.θ.view(1, self.num_site, 1, self.M, 1, self.P) #size (N,L,M,M,P,P)
        Δx = x.view(input_ds.shape[0], self.num_site, 1, 1, self.P, 1) - x.view(input_ds.shape[0], self.num_site, 1, 1, 1, self.P) #size (N,L,M,M,P,P)
        coef_list = self.coef # size (M,P)
        coef_tensor = torch.einsum('mp,nq->mnpq', coef_list, coef_list) # size (M,M,P,P)
        prob = torch.sum(coef_list.unsqueeze(0) * torch.prod(torch.cos(θ_x), dim = 1), dim = (1,2))**2
        normalization = torch.sum(coef_tensor.unsqueeze(0) * torch.prod(torch.cos(Δθ + Δx), dim = 1), dim = (1,2,3,4))
        prob_norm = prob/normalization

        if self.M == 1 and self.P == 1:
            reg_c_term = 0
        else:
            reg_c_term = torch.mean(self.coef**2)
        reg_theta_term = torch.mean(self.θ**2)
        loss0 = torch.mean((-torch.log(prob_norm + 1e-20) - anomaly_score)**2)
        loss =  loss0
        if self.opt_method['reg_param']['c'] != 0:
            loss += self.opt_method['reg_param']['c'] * (reg_c_term)
        if self.opt_method['reg_param']['theta'] != 0:
            loss += self.opt_method['reg_param']['theta'] * (reg_theta_term)
        

        return loss
   
    def norm_loss(self, dataset):
        p_range = torch.arange(self.P, dtype=self.θ.dtype, device=self.θ.device)
        scaling_factor = torch.pi / (2**(self.delta_P*(p_range + 1)))
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

    def local_0_loss(self, dataset):
        p_range = torch.arange(self.P, dtype=self.θ.dtype, device=self.θ.device)
        scaling_factor = torch.pi / (2**(self.delta_P*(p_range + 1)))
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
        scaling_factor = torch.pi / (2**(self.delta_P*(p_range + 1)))
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
        loss = torch.mean((1 - pauli_z_mean/normalize)**2) \
                        + self.opt_method['reg_param']['c'] * (reg_c_term) \
                        + self.opt_method['reg_param']['theta_m'] * (reg_theta_m_term) \
                        + self.opt_method['reg_param']['theta_p'] * (reg_theta_p_term)
        return loss

    def measurement(self, input_ds):
        p_range = torch.arange(self.P, dtype=self.θ.dtype, device=self.θ.device)
        scaling_factor = torch.pi / (2**(self.delta_P*(p_range + 1)))
        x = scaling_factor.view(1, 1, 1, self.P)*(input_ds.to(self.device)[:,:,None,None]) #size (N,L,M,P)
        θ_x = self.θ.unsqueeze(0) + x #size (N,L,M,P)
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
        num_data = input_ds.shape[0]
        num_batch = 64
        amplitude = []
        p_range = torch.arange(self.P, dtype=self.θ.dtype, device=self.θ.device)
        scaling_factor = torch.pi / (2**(self.delta_P*(p_range + 1)))
        for index_start in range(0, num_data, num_batch):
            index_end = min(index_start + num_batch, num_data)
            x = scaling_factor.view(1, 1, self.P)*(input_ds[index_start:index_end,:,None]) #size (N,L,P)
            θ_x = self.θ.unsqueeze(0) + x[:,:,None,:] #size (N,L,M,P)
            Δθ = self.θ.view(1, self.num_site, self.M, 1, self.P, 1) - self.θ.view(1, self.num_site, 1, self.M, 1, self.P) #size (N,L,M,M,P,P)
            Δx = x.view(x.shape[0], self.num_site, 1, 1, self.P, 1) - x.view(x.shape[0], self.num_site, 1, 1, 1, self.P) #size (N,L,M,M,P,P)
            coef_list = self.coef
            coef_tensor = torch.einsum('mp,nq->mnpq', coef_list, coef_list) # size (M,M,P,P)
            prob = torch.sum(coef_list.unsqueeze(0) * torch.prod(torch.cos(θ_x), dim = 1), dim = (1,2))**2
            normalization = torch.sum(coef_tensor.unsqueeze(0) * torch.prod(torch.cos(Δθ + Δx), dim = 1), dim = (1,2,3,4))
            amplitude_0 = prob/normalization
            amplitude.append(amplitude_0)
            del amplitude_0
        amplitude = torch.cat(amplitude)
        return amplitude
    
    def local_0_test(self, input_ds):
        input_ds = input_ds.to(self.θ.device)
        p_range = torch.arange(self.P, dtype=self.θ.dtype, device=self.θ.device)
        scaling_factor = torch.pi / (2**(self.delta_P*(p_range + 1)))
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
        scaling_factor = torch.pi / (2**(self.delta_P*(p_range + 1)))
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
        scaling_factor = torch.pi / (2**(self.delta_P*(p_range + 1)))
        x = scaling_factor.view(1, 1, self.P)*(input_ds[:,:,None]) #size (N,L,P)
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

#==================== Preprocessing ====================#

def get_rawdata(filename):
    if filename == "creditcard":
        csv_rawdata = pd.read_csv(f'{filename}.csv')
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
        input_ds_raw = torch.tensor(csv_rawdata.iloc[:, 1:].values, dtype=torch.float32)
        target_ds = torch.tensor(csv_rawdata.iloc[:, 0].values, dtype=torch.long)
        # input_ds = input_ds_raw
        L = input_ds_raw.shape[1]
        input_ds = torch.zeros_like(input_ds_raw, dtype=torch.float32)
        for l in range(L):
            col = input_ds_raw[:,l]
            uniques = torch.unique(col)
            k = len(uniques)
            if k == 1:
                input_ds[:,l] = 0
                continue
            norm_vals = torch.linspace(0.0, 1.0, k)
            idx = torch.searchsorted(uniques, col)
            input_ds[:, l] = norm_vals[idx]

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

        target_idx = df.shape[1] - 1
        df[target_idx] = df[target_idx].replace({1: 0, 7: 0, 3: 0, 5: 1, 2: 1, 4: 1})

        # Create input and target tensors
        input_ds_raw = torch.tensor(df.iloc[:,:target_idx].values, dtype=torch.float32)
        target_ds = torch.tensor(df[target_idx].values, dtype=torch.int64)  # Use int64 for classification

        L = input_ds_raw.shape[1]
        input_ds = torch.zeros_like(input_ds_raw, dtype=torch.float32)
        for l in range(L):
            col = input_ds_raw[:,l]
            uniques = torch.unique(col)
            k = len(uniques)
            if k == 1:
                input_ds[:,l] = 0
                continue
            norm_vals = torch.linspace(0.0, 1.0, k)
            idx = torch.searchsorted(uniques, col)
            input_ds[:, l] = norm_vals[idx]

        print("Input shape:", input_ds.shape)
        print("Target shape:", target_ds.shape)

    return {"input": input_ds, "target": target_ds}

def preprocessing(dataset: dict, method: str = 'quantile'):
    """
    Normalizes a 2D tensor along a specified dimension.

    Args:
        dataset (dict): The input dict with 'input' and 'target'.
        method (str): The normalization method.
                      Options: 'min-max', 'z-score', 'l2', 'quantile'.
        dim (int): The dimension to normalize along.
                   dim=0 normalizes each feature (column) across all samples.
                   dim=1 normalizes each sample (row) across all its features.

    Returns:
        normalized_dataset (dict): The normalized data dict.
    """

    X = dataset['input']
    if X.dim() != 2:
        raise ValueError("Input tensor must be 2-dimensional (N, L)")
    N, _ = X.shape
    epsilon = 1e-8
    dim = 0

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
        X_qn = rank/N
        print(X_qn.shape)

    elif method == 'gaussian':
        sorted_idx = torch.argsort(X, dim = 0)
        rank = torch.empty_like(sorted_idx)
        rank.scatter_(0, sorted_idx, torch.arange(1,X.shape[0]+1).unsqueeze(1).expand_as(X))
        u = rank/(N+1)
        finfo = torch.finfo(X.dtype)
        u = u.clamp(min=finfo.eps, max=1.0 - finfo.eps)
        X_qn = torch.sqrt(torch.tensor(2.0, device=X.device, dtype=X.dtype)) * torch.erfinv(2.0 * u - 1.0)

    elif method == 'robust':
        # Scales data based on quantiles, robust to outliers
        # Typically uses the Interquartile Range (IQR)
        q1 = torch.quantile(X, 0.25, dim=dim, keepdim=True)
        q3 = torch.quantile(X, 0.75, dim=dim, keepdim=True)
        median = torch.quantile(X, 0.5, dim=dim, keepdim=True)
        iqr = q3 - q1
        X_qn = (X - median) / (iqr + epsilon)
    elif method == None:
        return dataset
    else:
        raise ValueError("Invalid normalization method. Choose from 'min-max', 'z-score', 'l2', 'quantile', 'robust'.")
    normalized_dataset = {'input': X_qn, 'target': dataset['target']}
    return normalized_dataset

#==================== ML utilities ====================#

def random_selection(dataset: dict, num_random: int):
    """
    Select data points randomly
    """
    num_data = len(dataset['input'])
    indices = torch.randperm(num_data)
    selected_idx = indices[:num_random]
    remaining_idx = indices[num_random:]
    selected = {k:v[selected_idx] for (k,v) in dataset.items()}
    remaining = {k:v[remaining_idx] for (k,v) in dataset.items()}
    return selected, remaining


def manual_dataloader(dataset, batch_size=100, shuffle=True):
    """
    Manual dataloader in batch optimization
    """
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

#==================== Training and test ====================#

def training(training_set, system_param, opt_method):
    model = MANTIS(system_param,opt_method).to(opt_method['device'])
    num_epochs = opt_method['num_epochs']
    optimizer = torch.optim.AdamW(model.parameters(), lr=opt_method['lr'])
    print(f'{num_epochs} epoches in total')
    h5_filename = f"history/history_{opt_method['detail_name']}.h5"
    if os.path.exists(h5_filename):
        os.remove(h5_filename)
    h5f = h5py.File(h5_filename, "w")
    for epoch in range(num_epochs):
        loss = torch.zeros(1, dtype = torch.float32).to(opt_method['device'])
        count = 0
        for batch in manual_dataloader(training_set, batch_size=opt_method['num_batch']):
            optimizer.zero_grad()
            count += 1
            loss_batch = model.loss_fn(batch)
            if loss_batch < 0:
                raise "Amplitude > 1"
            loss_batch.backward()
            optimizer.step()
            loss += loss_batch * len(batch['input'])
        epoch_loss = loss / len(training_set['input'])
        if (epoch+1)%5 == 0 :
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss.item():.8f}')
    theta_log = h5f.create_dataset("theta", shape=(system_param['num_site'], system_param['M'], system_param['P']), dtype='f')
    theta_log[:] = model.θ.detach().cpu().numpy()
    coef_log = h5f.create_dataset("coef", shape=(system_param['M'], system_param['P']), dtype='f')
    coef_log[:] = model.coef.detach().cpu().numpy()
    h5f.close()
    return model, epoch_loss.item()

@torch.no_grad()
def testing(dataset, model):
    testing = model(dataset['input']).detach()
    target = dataset['target'].cpu().numpy()
    y_scores_for_roc = 1 - testing.cpu().numpy()
    fpr, tpr, thresholds = roc_curve(target, y_scores_for_roc)
    
    #AUROC
    roc_auc = auc(fpr, tpr)

    #AUPRC
    auprc = average_precision_score(target, y_scores_for_roc)
    precision, recall, thresholds_AUPRC = precision_recall_curve(target, y_scores_for_roc)

    return {'roc_auc': roc_auc, 'auprc': auprc}

    

