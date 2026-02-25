import torch
import torch.nn as nn
from torch.func import vmap  # PyTorch >= 2.0; for older use: from functorch import vmap
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import average_precision_score, precision_recall_curve
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class TNAD(nn.Module):
    def __init__(self, system_param, opt_method):
        super(TNAD, self).__init__()

        self.system_param = system_param
        self.opt_method = opt_method

        self.num_site = system_param['num_site']  # L
        self.P = system_param['P']                # physical dimension per site
        self.reg_param = opt_method['reg_param']
        self.chi = opt_method.get('chi', 5)       # MPO bond dimension
        self.device = opt_method['device']

        # --- MPO parameters ---
        mpo_params = []
        for l in range(self.num_site):
            if l == 0:
                Dl, Dr = 1, self.chi          # left boundary
            elif l == self.num_site - 1:
                Dl, Dr = self.chi, 1          # right boundary
            else:
                Dl, Dr = self.chi, self.chi   # bulk

            # W[l] has shape (Dl, Dr, P_in, P_out)
            W_l = nn.Parameter(torch.randn(Dl, Dr, self.P, self.P))
            mpo_params.append(W_l)

        self.mpo_tensors = nn.ParameterList(mpo_params)

    def MPO(self):
        return list(self.mpo_tensors)

    # --------- Encoding / MPS construction ---------
    def phi_encoding(self, x):
        """
        x: tensor of shape (L,)  (one sample, all sites)
        returns: tensor of shape (L, P)
        Assumes P = 2k with k = P//2, using π / 2^p encoding.
        """
        k = self.P // 2
        device = x.device

        p = torch.arange(1, k + 1, device=device)  # (k,)
        freq = torch.pi / (2.0 ** p)               # (k,)

        # x: (L,) -> (L,1) for broadcasting
        x_expand = x.unsqueeze(-1)                 # (L,1)
        angles = x_expand * freq                   # (L,k)

        cos_vals = torch.cos(angles)               # (L,k)
        sin_vals = torch.sin(angles)               # (L,k)

        phi = torch.cat([cos_vals, sin_vals], dim=-1)  # (L, 2k) == (L, P)
        phi = phi / torch.sqrt(torch.tensor(float(k), device=device))
        return phi

    def MPS_input(self, data_1d):
        """
        data_1d: shape (L,)
        returns: MPS tensor of shape (L, P)  (product state)
        """
        return self.phi_encoding(data_1d)

    # --------- MPO -> MPS application ---------
    def apply_MPO_to_MPS(self, MPS):
        """
        MPS: tensor of shape (L, P)
        mpo_tensors[l]: (Dl, Dr, P_in, P_out)

        returns: list of length L, each element A_l with shape (Dl, Dr, P_out)
        """
        MPS_out = []
        for l in range(self.num_site):
            W = self.mpo_tensors[l]   # (Dl, Dr, P, P)
            v = MPS[l]                # (P,)

            # W_{lrso} v_s -> B_{lro}
            # l = left bond, r = right bond, s = in phys, o = out phys
            B = torch.einsum('lrso,s->lro', W, v)  # (Dl, Dr, P)
            MPS_out.append(B)

        return MPS_out

    # --------- Norms / Regularization ---------
    def MPO_norm(self):
        """
        Simple Frobenius norm over all MPO parameters.
        """
        sq_sum = torch.zeros((), device=self.mpo_tensors[0].device)
        for W in self.mpo_tensors:
            sq_sum = sq_sum + W.pow(2).sum()
        return torch.sqrt(sq_sum)


    def MPS_norm(self, MPS_list, eps=1e-30, verbose=False):
        A0 = MPS_list[0]
        device, dtype = A0.device, A0.dtype

        E = torch.ones(1, 1, device=device, dtype=dtype)
        log_scale = torch.zeros((), device=device, dtype=torch.float64)

        for i, A in enumerate(MPS_list):
            E = torch.einsum('ab,acp,bdp->cd', E, A.conj(), A)
            # rescale environment to keep numbers bounded
            s = E.abs().max()
            E = E / s
            log_scale = log_scale + torch.log(s.double())
            if verbose:
                print(f"site {i}: max|E|={s.item():.3e}, log_scale={log_scale.item():.3e}")

        # <psi|psi> = E_scalar * exp(log_scale)
        overlap = E.squeeze().real  # should be 1x1 at the end
        overlap = overlap.clamp_min(0.0)  # numerical safety

        # ||psi|| = sqrt(overlap * exp(log_scale)) = exp(0.5*(log_scale + log(overlap)))
        log_norm = 0.5 * (log_scale + torch.log(overlap.double() + 1e-300))
        norm = torch.exp(log_norm).to(dtype)
        return norm

    # --------- Loss function (vectorized over batch) ---------
    def loss_fn(self, dataset):
        """
        dataset['input']: tensor of shape (N, L)
        loss = mean_sample ||MPO(MPS(x_n))|| + λ ||MPO||_F
        """
        train_data = dataset['input'].to(self.device)  # (N, L)
        mpo_norm_val = self.MPO_norm()

        def single_sample_loss(data_1d):
            # data_1d: (L,)
            MPS = self.MPS_input(data_1d)          # (L, P)
            MPS_out = self.apply_MPO_to_MPS(MPS)   # list of (Dl, Dr, P)
            norm = self.MPS_norm(MPS_out)          # scalar
            
            return (torch.log(norm + 1e-10) - 1)**2

        # vmap over batch dimension N
        per_sample_loss = vmap(single_sample_loss)(train_data)  # (N,)
        loss = per_sample_loss.mean() + self.reg_param * torch.relu(torch.log(mpo_norm_val))
        return loss
    
    def forward(self, input_ds):
        input_ds = input_ds.to(self.device)
        num_data = len(input_ds)
        num_batch = 64
        amplitude = []
        def single_sample_loss(data_1d):
            # data_1d: (L,)
            MPS = self.MPS_input(data_1d)          # (L, P)
            MPS_out = self.apply_MPO_to_MPS(MPS)   # list of (Dl, Dr, P)
            return self.MPS_norm(MPS_out)          # scalar
        for index_start in range(0, num_data, num_batch):
            index_end = min(index_start + num_batch, num_data)
            input_batch = input_ds[index_start:index_end]
            per_sample_loss = vmap(single_sample_loss)(input_batch)
            amplitude.append(per_sample_loss)
            del per_sample_loss
        amplitude = torch.cat(amplitude)
        return amplitude

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

        # class_counts = df[df.columns[-1]].value_counts()
        # print("Class counts:\n", class_counts)
        target_idx = df.shape[1] - 1
        df[target_idx] = df[target_idx].replace({1: 0, 7: 0, 3: 0, 5: 1, 2: 1, 4: 1})

        # Create input and target tensors
        input_ds_raw = torch.tensor(df.iloc[:,:target_idx].values, dtype=torch.float32)
        target_ds = torch.tensor(df[target_idx].values, dtype=torch.int64)  # Use int64 for classification
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

        print("Input shape:", input_ds.shape)
        print("Target shape:", target_ds.shape)

    return {"input": input_ds, "target": target_ds}

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

    elif method == 'gaussian':
        sorted_idx = torch.argsort(X, dim = 0)
        rank = torch.empty_like(sorted_idx)
        rank.scatter_(0, sorted_idx, torch.arange(1,X.shape[0]+1).unsqueeze(1).expand_as(X))
        u = rank/(N+1)
        finfo = torch.finfo(X.dtype)
        u = u.clamp(min=finfo.eps, max=1.0 - finfo.eps)
        X_qn = torch.sqrt(torch.tensor(2.0, device=X.device, dtype=X.dtype)) * torch.erfinv(2.0 * u - 1.0)

    # print(f'max {torch.max(X_qn)}, min {torch.min(X_qn)}')
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

def random_selection(dataset, num_random):
    num_data = len(dataset['input'])
    indices = torch.randperm(num_data)
    selected_idx = indices[:num_random]
    remaining_idx = indices[num_random:]
    selected = {k:v[selected_idx] for (k,v) in dataset.items()}
    remaining = {k:v[remaining_idx] for (k,v) in dataset.items()}
    return selected, remaining

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

# Machine Learning operation

def training(dataset, system_param, opt_method):
    model = TNAD(system_param, opt_method).to(opt_method['device'])
    optimizer = torch.optim.AdamW(model.mpo_tensors.parameters(), lr=opt_method['lr'])
    num_epochs = opt_method['num_epochs']
    for epoch in range(num_epochs):
        loss = torch.zeros(1).to(opt_method['device'])
        for data in manual_dataloader(dataset, batch_size=opt_method['num_batch'], shuffle=True):
            optimizer.zero_grad()
            loss_batch = model.loss_fn(data)
            loss_batch.backward()
            optimizer.step()
            # print(loss_batch.item(), end=' ')
            loss += loss_batch * len(data['input'])
        epoch_loss = loss / len(dataset['input'])
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss.item():.4f}')
    return model, epoch_loss.item()
    
def testing(dataset, model):
    testing = model(dataset['input']).detach().cpu().numpy()
    print(testing)
    
    target = dataset['target'].cpu().numpy()
    y_target = target  # invert labels: normal=0 -> 1, anomaly=1 -> 0
    fpr, tpr, thresholds = roc_curve(y_target, testing)
    # print(fpr, tpr, thresholds)
    roc_auc = auc(fpr, tpr)
    print(f'ROC-AUC: {roc_auc}')
    #AUPRC
    auprc = average_precision_score(target, testing)
    print(f'AUPRC: {auprc}')
    return roc_auc, auprc

rawdata = 'creditcard' # "satellite", "lymphography" #"creditcard", "wine", 'thyroid'

local_rank = int(os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK", 0))
if torch.cuda.is_available():
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
else:
    device = torch.device("cpu")


dataset = get_rawdata(rawdata)

if rawdata == 'lymphography' or rawdata == 'satellite':
    encode = None
else:
    encode = 'quantile'

dataset_r = preprocessing(dataset, encode)

mask_normal = dataset_r["target"] == 0
normal_set = {k:v[mask_normal] for (k,v) in dataset_r.items()}

mask_anomaly = dataset_r["target"] == 1
anomaly_set = {k:v[mask_anomaly] for (k,v) in dataset_r.items()}

# counts
total_normals = len(normal_set["target"])
total_anomalies = len(anomaly_set["target"])

num_train = total_normals // 2
num_test_normal = total_normals - num_train   # the remaining half
num_test_anomaly = total_anomalies            # use all anomalies

# split
training_set, remaining_set_normal = random_selection(normal_set, num_train)
testing_set_normal, _ = random_selection(remaining_set_normal, num_test_normal)
testing_set_anomaly, _ = random_selection(anomaly_set, num_test_anomaly)

testing_set = {k: torch.cat((testing_set_normal[k], testing_set_anomaly[k]), dim=0) for k in testing_set_normal}

system_param = {
    "P": 4, #int(os.environ['P']), # Degree of encoding
    "num_site": dataset['input'].shape[-1]
}

if rawdata == 'creditcard' or rawdata == 'satellite':
    num_batch = 512
else:
    num_batch = 32

opt_method = {
    "num_batch": num_batch,
    "num_epochs": 40,#(10000 * num_batch) // num_train, #200000//num_train,
    "lr": 1e-3,
    "device": device,
    "reg_param": 0.1,
    "chi": 5
}



opt_method['detail_name'] = f'{rawdata}_{system_param['P']}'


num_samples = 20
results_list = []

if __name__ == "__main__":
    for _ in range(num_samples):
        #===========================#
        #       Data splitting
        #===========================#
        mask_normal = dataset_r["target"] == 0
        mask_anomaly = dataset_r["target"] == 1

        normal_set = {k:v[mask_normal] for (k,v) in dataset_r.items()}
        anomaly_set = {k:v[mask_anomaly] for (k,v) in dataset_r.items()}

        # counts
        total_normals = len(normal_set["target"])
        total_anomalies = len(anomaly_set["target"])

        num_train = total_normals // 2
        num_test_normal = total_normals - num_train   # the remaining half
        num_test_anomaly = total_anomalies            # use all anomalies

        # split
        training_set_normal, remaining_set_normal = random_selection(normal_set, num_train)

        testing_set_normal, _ = random_selection(remaining_set_normal, num_test_normal)

        training_set = training_set_normal
        testing_set_anomaly, _ = random_selection(anomaly_set, num_test_anomaly)
        testing_set = {k: torch.cat((testing_set_normal[k], testing_set_anomaly[k]), dim=0) for k in testing_set_normal}
        #===========================#
        #     Model training
        #===========================#
        model, epoch_loss = training(training_set, system_param, opt_method)
        roc, auprc = testing(testing_set, model)
        results = {
                'reg': opt_method['reg_param'],
                'lr': opt_method['lr'],
                'num_batch': num_batch,
                'loss': epoch_loss,
                'roc': roc,
                'auprc': auprc
            }
        results_list.append(results)
    results_df = pd.DataFrame(results_list)
    output_filename = f'acc/TNAD_acc_{opt_method['detail_name']}.csv'
    results_df.to_csv(output_filename, index=False)
    print(f"\n✅ Results successfully saved to '{output_filename}'")
    print("\nDataFrame Head:")
    print(results_df.head())