import torch
import torch.nn as nn
from torch.func import vmap  # PyTorch >= 2.0; for older use: from functorch import vmap
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import average_precision_score, precision_recall_curve
import os
import pandas as pd
import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score


def run_benchmarks(train_tensor, test_tensor, test_labels):
    """
    train_tensor: (N, D) - Normal training samples
    test_tensor:  (M, D) - Test samples (mixed)
    test_labels:  (M,)   - Ground truth (0=Normal, 1=Anomaly)
    """
    
    X_train = train_tensor.detach().cpu().numpy()
    X_test = test_tensor.detach().cpu().numpy()
    y_test = test_labels.detach().cpu().numpy()

    n_features = X_train.shape[1]

    # --- OC-SVM: Grid Sweep (Supervised Advantage) ---
    print("Running OC-SVM Grid Sweep...")
    gammas = [1/2**i for i in range(1, 11)] 
    nus = [0.01, 0.1]
    
    best_ocsvm_metrics = {"auprc": -1.0}

    for g in gammas:
        for n in nus:
            # print(f'Running gamma = {g}, nu = {n}')
            ocsvm = OneClassSVM(kernel='rbf', gamma=g, nu=n)
            ocsvm.fit(X_train)
            scores = -ocsvm.decision_function(X_test)
            
            auc_roc = roc_auc_score(y_test, scores)
            auprc = average_precision_score(y_test, scores)
            
            if auprc > best_ocsvm_metrics["auprc"]:
                # --- Calculate OCSVM Parameters ---
                # Parameters = (Support Vectors * Features) + Dual Coefs + Intercept
                n_sv = ocsvm.support_vectors_.shape[0]
                n_params = (n_sv * n_features) + n_sv + 1
                
                best_ocsvm_metrics = {
                    "auc_roc": auc_roc,
                    "auprc": auprc,
                    "gamma": g,
                    "nu": n,
                    "n_params": n_params,
                    "n_sv": n_sv # Useful to see just the count of SVs
                }

    # --- Isolation Forest: Original Paper Params ---
    print("Running Isolation Forest (100 trees, 256 sub-samples)...")
    iso_forest = IsolationForest(
        n_estimators=100, 
        max_samples=min(256, len(X_train)), 
        random_state=42,
        n_jobs=1
    )
    iso_forest.fit(X_train)
    
    if_scores = -iso_forest.decision_function(X_test)
    
    # --- Calculate IF Parameters ---
    # Parameters = Sum of all nodes in all trees
    total_nodes = sum(t.tree_.node_count for t in iso_forest.estimators_)

    if_metrics = {
        "auc_roc": roc_auc_score(y_test, if_scores),
        "auprc": average_precision_score(y_test, if_scores),
        "n_params": total_nodes
    }

    # --- Summary Table ---
    print("\n" + "="*80)
    print(f"{'Model':<20} | {'AUC-ROC':<10} | {'AUPRC':<10} | {'# Params':<12}")
    print("-" * 80)
    print(f"{'OC-SVM (Best)':<20} | {best_ocsvm_metrics['auc_roc']:.4f}     | {best_ocsvm_metrics['auprc']:.4f}     | {best_ocsvm_metrics['n_params']:<12}")
    print(f"{'Isolation Forest':<20} | {if_metrics['auc_roc']:.4f}     | {if_metrics['auprc']:.4f}     | {if_metrics['n_params']:<12}")
    print("="*80)
    print(f"OC-SVM Winning Params: gamma={best_ocsvm_metrics['gamma']}, nu={best_ocsvm_metrics['nu']}")
    print(f"OC-SVM Support Vectors: {best_ocsvm_metrics['n_sv']}")
    
    return best_ocsvm_metrics, if_metrics

def get_rawdata(filename):
    if filename == "creditcard":
        csv_rawdata = pd.read_csv(f'{filename}.csv')
        column = ["Time"]
        column.extend([f"V{i}" for i in range(1,29)])
        column.extend(["Amount"])
        input_ds = torch.tensor(csv_rawdata[column].values, dtype=torch.float32)
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
        # finfo = torch.finfo(X.dtype)
        # u = X.clamp(min=finfo.eps, max=1.0 - finfo.eps)
        # X_qn = torch.sqrt(torch.tensor(2.0, device=X.device, dtype=X.dtype)) * torch.erfinv(2.0 * u - 1.0)
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


#==================================
#==================================

# WORKING SPACE

#==================================
#==================================

rawdata = 'creditcard' # "satellite", "lymphography" #"creditcard", "wine", 'thyroid'

local_rank = int(os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK", 0))
if torch.cuda.is_available():
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
else:
    device = torch.device("cpu")


dataset = get_rawdata(rawdata)

if rawdata == 'lymphography' or rawdata == 'satellite':
    encode = None #'z-score' #'quantile' #None
else:
    encode = 'quantile' #'quantile'

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
    "num_site": dataset['input'].shape[-1]
}

if rawdata == 'creditcard' or rawdata == 'satellite':
    num_batch = 512
else:
    num_batch = 32

opt_method = {
    "num_batch": num_batch,
    "num_epochs": 20,#(10000 * num_batch) // num_train, #200000//num_train,
    "lr": 5e-4,
    "device": device,
    "reg_param": 0.1,
    "chi": 5
}



opt_method['detail_name'] = f'{rawdata}'


num_samples = 20
results_list = []

if __name__ == "__main__":
    for _ in range(num_samples):
        #===========================#
        #       Data splitting
        #===========================#
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
        training_set_normal, remaining_set_normal = random_selection(normal_set, num_train)

        testing_set_normal, _ = random_selection(remaining_set_normal, num_test_normal)

        training_set = training_set_normal
        testing_set_anomaly, _ = random_selection(anomaly_set, num_test_anomaly)
        testing_set = {k: torch.cat((testing_set_normal[k], testing_set_anomaly[k]), dim=0) for k in testing_set_normal}
        #===========================#
        #     Model training
        #===========================#
        # if rawdata == 'thyroid':
        #     testing_set['target'] = 1 - testing_set['target']
        results_oc, results_if = run_benchmarks(training_set['input'], testing_set['input'], testing_set["target"])
        # model, epoch_loss = training(training_set, system_param, opt_method)
        # roc, auprc = testing(testing_set, model)
        results = {
                'reg': opt_method['reg_param'],
                'lr': opt_method['lr'],
                'num_batch': num_batch,
                'svm_roc': results_oc['auc_roc'],
                'svm_auprc': results_oc['auprc'],
                'if_roc': results_if['auc_roc'],
                'if_auprc': results_if['auprc'],
                'svm_n_params': (results_oc["n_params"],results_oc["n_sv"]),
                'if_n_params': results_if["n_params"]
            }
        results_list.append(results)
    results_df = pd.DataFrame(results_list)
    output_filename = f'acc/OCSVM_IF_acc_{opt_method['detail_name']}.csv'
    results_df.to_csv(output_filename, index=False)
    print(f"\n✅ Results successfully saved to '{output_filename}'")
    print("\nDataFrame Head:")
    print(results_df.head())