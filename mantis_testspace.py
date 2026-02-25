from mantis_module import *

rawdata = 'creditcard' # "satellite", "lymphography" #"creditcard", "wine", 'thyroid'

#======== Device ========#

local_rank = int(os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK", 0))
if torch.cuda.is_available():
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
else:
    device = torch.device("cpu")

#======== Load dataset ========#

dataset_rrr = get_rawdata(rawdata)
dataset = dataset_rrr
feature_selection = [] #[1,2,3,4,5,6,7,8,9,10,11,16,17,27,28]

if len(feature_selection) > 0:
    input_selected = dataset_rrr['input']
    dataset['input'] = input_selected[:,feature_selection]

# encode = 'quantile' #'quantile' #'min-max', 'z-score', 'l2', 'quantile', 'robust'
if rawdata == 'lymphography' or rawdata == 'satellite':
    encode = None
else:
    encode = 'quantile'

dataset_r = preprocessing(dataset, encode)

num_samples = 20 # Number of results reproduction
results_list = []


if __name__ == "__main__":
    
    system_param = {
        "M": 30, #int(os.environ['M']) # Number of MPOs
        "P": 2, #int(os.environ['P']) # Degree of encoding
        "delta_P": 1, #In work, we fixed delta_P = 1
        "num_site": dataset['input'].shape[-1]
    }

    if rawdata == 'creditcard' or rawdata == 'satellite':
        num_batch = 512
    else:
        num_batch = 32


    opt_method = {
        "opt_mode": "global_dl", # "global", "local", "global_dl"
        "encoding": "fourier", # "fourier"
        "num_batch": num_batch,
        "num_sweeps": 3,
        "sweep_size": 2,
        "lr": 0.01,
        "device": device,
        "set_initial": False, # Set all zeros or not
        "loss_fn": "log", # "fubini", "cross", "mse", "hamming", "log", "local_0", "pauli_z"
        "test_fn": "amplitude", # 'amplitude', 'local_0', 'hamming', "pauli_z"
        "L-root": False,
        "num_measure": 100,
        "rawdata": rawdata,
    }

    include_anomaly = False # This experiment will not include anomalies

    λc = 0.01
    λ = 0.001
    best_param = {"c": λc, "theta": λ}
    print("Best params:", best_param)
    
    opt_method["reg_param"] = best_param
    opt_method["lr"] = 1e-2
    opt_method["num_batch"] = num_batch

    #========== Name of an experiment==========#
    if encode != 'quantile':
        opt_method['detail_name'] = f'{rawdata}_({system_param['M']},{system_param['P']})_{opt_method['loss_fn']}_{opt_method["encoding"]}_{encode}'
    else:
        if include_anomaly:
            opt_method['detail_name'] = f'{rawdata}_({system_param['M']},{system_param['P']})_{opt_method['loss_fn']}_{opt_method["encoding"]}_w_neg'
        else:
            opt_method['detail_name'] = f'{rawdata}_({system_param['M']},{system_param['P']})_{opt_method['loss_fn']}_{opt_method["encoding"]}'
    
    if λc == 0 and λ == 0:
        opt_method['detail_name'] += '_noreg'

    if len(feature_selection) > 0:
        opt_method['detail_name'] += '_FI'
    
    print(opt_method['detail_name'])
    
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

        if include_anomaly:
            num_train_anomaly = total_anomalies // 2
            training_set_anomaly, remaining_set_anomaly = random_selection(anomaly_set, num_train_anomaly)
            training_set = {k: torch.cat((training_set_normal[k], training_set_anomaly[k]), dim=0) for k in training_set_normal}
            num_test_anomaly = total_anomalies - num_train_anomaly
        else:
            training_set = training_set_normal
        testing_set_anomaly, _ = random_selection(anomaly_set, num_test_anomaly)
        testing_set = {k: torch.cat((testing_set_normal[k], testing_set_anomaly[k]), dim=0) for k in testing_set_normal}
        #===========================#
        #     Model training
        #===========================#
        opt_method["num_epochs"] = (15000 * num_batch) // num_train
        model, epoch_loss = training(training_set, system_param, opt_method)
        acc = testing(testing_set, model)
        results = {
                'reg': best_param,
                'lr': 1e-2,
                'num_batch': num_batch,
                'loss': epoch_loss,
                'roc': acc['roc_auc'],
                'auprc': acc['auprc']
            }
        results_list.append(results)
        print(f'AUC-ROC = {acc['roc_auc']}')
        print(f'AUPRC = {acc['auprc']}')
    results_df = pd.DataFrame(results_list)
    output_filename = f'acc/acc_{opt_method['detail_name']}.csv'
    results_df.to_csv(output_filename, index=False)
    print(f"\n✅ Results successfully saved to '{output_filename}'")
    print("\nDataFrame Head:")
    print(results_df.head())
