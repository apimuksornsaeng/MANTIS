from mantis_module import *

torch.set_num_threads(min(8, os.cpu_count() or 8))

rawdata = 'thyroid' # "satellite", "lymphography" #"creditcard", "wine", 'thyroid'

local_rank = int(os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK", 0))
if torch.cuda.is_available():
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
else:
    device = torch.device("cpu")

# dataset = {'input': torch.randn(50,3), 'target': torch.randn(50)}


dataset = get_rawdata(rawdata)

#'min-max', 'z-score', 'l2', 'quantile', 'robust'
encode = 'quantile'

dataset_r = preprocessing(dataset, encode)

# plt.figure()
# plt.scatter(dataset['input'][:,1],dataset_r['input'][:,1])
# plt.xlabel(r'$V_2$')
# plt.ylabel(r'$\tilde{V}_2$')
# plt.savefig('normalize_data.pdf')

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
# start_train = 0
# mask = training_set_r["target"] == 0

# training_set = {k:v[mask] for (k,v) in training_set_r.items()}

# anomaly_mask = training_set_r["target"] == 1

# anomaly_set = {k:v[anomaly_mask] for (k,v) in training_set_r.items()}

# print(training_set['input'].shape)
# print(training_set['target'].shape)

system_param = {
    "M": int(os.environ['M']), #20, #2**int(os.environ['M']),
    "P": int(os.environ['P']), # Degree of encoding
    "num_site": dataset['input'].shape[-1]
}

num_batch = 32

opt_method = {
    "opt_mode": "global_dl", # "global", "local", "global_dl"
    "encoding": "fourier", # "fourier"
    "num_batch": num_batch,
    "num_epochs": (2000 * num_batch) // num_train, #200000//num_train,
    "num_sweeps": 3,
    "sweep_size": 2,
    "lr": 0.01,
    "device": device,
    "set_initial": False, # Set all zeros or not
    "loss_fn": "log", # "fubini", "cross", "mse", "hamming", "log", "local_0", "pauli_z"
    "test_fn": "amplitude", # 'amplitude', 'local_0', 'hamming', "pauli_z"
    "L-root": False,
    "num_measure": 100
}

opt_method['detail_name'] = f'{rawdata}_({system_param['M']},{system_param['P']})_{opt_method['loss_fn']}_{opt_method["encoding"]}'


num_samples = 100
results_list = []
search_space = [(reg_c,reg_theta_m,reg_theta_p,lr,bs) 
                        for reg_c in [0.0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]
                        for reg_theta_m in [0.0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]
                        for reg_theta_p in [0.0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]
                        for lr in [1e-2]
                        for bs in [32]]

# def bayesian_optimization(trial):
#     reg_c = trial.suggest_float("c", 1e-4, 1.0, log=True)
#     reg_theta_m = trial.suggest_float("theta_m", 1e-4, 1.0, log=True)
#     reg_theta_p = trial.suggest_float("theta_p", 1e-4, 1.0, log=True)
#     reg_param = {'c': reg_c, 'theta_m': reg_theta_m, 'theta_p': reg_theta_p}
#     opt_method["reg_param"] = reg_param
#     opt_method["lr"] = 1e-2
#     opt_method["num_batch"] = 32
#     num_samples = 50
#     mean_roc = 0
#     for _ in range(num_samples):
#         model, _ = new_training(training_set, system_param, opt_method)
#         acc = testing(testing_set, model)
#         roc = acc['roc_auc']
#         mean_roc += roc
#     return mean_roc/num_samples

#===========================#
#           GPU setup
#===========================#


def _set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)

def _worker_run(gpu_id, run_indices, trial_params, training_set, testing_set, system_param, base_opt_method, out_q):
    """
    One process pinned to a single GPU runs several realizations sequentially.
    """
    # Pin this process to the given GPU
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu_id)
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    mean_roc_local = 0.0
    count_local = 0

    for idx in run_indices:
        # make each realization reproducible but distinct
        _set_seed(10_000 + idx)

        # copy / prepare opt_method for this run
        opt_method = dict(base_opt_method)  # shallow copy is OK if values are immutables / small dicts
        opt_method["reg_param"] = dict(trial_params["reg_param"])  # copy nested dict too

        # --- your training & testing ---
        model, _ = new_training(training_set, system_param, opt_method)
        acc = testing(testing_set, model)
        roc = float(acc["roc_auc"])

        mean_roc_local += roc
        count_local += 1

        # optional: free GPU memory between runs
        if torch.cuda.is_available():
            del model
            torch.cuda.empty_cache()

    # push partial sums to the parent
    out_q.put((mean_roc_local, count_local))


def bayesian_optimization(trial):
    # ----- sample hyperparams ONCE (Optuna stays in the parent process) -----
    reg_c = trial.suggest_float("c", 1e-3, 1e-1, log=True)
    reg_theta_m = trial.suggest_float("theta_m", 1e-3, 1e-1, log=True)
    reg_theta_p = trial.suggest_float("theta_p", 1e-3, 1e-1, log=True)

    trial_params = {
        "reg_param": {"c": reg_c, "theta_m": reg_theta_m, "theta_p": reg_theta_p},
    }

    base_opt_method = dict(opt_method) if "opt_method" in globals() else {}
    base_opt_method["reg_param"] = trial_params["reg_param"]
    base_opt_method.setdefault("lr", 1e-2)
    base_opt_method.setdefault("num_batch", 32)

    num_samples = 5
    # choose available GPUs (default to 4 ids: 0..3, but trim to available)
    requested_gpu_ids = [0] #[0, 1, 2, 3]
    if torch.cuda.is_available():
        available = torch.cuda.device_count()
        gpu_ids = [g for g in requested_gpu_ids if g < available]
        if not gpu_ids:
            gpu_ids = [0]  # fall back to single GPU 0
    else:
        gpu_ids = [None]  # CPU fallback

    num_workers = len(gpu_ids)

    # split the realizations across workers (nearly even chunks)
    indices = list(range(num_samples))
    chunk_size = math.ceil(num_samples / num_workers)
    chunks = [indices[i:i + chunk_size] for i in range(0, len(indices), chunk_size)]

    # multiprocessing setup
    ctx = mp.get_context("spawn")  # safe with CUDA
    out_q = ctx.Queue()
    procs = []

    # (Optional) set a global base seed in parent for reproducibility of the split
    _set_seed(12345)

    for worker_id, run_indices in enumerate(chunks):
        gpu_id = gpu_ids[worker_id if worker_id < len(gpu_ids) else -1]
        p = ctx.Process(
            target=_worker_run,
            args=(gpu_id, run_indices, trial_params, training_set, testing_set, system_param, base_opt_method, out_q),
        )
        p.start()
        procs.append(p)

    # gather partial sums
    total_sum = 0.0
    total_count = 0
    for _ in procs:
        part_sum, part_count = out_q.get()  # blocks until worker posts
        total_sum += part_sum
        total_count += part_count

    # join workers
    for p in procs:
        p.join()

    # safety check
    if total_count == 0:
        # If nothing ran (shouldn't happen), return a sentinel bad score
        return float("-inf")

    mean_roc = total_sum / total_count
    return mean_roc



if __name__ == "__main__":
    # study = optuna.create_study(direction="maximize",
    #                             sampler=optuna.samplers.TPESampler(seed=1337),
    #                             pruner=optuna.pruners.MedianPruner(n_warmup_steps=5))
    # study.optimize(bayesian_optimization, n_trials=10, gc_after_trial=True)
    # best_param = study.best_params
    # print("\nBest CV AUC:", study.best_value)
    best_param = {"c": 0.01, "theta_m": 0.002, "theta_p": 0.007}
    print("Best params:", best_param)
    opt_method["reg_param"] = best_param
    opt_method["lr"] = 1e-2
    opt_method["num_batch"] = 32
    for _ in range(num_samples):
        model, epoch_loss = new_training(training_set, system_param, opt_method)
        acc = testing(testing_set, model)
        results = {
                'reg': best_param,
                'lr': 1e-2,
                'num_batch': num_batch,
                'loss': epoch_loss,
                'roc': acc['roc_auc']
            }
        results_list.append(results)
    results_df = pd.DataFrame(results_list)
    output_filename = f'acc/acc_{opt_method['detail_name']}.csv'
    results_df.to_csv(output_filename, index=False)
    print(f"\n✅ Results successfully saved to '{output_filename}'")
    print("\nDataFrame Head:")
    print(results_df.head())


# for param_search in search_space:
#     reg_c,reg_theta_m,reg_theta_p,lr,bs = param_search
#     reg_param = {'c': reg_c, 'theta_m': reg_theta_m, 'theta_p': reg_theta_p}
#     opt_method["reg_param"] = reg_param
#     opt_method["lr"] = lr
#     opt_method["num_batch"] = bs
#     for _ in range(num_samples):
#         model, epoch_loss = new_training(training_set, system_param, opt_method)
#         # find_optimal_threshold(model, validate_set)
#         acc = testing(testing_set, model)
#         results = {
#             'reg': reg_param,
#             'lr': lr,
#             'num_batch': bs,
#             'loss': epoch_loss,
#             'roc': acc['roc_auc'],
#             'acc': acc['acc']
#         }
#         results_list.append(results)
#         print(results)
#         del epoch_loss, acc, model
#         if torch.cuda.is_available():
#             torch.cuda.empty_cache()
# results_df = pd.DataFrame(results_list)
# output_filename = f'acc/acc_{opt_method['detail_name']}.csv'
# results_df.to_csv(output_filename, index=False)
# print(f"\n✅ Results successfully saved to '{output_filename}'")
# print("\nDataFrame Head:")
# print(results_df.head())

