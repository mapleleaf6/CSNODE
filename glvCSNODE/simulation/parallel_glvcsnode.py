import numpy as np
import os 

import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
from copy import deepcopy
import time
import pickle
from tqdm import tqdm
import traceback
from torchdiffeq import odeint
import concurrent.futures
import math
import torch.nn.utils.parametrize as P
import torch.nn.functional as F


device = torch.device('cpu')


from rpy2.robjects import r
from rpy2.robjects import pandas2ri
pandas2ri.activate()


class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.wait = 0
        self.stopped = False

    def __call__(self, current_loss):
        if current_loss + self.min_delta < self.best_loss:
            self.best_loss = current_loss
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped = True


def calculate_rmse_rrmse(A_pred, r_pred, A_true, r_true):

    A_rmse = np.sqrt(np.mean(np.square(A_pred - A_true)))
    

    A_norm = np.sqrt(np.mean(np.square(A_true)))
    

    A_rrmse = round(A_rmse / A_norm, 10)
    

    r_rmse = np.sqrt(np.mean(np.square(r_pred - r_true)))
    

    r_norm = np.sqrt(np.mean(np.square(r_true)))
    

    r_rrmse = round(r_rmse / r_norm, 10)
    

    return A_rrmse, r_rrmse


def train_and_evaluate_model_shoot(
    train_indices, val_indices, begin_shoot, end_shoot, traj_batch_t, device,
    lambda_factor, cross_section_p, epochs, lr):

    train_begin_shoot = begin_shoot[train_indices]
    train_end_shoot = end_shoot[train_indices]

    val_begin_shoot = begin_shoot[val_indices]
    val_end_shoot = end_shoot[val_indices]

    random_seed = 66
    num = 0

    while 1:
        try:

            np.random.seed(random_seed)
            torch.manual_seed(random_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(random_seed)

            d = begin_shoot.shape[-1]
            A_init_np = np.random.normal(loc=0.0, scale=0.1, size=(d, d))
            r_init_np = np.random.uniform(low=0.0, high=1.0, size=d)


            OdeModel = glv_matrix(A_init_np, r_init_np)



            optimizer = optim.LBFGS(OdeModel.parameters(), lr=lr, tolerance_grad=1e-32, tolerance_change=1e-32, line_search_fn = 'strong_wolfe')


            early_stopper = EarlyStopping(patience=20, min_delta=1e-11)

            range_function = get_range_function(epochs, False)


            train_model_shoot_SmoothL1_cv(
                OdeModel, optimizer, range_function, train_begin_shoot, train_end_shoot, traj_batch_t, device, cross_section_p, early_stopper, lambda_factor
            )



            with torch.no_grad():
                
                val_pred = odeint(OdeModel, val_begin_shoot, traj_batch_t[:2]).to(device)
                val_loss = loss_fn(val_pred[-1], val_end_shoot)

            break

        except Exception as e:

            print(f"An error occurred: {e}")
            random_seed += 1
            num += 1

            if num >= 5:
                val_loss = np.float('inf')
                break

    return val_loss


def process_fold_shoot(fold_indices, begin_shoot, end_shoot, traj_batch_t, device,
                 lambda_factor, cross_section_p, epochs, lr):
    train_indices, val_indices = fold_indices
    val_loss = train_and_evaluate_model_shoot(
        train_indices, val_indices, begin_shoot, end_shoot, traj_batch_t, device,
        lambda_factor, cross_section_p, epochs, lr
    )
    return val_loss



def cross_validation_shoot(traj_tensor, traj_batch_t, cross_section_p, lambda_factors, k_fold=5, epochs=1000, device='cpu', lr = 0.1):


    begin_shoot = traj_tensor[:-1, :]
    end_shoot = traj_tensor[1:, :]

    end_shoot[begin_shoot == 0] = 0

    total_samples = begin_shoot.shape[0]


    from sklearn.model_selection import KFold
    kf = KFold(n_splits=k_fold, shuffle=False)
    folds = [(train_indices, val_indices) for train_indices, val_indices in kf.split(np.arange(total_samples))]

    best_avg_val_loss = float('inf')
    best_lambda_factor = None

    for lambda_factor in lambda_factors:
        val_losses = []



        with concurrent.futures.ProcessPoolExecutor(max_workers=k_fold) as executor:

            futures = [
                executor.submit(
                    process_fold_shoot,
                    fold_indices,
                    begin_shoot,
                    end_shoot,
                    traj_batch_t,
                    device,
                    lambda_factor,
                    cross_section_p, 
                    epochs,
                    lr
                )
                for fold_indices in folds
            ]


            for future in concurrent.futures.as_completed(futures):
                val_losses.append(future.result())

        avg_val_loss = np.mean(val_losses)
        print(f"Lambda: {lambda_factor}, Avg Validation Loss: {avg_val_loss}")
        if avg_val_loss < best_avg_val_loss:
            best_avg_val_loss = avg_val_loss
            best_lambda_factor = lambda_factor
    print(f"Best Lambda Factor in First Stage: {best_lambda_factor}")
    return best_lambda_factor



class NegSoftplusDiag(nn.Module):


    def __init__(self):
        super().__init__()

    def forward(self, X: torch.Tensor) -> torch.Tensor:

        diag_raw = torch.diag(X)                       

        diag_neg = - F.softplus(diag_raw) 

        return X - torch.diag(diag_raw) + torch.diag(diag_neg)


class SoftplusPos(nn.Module):


    def __init__(self):
        super().__init__()


    def forward(self, v: torch.Tensor) -> torch.Tensor:
        
        return F.softplus(v)



class glv_matrix(nn.Module):

    def __init__(self, A, r):

        super(glv_matrix, self).__init__()

        self.A = torch.nn.Parameter(torch.tensor(A, dtype=torch.float), requires_grad = True)
        self.r  = torch.nn.Parameter(torch.tensor(r, dtype=torch.float),  requires_grad = True)

        self.d = len(r)


        P.register_parametrization(self, "A", NegSoftplusDiag())  
        P.register_parametrization(self, "r", SoftplusPos())    

    def penalty(self, y):

        I = (y > 0).int()

        return torch.mul(y @ self.A.T + self.r, I)

    def forward(self, t, y):
        return torch.mul(y @ self.A.T + self.r, y)  


def get_range_function(epochs, tqdm_verbose):
    if tqdm_verbose:
        range_function = tqdm(range(epochs))
    else:
        range_function = range(epochs)

    return range_function


def loss_fn(pred, y):
    return torch.mean(torch.square(pred - y))


def train_model_shoot_SmoothL1_cv(OdeModel, optimizer, range_function, begin_shoot, end_shoot, traj_batch_t, device, cross_section_p, early_stopper, lambda_factor):

    d = begin_shoot.shape[-1]

    for e in range_function:

        def closure():
            optimizer.zero_grad()


            pred_obs = odeint(OdeModel, begin_shoot, traj_batch_t[:2]).to(device)

            loss = loss_fn(pred_obs[-1], end_shoot)

            loss += lambda_factor * torch.mean(torch.square(OdeModel.penalty(cross_section_p)))

            loss += 1e-3 * torch.mean(torch.square(OdeModel.A[~torch.eye(d, dtype=bool)]))


            loss.backward()
            return loss


        optimizer.step(closure)
        loss = closure().item()

        early_stopper(loss)
        if early_stopper.stopped:

            break


def train_model_shoot_SmoothL1(OdeModel, optimizer, range_function, traj_tensor, traj_batch_t, device, cross_section_p, early_stopper, lambda_factor):

    begin_shoot = traj_tensor[:-1, :]
    end_shoot = traj_tensor[1:, :]

    end_shoot[begin_shoot == 0] = 0

    d = begin_shoot.shape[-1]

    for e in range_function:



        def closure():
            optimizer.zero_grad()


            pred_obs = odeint(OdeModel, begin_shoot, traj_batch_t[:2]).to(device)
            
            loss = loss_fn(pred_obs[-1], end_shoot)

            loss += lambda_factor * torch.mean(torch.square(OdeModel.penalty(cross_section_p))) 

            loss += 1e-3 * torch.mean(torch.square(OdeModel.A[~torch.eye(d, dtype=bool)]))


            loss.backward()
            return loss


        optimizer.step(closure)
        loss = closure().item()

        early_stopper(loss)
        if early_stopper.stopped:
            print("Training stopped due to early stopping")
            break



def node_process_main(d, lr=0.01, epochs=1000, tqdm_verbose=False):




    traj_folder = "data_generate"
    data_folder = traj_folder


    data_file = f"{data_folder}/obs.rds"
    readRDS = r['readRDS']
    data = readRDS(data_file)

    data = dict(zip(data.names, map(list,list(data))))
    
    obs_time = np.array(data['obs_time']) 


    A_true = np.array(data['A'])
    r_true = np.array(data['r'])


    select_subject_obs = np.array(data['select_subject_obs_list'])[0]

    non_selected_steady_state_absolute = np.array(data['non_selected_steady_state_absolute']).T


    np.random.seed(66)
    torch.manual_seed(66)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(66)



    traj_batch_t = torch.tensor(obs_time, dtype=torch.float).to(device)
    traj_tensor = torch.tensor(select_subject_obs, dtype=torch.float).to(device).squeeze()


    cross_section_p = torch.tensor(non_selected_steady_state_absolute, dtype=torch.float).to(device)


    lambda_factors = [1e-3, 1e-2, 1e-1, 1, 1e1, 1e2]

    start_time = time.time()


    lambda_factor_first_stage = cross_validation_shoot(traj_tensor, traj_batch_t, cross_section_p, lambda_factors, lr=lr)


    first_stage_time_cv = time.time()

    random_seed = 66

    num = 0
    while 1:
        try:

            np.random.seed(random_seed)
            torch.manual_seed(random_seed)

            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(random_seed)


            A_init_np = np.random.normal(loc=0.0, scale=0.1, size=(d, d))
            r_init_np = np.random.uniform(low=0.0, high=1.0, size=d)
            


            OdeModel = glv_matrix(A_init_np, r_init_np)
            optimizer = optim.LBFGS(OdeModel.parameters(), lr=lr, tolerance_grad=1e-32, tolerance_change=1e-32, line_search_fn = 'strong_wolfe')

            range_function = get_range_function(epochs, tqdm_verbose)

            early_stopper = EarlyStopping(patience=20, min_delta=1e-11)

            print()

            train_model_shoot_SmoothL1(OdeModel, optimizer, range_function, traj_tensor.squeeze(), traj_batch_t, device, cross_section_p, early_stopper, lambda_factor = lambda_factor_first_stage)

            A_node_first_stage = deepcopy(OdeModel.A.detach().numpy())
            r_node_first_stage = deepcopy(OdeModel.r.detach().numpy())

            A_node_first_stage_rrmse = np.sqrt(np.mean(np.square(A_node_first_stage - A_true))) / np.sqrt(np.mean(np.square(A_true)))
            r_node_first_stage_rrmse = np.sqrt(np.mean(np.square(r_node_first_stage - r_true))) / np.sqrt(np.mean(np.square(r_true)))

            A_node_first_stage_rmse = np.sqrt(np.mean(np.square(A_node_first_stage - A_true)))
            r_node_first_stage_rmse = np.sqrt(np.mean(np.square(r_node_first_stage - r_true)))

            print("First Stage Node:")
            print("A_node_first_stage_est_rrmse: ", A_node_first_stage_rrmse)
            print("r_node_first_stage_est_rrmse: ", r_node_first_stage_rrmse)
            print("A_node_first_stage_est_rmse: ", A_node_first_stage_rmse)
            print("r_node_first_stage_est_rmse: ", r_node_first_stage_rmse)
            print()

            break

        except Exception as e:

            print(f"An error occurred: {e}")
            traceback.print_exc()  
            
            random_seed += 1
            num += 1

            if num > 20:
                break        
        
        

    first_stage_time = time.time()


    node_fit_time = time.time()

    A_node = OdeModel.A.detach().numpy()
    r_node = OdeModel.r.detach().numpy()


    A_node_rrmse = np.sqrt(np.mean(np.square(A_node - A_true))) / np.sqrt(np.mean(np.square(A_true)))
    r_node_rrmse = np.sqrt(np.mean(np.square(r_node - r_true))) / np.sqrt(np.mean(np.square(r_true)))

    A_node_rmse = np.sqrt(np.mean(np.square(A_node - A_true)))
    r_node_rmse = np.sqrt(np.mean(np.square(r_node - r_true)))

    print("node:")
    print("A_node_est_rrmse: ", A_node_rrmse)
    print("r_node_est_rrmse: ", r_node_rrmse)
    print("A_node_est_rmse: ", A_node_rmse)
    print("r_node_est_rmse: ", r_node_rmse)
    print("total_fit_time: ", node_fit_time - start_time)


    results = dict()
    results['select_subject_obs'] = select_subject_obs
    results['obs_time'] = obs_time
    results['A_node'] = A_node
    results['r_node'] = r_node

    results['A_node_rrmse'] = A_node_rrmse
    results['r_node_rrmse'] = r_node_rrmse
    results['A_node_rmse'] = A_node_rmse
    results['r_node_rmse'] = r_node_rmse

    results['A_init'] = A_init_np
    results['r_init'] = r_init_np


    results['A_true'] = A_true
    results['r_true'] = r_true

    
    results['A_node_first_stage'] = A_node_first_stage
    results['r_node_first_stage'] = r_node_first_stage

    results['A_node_first_stage_rrmse'] = A_node_first_stage_rrmse
    results['r_node_first_stage_rrmse'] = r_node_first_stage_rrmse
    results['A_node_first_stage_rmse'] = A_node_first_stage_rmse
    results['r_node_first_stage_rmse'] = r_node_first_stage_rmse

    results['lambda_factor_first_stage'] = lambda_factor_first_stage


    results['first_stage_time_cv'] = first_stage_time_cv
    results['first_stage_time'] = first_stage_time
    results['node_fit_time'] = node_fit_time


    file_folder = data_folder
    file_folder = file_folder
    file_path = os.path.join(file_folder, "results_glvcsnode.pkl")


    if not os.path.exists(file_folder):
        os.makedirs(file_folder)

    with open(file_path, 'wb') as f:
        pickle.dump(results, f)

    print("save the results successfully!")


                


def main():

    d = 15

    node_process_main(d=d)


if __name__ == "__main__":
    main()  