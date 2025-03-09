import os, sys, shutil
#sys.path.append('/projects/b1139/environments/emod_torch_tobias/lib/python3.8/site-packages/')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json
sys.path.append("../")
import manifest as manifest

import torch
from torch import tensor
torch.set_default_dtype(torch.float64)
from botorch.models import SingleTaskGP
from botorch.models.transforms import Normalize, Standardize
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.constraints import Interval, GreaterThan, LessThan
from gpytorch.likelihoods import GaussianLikelihood

def get_script_path():
    return os.path.dirname(os.path.realpath(sys.argv[0]))


def fit_GP_to_objective(exp='',site='',metric='',n_prior=0):
    path_to_here = get_script_path()
    # skip no_blood objectives
    if(metric=="no_blood"):
        print("Not supported for 'no_blood' objective.")
        return
    
    print(f"site: {site} metric: {metric}",flush=True)
    
    # Load acquisition function details
    with open(f'{path_to_here}/output/{exp}/TurboThompson.json') as f:
        data = json.load(f)
    df = pd.DataFrame({'value': data})
    
    print(df)
    
    # Get batch size
    batch_size=df.at['batch_size', 'value']
    
    # Load input X - parameters
    X = torch.load(os.path.join(path_to_here,'output',exp,'X.pt'))
    
    # Load output Y - scores
    Y = pd.read_csv(os.path.join(path_to_here,'output',exp,'all_LL.csv'))
    
    # filter out rows with NULL LL or with no_blood score > 0
    missing_ll_rows = Y[Y['ll'].isnull()]
    bloodless = Y[(Y['ll']>0) & (Y['metric']=="no_blood")]
    
    # Getting unique combinations of 'round' and 'parameter'
    missing = pd.concat([bloodless[['round','param_set']],missing_ll_rows[['round','param_set']]])
    unique_combinations = missing[['round', 'param_set']].drop_duplicates()
    
    # Convert to unique parameter_set ids
    ids = unique_combinations['param_set'] + unique_combinations['round']*batch_size
    ids = ids.values.tolist()
    ids=ids + [x+1 for x in list(range(n_prior))]
    ids = np.unique(ids)
    ids = [int(i) for i in ids]

    # Reformat scores
    YY = Y[(Y['site'] == site) & (Y['metric'] == metric)]
    YY['ps'] = YY['round'] * batch_size + YY['param_set'] 
    YY=YY[~YY['ps'].isin(ids)]
    scores = YY['ll'] * YY['my_weight']
    scores = scores[scores!=0]
    scores=torch.tensor(scores.values).unsqueeze(1)

    # Train single-task GP
    gp = SingleTaskGP(
        train_X=X,
        train_Y=scores,
        likelihood=GaussianLikelihood(noise_constraint=GreaterThan(1e-6)),
        outcome_transform=Standardize(m=scores.shape[-1]),
    )
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    
    # Fit GP
    fit_gpytorch_mll(mll)
    try:
        length_scales = gp.covar_module.base_kernel.lengthscale
    except Exception as e:
        try:
            length_scales = gp.covar_module.data_covar_module.lengthscale
        except:
            pass

    # Collect values
    results = []
    for id in range(1, length_scales.shape[1]+1):  # ID from 1 to n_parameters
        value = length_scales[0, id-1].item()  # Get the value from the tensor
        results.append({'metric': metric, 'site': site, 'id': id, 'value': value})

    # Save results as CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(f'{path_to_here}/output/{exp}/performance/GP/{metric}_{site}_LS.csv', index=False)
    
    return


def fit_GP_to_environment_objective(exp='',scoretype=''):
    path_to_here = get_script_path()
    
    print(f"score_type: {scoretype}",flush=True)
    
    # Load acquisition function details
    with open(f'{path_to_here}/output/{exp}/TurboThompson.json') as f:
        data = json.load(f)
    df = pd.DataFrame({'value': data})
    
    # Get batch size
    batch_size=df.at['batch_size', 'value']
    
    # Load input X - parameters
    X = torch.load(os.path.join(path_to_here,'output',exp,'X.pt'))
    
    # Load output Y - scores
    Y = pd.read_csv(os.path.join(path_to_here,'output',exp,'all_LL.csv'))
    # Reformat scores
    ps = Y['param_set']
    r = Y['round']
    Y['param_set']=ps
    Y['round']=r
    Y = pd.melt(Y,id_vars=["param_set","round"],value_vars=['eir_score','shape_score','intensity_score','prevalence_score'])
    #print(Y)
    results = []
    # Getting unique combinations of 'round' and 'parameter'
    unique_combinations = Y[['round', 'param_set']].drop_duplicates()
    #print(unique_combinations)
    
    # Convert to unique parameter_set ids
    ids = unique_combinations['param_set'] + unique_combinations['round']*batch_size
    ids = ids.values.tolist()
    ids=ids
    ids = np.unique(ids)
    ids = [int(i) for i in ids]
    #print(scoretype)
    # Reformat scores
    YY = Y[Y['variable']==scoretype]
    YY['ps'] = YY['round'] * batch_size + YY['param_set'] 
    YY=YY[YY['ps'].isin(ids)]
    scores = YY['value']
    scores=torch.tensor(scores.values).unsqueeze(1)
    #exit(1)
    # Train single-task GP
    gp = SingleTaskGP(
        train_X=X,
        train_Y=scores,
        likelihood=GaussianLikelihood(noise_constraint=GreaterThan(1e-6)),
        outcome_transform=Standardize(m=scores.shape[-1]),
    )
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    
    # Fit GP
    fit_gpytorch_mll(mll)
    try:
        length_scales = gp.covar_module.base_kernel.lengthscale
    except Exception as e:
        try:
            length_scales = gp.covar_module.data_covar_module.lengthscale
        except:
            pass

    # Collect values
    for id in range(1, length_scales.shape[1]+1):  # ID from 1 to n_parameters
        value = length_scales[0, id-1].item()  # Get the value from the tensor
        results.append({'score_type': scoretype, 'id': id, 'value': value})

    # Save results as CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(f'{path_to_here}/output/{exp}/performance/GP/{scoretype}_LS.csv', index=False)
    
    return



if __name__=="__main__":
    # Example run command for single site-metric
    fit_GP_to_objective(exp="test_new_env",
                        scoretype='eir_score')
    
    

