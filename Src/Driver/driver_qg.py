import json
import argparse
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import math
import time
import datetime
import numpy as np
import IPython.display as display
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from typing import Tuple, Union, Optional, List
from tqdm.notebook import tqdm
import torch.optim as optim
import dataclasses
import matplotlib.patches as patches
import matplotlib.ticker as ticker
import os
import sys
import warnings
import importlib

warnings.filterwarnings("ignore", category=UserWarning)

### Parse args
def load_config(config_path):
    with open(config_path, 'r') as file:
        config = json.load(file)
    config.pop('_comment', None)  # Remove comments if present
    return config
    
sys.path.append('/gdata/projects/ml_scope/Turbulence/QG_V0001/Src')

# Create an argument parser
parser = argparse.ArgumentParser(description='Parse args')

# Add the run_num argument to specify the run number
parser.add_argument('--run_num', type=int, help='Run number for configuration file', required=True)

# Parse the command-line arguments
args = parser.parse_args()
run_number = args.run_num

## Load config file
config_module = f'Config.Run{run_number:05d}'

from Utils.utils import print_config

try:
    config = importlib.import_module(config_module)
    print(f"Successfully loaded configuration from {config_module}")
    now = datetime.datetime.now()
    print(now.strftime("%Y-%m-%d %H:%M:%S"))
    print_config(config.params) 
except ModuleNotFoundError:
    print(f"Configuration file for run {run_num} not found.")
    now = datetime.datetime.now()
    print(now.strftime("%Y-%m-%d %H:%M:%S"))
    raise

## Set-up grid
from Grid.grid import Grid
grid_DNS=Grid(config.params.grid.Lx,config.params.grid.Ly,config.params.grid.Nx,config.params.grid.Ny)

## Set-up spectral derivatives and operators
from Operators.operators import SpectralDerivatives, LinearOperator, NonlinearOperator
spec_deriv_DNS=SpectralDerivatives(grid_DNS)
linop_DNS = LinearOperator(spec_deriv_DNS,config.params.pde)
nonlinop_DNS = NonlinearOperator(spec_deriv_DNS,config.params.pde)

print(f"Successfully loaded derivatives and operators")
now = datetime.datetime.now()
print(now.strftime("%Y-%m-%d %H:%M:%S"))

## Set-up initial conditions and masks
from Initial_forcing.ics import init_randn
init_conds_DNS =  init_randn(config.params.ic.energy, config.params.ic.wavenumbers, grid_DNS, spec_deriv_DNS, config.params.ic.seed)

from Masks.masks import create_circular_mask
obstacle_mask_DNS =  create_circular_mask(grid_DNS,config.params.mask.r,config.params.mask.tol)

print(f"Successfully created initial conditions and obstacles")
now = datetime.datetime.now()
print(now.strftime("%Y-%m-%d %H:%M:%S"))

## Run simulation
print(f"Simulation started")
now = datetime.datetime.now()
print(now.strftime("%Y-%m-%d %H:%M:%S"))

from Simulation.simulation import Simulation

sim_DNS = Simulation(grid_DNS,config.params.pde,spec_deriv_DNS,linop_DNS,nonlinop_DNS,init_conds_DNS,obstacle_mask_DNS,None,
                     config.params.time.dt,config.params.time.T)

solution_field = sim_DNS.run()

print(f"Simulation completed successfully")
now = datetime.datetime.now()
print(now.strftime("%Y-%m-%d %H:%M:%S"))


## Save numpy files

base_dir = '/gdata/projects/ml_scope/Turbulence/QG_V0001/Results'
save_dir = os.path.join(base_dir, f'Run{run_number:05d}')
os.makedirs(save_dir, exist_ok=True)
file_name = f'vorticity_Run{run_number:05d}.npy'
file_path = os.path.join(save_dir, file_name)
np.save(file_path, solution_field[:, :, ::500].cpu().numpy()) ## Save every 500th timestep

print(f"Simulation np files saved successfully")
now = datetime.datetime.now()
print(now.strftime("%Y-%m-%d %H:%M:%S"))

