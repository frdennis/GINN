import torch
import torch.nn as nn
from torch.utils.data import Dataset
import pandas as pd
import numpy as np


class starloader(Dataset):
    def __init__(self, file="/mn/stornext/d8/data/dennisfr/PINN/S-Star-PINN/data/astrometry_NACO.csv"):
        data = pd.read_csv(file)

        micro_arcsec_to_au = torch.pi / (3600*180) * 8 * 1000 * 206265 #* 1e-2 # to units 1e-2 AU

        x = data.iloc[:,1].values * micro_arcsec_to_au
        y = data.iloc[:,3].values * micro_arcsec_to_au


        # Y values
        self.u = 1/np.sqrt(x**2 + y**2)

        phi = np.arccos(x*self.u)
        self.phi = np.where(y<0., -phi, phi)

        self.u = torch.tensor(self.u, dtype=torch.float32)
        self.phi = torch.tensor(self.phi, dtype=torch.float32)

    def __len__(self):
        """
            Returns number of data-points in data set.
        """
        return len(self.phi)
    
    def __getitem__(self, idx):
        # TODO: * for index idx get part of data...
        #       * depending on the input of the model, we might want to 
        #           sample a couple of points from the files...
        #       * load input and target file
        #       * normalization of data?

        return torch.tensor([self.phi[idx]]), torch.tensor([self.u[idx]])
    

if __name__ == "__main__":
    loader = starloader()



