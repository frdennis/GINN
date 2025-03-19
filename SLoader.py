import torch
import torch.nn as nn
from torch.utils.data import Dataset
import pandas as pd
import numpy as np


class SLoader():
    def __init__(self, files=["data/astrometry_NACO.csv", "data/astrometry_SHARP.csv"], 
                 scaling=1e0, augment_data=False, std_data=False):
        self.augment_data = augment_data
        self.std_data = std_data
        self.convertion_const = torch.pi / (3600*180) * 8.3 * 1000 * 206265 * scaling
        self.load_data(files)

        self.u = 1/np.sqrt(self.x**2 + self.y**2)
        phi = np.arccos(self.x*self.u) 
        self.phi = torch.where(self.y<0., -phi, phi).float()
        #self.phi[:len(y1)] += torch.pi * 2
        self.phi = torch.where(self.phi<0, self.phi+2*torch.pi, self.phi)
        self.phi = torch.where(self.phi>2.2*torch.pi, self.phi-2*torch.pi, self.phi)

        self.u = torch.tensor(self.u, dtype=torch.float32)
        #self.norm_const = torch.mean(self.u)
        
    def load_data(self, files):
        y = []
        y_std = []
        x = []
        x_std = []
        for file in files:
            data = pd.read_csv(file)

            y_ = data.iloc[:,1].values * self.convertion_const 
            y_std_ = data.iloc[:,2].values * self.convertion_const
            x_ = data.iloc[:,3].values * self.convertion_const
            x_std_ = data.iloc[:,4].values * self.convertion_const

            y.append(torch.tensor(y_))
            y_std.append(torch.tensor(y_std_))
            x.append(torch.tensor(x_))
            x_std.append(torch.tensor(x_std_))

        if len(x) != 1:
            self.x = torch.cat(x).unsqueeze(1)
            self.y = torch.cat(y).unsqueeze(1)
            self.x_std = torch.cat(x_std).unsqueeze(1)
            self.y_std = torch.cat(y_std).unsqueeze(1)
        else:
            self.x = torch.tensor(x[0]).unsqueeze(1)
            self.y = torch.tensor(y[0]).unsqueeze(1)
            self.x_std = torch.tensor(x_std[0]).unsqueeze(1)
            self.y_std = torch.tensor(y_std[0]).unsqueeze(1)

    def __call__(self):
        phi = self.phi
        u = self.u

        if self.std_data:
            x_ = self.x + torch.normal(0, self.x_std)
            y_ = self.y + torch.normal(0, self.y_std)
            u = 1/torch.sqrt(x_**2 + y_**2)
            phi = np.arccos(x_*u) 
            phi = torch.where(y_<0., -phi, phi).float()
            phi = torch.where(phi<0, phi+2*torch.pi, phi)
            phi = torch.where(phi>2.2*torch.pi, phi-2*torch.pi, phi)

        if self.augment_data:
            phi = (phi + (torch.randint(low=0,high=3,size=phi.shape) - 1) * 2 * torch.pi)


        return torch.tensor(phi, dtype=torch.float32), torch.tensor(u, dtype=torch.float32) # norm const
    
if __name__ == "__main__":
    d = SLoader(files=["data/astrometry_NACO.csv", "data/astrometry_SHARP.csv"])
    #d = SLoader(files=["data/astrometry_NACO.csv"])
    x, y = d()
    print(x,y)
    print(x.shape, y.shape)
    #x, y = d()
    #plt.plot(x,y,'kx')