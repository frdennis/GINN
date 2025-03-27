import torch
import torch.nn as nn
from torch.utils.data import Dataset
import pandas as pd
import numpy as np


class SLoader():
    def __init__(self, files=["data/astrometry_NACO.csv", "data/astrometry_SHARP.csv"], 
                 scaling=1e0, augment_data=False, std_data=False):
        # Actions to be done on the data during retrieval
        self.augment_data = augment_data
        self.std_data = std_data
        
        # 
        self.distance = 7900 # pc

        self.load_data(files)
        self.projection()
        self.y *= scaling 
        self.x *= scaling


        self.u = 1/np.sqrt(self.x**2 + self.y**2)
        phi = np.arccos(self.x*self.u) 
        self.phi = torch.where(self.y<0., -phi, phi).float()
        # self.phi[:len(y1)] += torch.pi * 2
        self.phi = torch.where(self.phi<0, self.phi+2*torch.pi, self.phi)
        self.phi = torch.where(self.phi>2.2*torch.pi, self.phi-2*torch.pi, self.phi)

        self.u = torch.tensor(self.u, dtype=torch.float32)
        
    def load_data(self, files):
        """
        load_data:
            Loads data and error, and concatenates into tensors for x- and y-coordinates
        
        ARGS:
            * files :   List of files to load. 
        """
        y = []
        y_std = []
        x = []
        x_std = []
        for file in files:
            data = pd.read_csv(file)

            y_ = data.iloc[:,3].values 
            y_std_ = data.iloc[:,4].values
            x_ = data.iloc[:,1].values 
            x_std_ = data.iloc[:,2].values

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

    def projection(self, angles=[134.18, 227.85, 66.12]):
        """
        project:

        TODO: 
            * Project error

        """
        pc_to_au = 206265
        mas_to_rad = np.pi / (180 * 3600)

        i = np.radians(angles[0])
        Ω = np.radians(angles[1])
        ω = np.radians(angles[2])

        # Thiele-Innes elements
        A = np.cos(ω)*np.cos(Ω) - np.sin(ω)*np.sin(Ω)*np.cos(i)
        B = np.cos(ω)*np.sin(Ω) + np.sin(ω)*np.cos(Ω)*np.cos(i)
        F = -np.sin(ω)*np.cos(Ω) - np.cos(ω)*np.sin(Ω)*np.cos(i)
        G = -np.sin(ω)*np.sin(Ω) + np.cos(ω)*np.cos(Ω)*np.cos(i)

        # Transformation matrix (observer -> orbital plane)
        M_inv = np.array([[F, -G], [-A, B]]) / (B*F - A*G)

        X = self.x * mas_to_rad * self.distance * pc_to_au
        Y = self.y * mas_to_rad * self.distance * pc_to_au

        self.x = M_inv[0,0] * (-X) + M_inv[0,1] * Y
        self.y = M_inv[1,0] * (-X) + M_inv[1,1] * Y


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

    x = d.x
    y = d.y    
    #x, y = d()

    import matplotlib.pyplot as plt
    plt.plot(x,y,'kx')
    plt.show()