import torch
import pandas as pd
import numpy as np


class SLoader():
    def __init__(self, star="S2", file_path = 'data/asu.csv',
                 scaling=1e0, augment_data=False, std_data=False):
        # Actions to be done on the data during retrieval
        self.augment_data = augment_data
        self.std_data = std_data
        self.file_path = file_path
        self.star = star
        
        # 
        self.distance = 7900 # pc

        self.load_data()
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
        
    def load_data(self):
        """
        load_data:
            Loads data and error, and concatenates into tensors for x- and y-coordinates
        
        ARGS:
            * files :   List of files to load. 
        """
        df = pd.read_csv(self.file_path, sep=',')
        df.columns = df.columns.str.strip()

        try:
            x = pd.to_numeric(df['oRA-'+self.star].str.strip(), errors='coerce').values  # RA (arcsec)
            y = pd.to_numeric(df['oDE-'+self.star].str.strip(), errors='coerce').values  # Dec (arcsec)
            x_e = pd.to_numeric(df['e_oRA-'+self.star].str.strip(), errors='coerce').values  # RA (arcsec)
            y_e = pd.to_numeric(df['e_oDE-'+self.star].str.strip(), errors='coerce').values  # Dec (arcsec)
        except KeyError: 
            raise KeyError(f"{self.star} is not a registered star in the file {self.file_path}")

        # remove nans
        self.x = x[~np.isnan(x)]
        self.y = y[~np.isnan(y)]        
        self.x_e = x_e[~np.isnan(x_e)]
        self.y_e = y_e[~np.isnan(y_e)]

        # convert to tensor with correct shape
        self.x = torch.tensor(self.x).unsqueeze(1)
        self.y = torch.tensor(self.y).unsqueeze(1)
        self.x_e = torch.tensor(self.x_e).unsqueeze(1)
        self.y_e = torch.tensor(self.y_e).unsqueeze(1)


    def projection(self):
        """
        project:

        TODO: 
            * Project error

        """
        pc_to_au = 206265
        mas_to_rad = np.pi / (180 * 3600 * 1000)


        df = pd.read_csv("data/angles.csv", index_col=0)

        i = np.radians(df.loc[self.star, "i"])
        Ω = np.radians(df.loc[self.star, "Ω"])
        ω = np.radians(df.loc[self.star, "ω"])

        # Thiele-Innes elements
        A = np.cos(ω)*np.cos(Ω) - np.sin(ω)*np.sin(Ω)*np.cos(i)
        B = np.cos(ω)*np.sin(Ω) + np.sin(ω)*np.cos(Ω)*np.cos(i)
        F = -np.sin(ω)*np.cos(Ω) - np.cos(ω)*np.sin(Ω)*np.cos(i)
        G = -np.sin(ω)*np.sin(Ω) + np.cos(ω)*np.cos(Ω)*np.cos(i)

        # Transformation matrix (observer -> orbital plane)
        M_inv = np.array([[F, -G], [-A, B]]) / (B*F - A*G)

        X = self.x * mas_to_rad * self.distance * pc_to_au
        Y = self.y * mas_to_rad * self.distance * pc_to_au

        self.x = M_inv[0,0] * (X) + M_inv[0,1] * Y
        self.y = M_inv[1,0] * (X) + M_inv[1,1] * Y


    def __call__(self):
        phi = self.phi
        u = self.u

        if self.std_data:
            x_ = self.x + torch.normal(0, self.x_e)
            y_ = self.y + torch.normal(0, self.y_e)
            u = 1/torch.sqrt(x_**2 + y_**2)
            phi = np.arccos(x_*u) 
            phi = torch.where(y_<0., -phi, phi).float()
            phi = torch.where(phi<0, phi+2*torch.pi, phi)
            phi = torch.where(phi>2.2*torch.pi, phi-2*torch.pi, phi)

        if self.augment_data:
            phi = (phi + (torch.randint(low=0,high=3,size=phi.shape) - 1) * 2 * torch.pi)


        return torch.tensor(phi, dtype=torch.float32), torch.tensor(u, dtype=torch.float32) # norm const
    
    def __len__(self):
        return len(self.phi)
    
if __name__ == "__main__":
    d = SLoader(star="S2")

    x = d.x
    y = d.y    

    phi = d.phi
    u = d.u

    print(phi, u)

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10,5), ncols=2)
    ax[0].plot(phi, u, 'kx')
    ax[0].grid()

    ax[1].plot(-x,-y,'kx')
    ax[1].grid()
    ax[1].axis('equal')
    plt.show()