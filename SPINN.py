import torch
import torch.nn as nn

def grad(out, inp):
    return torch.autograd.grad(out, 
                               inp, 
                               grad_outputs=torch.ones_like(out), 
                               create_graph=True)



class NNBlock(nn.Module):
    """
    Standard Feed Forward Neural Network 
    """
    def __init__(self, in_chan, out_chan, chans=[5,10,5], dropout_prob=0.1):
        super().__init__()
        #self.in_block = nn.Linear(in_chan, chans[0])
        layers = []

        layers.append(nn.Linear(in_chan, chans[0]))
        layers.append(nn.Tanh())
        #layers.append(nn.Dropout(p=dropout_prob))
        for i in range(len(chans) - 1):
            layers.append(nn.Linear(chans[i], chans[i+1]))
            layers.append(nn.Tanh())
            #layers.append(nn.Dropout(p=dropout_prob))

        layers.append(nn.Linear(chans[-1], out_chan))
        self.blocks = nn.Sequential(*layers)

    def forward(self, x):
        return self.blocks(x)


class RandomFourierEncoding(nn.Module):
    def __init__(self, num_features=50, sigma=1):
        self.B = torch.randn(num_features, input_dim) * sigma
        self.num_features = num_features

    def forward(self, x):
        # Compute the Fourier features: γ(x) = [cos(Bx), sin(Bx)]
        # B is applied element-wise to each sample
        cos_terms = torch.cos(torch.matmul(x, self.B.T))
        sin_terms = torch.sin(torch.matmul(x, self.B.T))
        return torch.cat([cos_terms, sin_terms], dim=-1)




class ModifiedMLP(nn.Module):
    def __init__(self, input_dim=1, output_dim=1, hidden_dims=[5,5,5]):
        super(ModifiedMLP, self).__init__()
        
        # Initialize encoders for input coordinates
        self.W1 = nn.Parameter(torch.randn(input_dim, hidden_dims[0]))  # W1 for U
        self.b1 = nn.Parameter(torch.zeros(hidden_dims[0]))  # b1 for U
        
        self.W2 = nn.Parameter(torch.randn(input_dim, hidden_dims[0]))  # W2 for V
        self.b2 = nn.Parameter(torch.zeros(hidden_dims[0]))  # b2 for V
        
        # Initialize layers for MLP
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, hidden_dims[0]))
        for i in range(len(hidden_dims) - 1):
            self.layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
        
        # Output layer
        self.out_layer = nn.Linear(hidden_dims[-1], output_dim)

        #self.act = nn.Sigmoid()
        self.act = nn.Tanh()
        
    def forward(self, x):
        # Encoders
        U = self.act(torch.matmul(x, self.W1) + self.b1)
        V = self.act(torch.matmul(x, self.W2) + self.b2)
        
        # Apply layers (with the modified MLP structure)
        h = x
        for l in range(len(self.layers)):
            f_l = self.layers[l](h)  # Apply linear transformation
            
            # Apply the pointwise multiplication as per the modification
            g_l = self.act(f_l) * U + (1 - self.act(f_l)) * V
            h = g_l  # Pass the result to the next layer
        
        # Final output
        output = self.out_layer(h)
        return output


if __name__ == "__main__":
    # Example usage
    input_dim = 1  # Example input dimension
    hidden_dims = [8, 8, 8]  # Example hidden layer dimensions
    output_dim = 1  # Output dimension (e.g., scalar for PDE residuals)

    model = ModifiedMLP(input_dim, output_dim, hidden_dims)

    # Example input
    x = torch.randn(16, input_dim)  # Batch of 10 samples

    # Forward pass
    output = model(x)
    print(output)



class PINN(nn.Module):
    def __init__(self, in_chan=1, out_chan=1, chans=[5,10,5], 
                e_ini=None, mu_ini=None, norm=1.):
        super().__init__()

        # Predicts Chi
        # self.nn = NNBlock(in_chan, out_chan, chans=chans)
        self.nn = ModifiedMLP(in_chan, out_chan, chans)

        self.norm = norm

        if e_ini == None:   
            self.e_ = torch.nn.Parameter(torch.abs(torch.randn(1)/10))
        else:
            self.e_ = torch.nn.Parameter(torch.tensor(e_ini))
        
        if mu_ini == None:
            self.mu_ = torch.nn.Parameter(torch.abs(torch.randn(1)/10))
        else:
            self.mu_ = torch.nn.Parameter(torch.tensor(mu_ini))
 
        # Enable gradient for trainable parameters
        self.e_requires_grad = True
        self.mu_.requires_grad = True
 
    def forward(self, x, _M):
        # Output u
        mu = self.get_mu()#.item()
        e  = self.get_e()#.item()

        u = mu * _M * (1 + e*torch.cos(self.nn(x)))
        return u
    
    def get_e(self):
        #return torch.abs(self.e_)
        return torch.tanh(torch.abs(self.e_))
    
    def get_mu(self):
        #return torch.abs(self.mu_)
        return torch.abs(self.mu_ * self.norm)
        
    def physical_loss(self, phi, lambda1=.5, lambda2=.5):
        chi = self.nn(phi)

        mu = self.get_mu()#.item()
        e  = self.get_e()#.item()

        dchi = grad(chi, phi)[0]
        ddchi = grad(dchi, phi)[0]

        ode1 = dchi**2 - (1 - 2*mu*(3 + e*torch.cos(chi)))
        ode2 = ddchi - mu*e*torch.sin(chi)

        loss1 = torch.mean(ode1**2)
        loss2 = torch.mean(ode2**2)
        #loss1 = torch.mean(torch.abs(ode1))
        #loss2 = torch.mean(torch.abs(ode2))

        loss1 *= lambda1
        loss2 *= lambda2
        return loss1 + loss2
    


class SPINN(nn.Module):
    def __init__(self, in_chan=1, out_chan=1, chans=[5,10,5],
                norm1=1., norm2=1.):
        super().__init__()
        self.M_ = torch.nn.Parameter(torch.tensor(1/0.042))
        self.M_.requires_grad = True

        # mu_ini = 1.4e-4
        self.S1 = PINN(in_chan, out_chan, chans, e_ini=0.425, mu_ini=.1, norm=norm1)
        self.S2 = PINN(in_chan, out_chan, chans, e_ini=0.88, mu_ini=1.8e-4, norm=norm2)

        self.norm1 = 1.
        self.norm2 = 1.
    
    def set_norm(self, norm1, norm2):
        self.norm1 = norm1
        self.norm2 = norm2

    def get_M(self):
        return torch.abs(self.M_)
    
    def forward(self, x1, x2):
        y1 = self.S1(x1, self.get_M())
        y2 = self.S2(x2, self.get_M())

        return y1, y2
    
    def physical_loss(self, phi):
        #phi = torch.linspace(-2.1*torch.pi, 4.2*torch.pi, steps=1000).view(-1,1).requires_grad_(True)
        loss1 = self.S1.physical_loss(phi, lambda1=.5, lambda2=.5)
        #phi = torch.linspace(-2.1*torch.pi, 4.2*torch.pi, steps=1000).view(-1,1).requires_grad_(True)
        loss2 = self.S2.physical_loss(phi, lambda1=.5, lambda2=.5)

        return loss1, loss2
    


if __name__ == "__main__":
    model = SPINN()
    print(model.S1.get_mu())
    print(model.S2.get_mu())


