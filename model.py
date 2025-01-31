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
    def __init__(self, in_chan, out_chan):
        super().__init__()

        self.blocks = nn.Sequential(
            nn.Linear(in_chan, 32),
            nn.Tanh(),
            nn.Linear(32, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, out_chan)
        )

    def forward(self, x):
        return self.blocks(x)
    

class PINN(nn.Module):
    """
    Physics Informed Neural Network. 
        * physical_loss :
            Calculates the physical loss using 
            - returns torch.tensor
    """
    def __init__(self, in_chan, out_chan):
        super().__init__()

        # Predicts Chi
        self.nn = NNBlock(in_chan, out_chan)

        # TODO: Implement trainable model parameters here
        self.e = torch.nn.Parameter(torch.tensor(0.8))
        self.mu = torch.nn.Parameter(torch.tensor(0.00018))
        self.M = torch.nn.Parameter(torch.tensor(.04))

        # Enable gradient for trainable parameters
        self.e.requires_grad = True
        self.mu.requires_grad = True
        self.M.requires_grad = True

    def forward(self, x):
        # Output u
        # TODO: no gradient for model params
        u = self.mu/self.M * (1 + self.e*torch.cos(self.nn(x)))
        return u

    def physical_loss(self, lambda1=.5, lambda2=.5):
        # TODO: Implement physical loss based on differential equations
        phi = torch.linspace(-16*3.1415, 16*3.1415, steps=1000).view(-1,1).requires_grad_(True)
        chi = self.nn(phi)

        dchi = grad(chi, phi)[0]
        ddchi = grad(dchi, phi)[0]
        # Second derivative

        ode1 = dchi**2 - 1 + 2*self.mu*(3 + self.e*torch.cos(chi))
        ode2 = ddchi - self.mu*self.e*torch.sin(chi)

        loss1 = torch.mean(ode1**2)
        loss2 = torch.mean(ode2**2)

        return loss1*lambda1 + loss2*lambda2

    

if __name__ == "__main__":
    nn = PINN(1, 1)
    
    x = torch.randn(1, 32, 1)
    print(nn.physical_loss())
    #print(y.shape)


