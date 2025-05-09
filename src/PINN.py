import torch
import torch.nn as nn

def grad(out, inp):
    return torch.autograd.grad(out, 
                               inp, 
                               grad_outputs=torch.ones_like(out), 
                               create_graph=True,
                               allow_unused=True)



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
        #y = torch.sin(self.in_block(x))
        #return self.blocks(y)
    
        return self.blocks(x)
    


class ffnn(nn.Module):
    """
    Standard Feed Forward Neural Network 
    """
    def __init__(self, in_chan, out_chan):
        super().__init__()
        self.l1 = nn.Linear(in_chan, out_chan)
        # self.blocks = nn.Sequential(
        #     nn.Linear(in_chan, 5),
        #     nn.Tanh(),
        #     nn.Linear(5, 10),
        #     nn.Tanh(),
        #     nn.Linear(10, 5),
        #     nn.Tanh(),
        #     nn.Linear(5, 1),
        #     nn.Linear(1, out_chan)
        # )

    def forward(self, x):
        #return self.blocks(x)
        return self.l1(x)


class PINN(nn.Module):
    """
    Physics Informed Neural Network. 
        * physical_loss :
            Calculates the physical loss using 
            - returns torch.tensor
    """
    def __init__(self, in_chan=1, out_chan=1, chans=[5,10,5]):
        super().__init__()

        # Predicts Chi
        self.nn = NNBlock(in_chan, out_chan, chans=chans)
        #self.nn = ffnn(in_chan, out_chan)

        #self.e_ = torch.nn.Parameter(torch.tensor(0.8))
        #self.mu_ = torch.nn.Parameter(torch.tensor(0.00018))
        #self.M_ = torch.nn.Parameter(torch.tensor(.04))

        self.e_ = torch.nn.Parameter(torch.abs(torch.randn(1)/10))
        self.mu_ = torch.nn.Parameter(torch.abs(torch.randn(1)/10))
        self.M_ = torch.nn.Parameter(torch.abs(torch.randn(1)/10))
        #self.p_ = torch.nn.Parameter(torch.abs(torch.randn(1)))

        # Use initial guess
        #self.e_ = torch.nn.Parameter(torch.arctanh(torch.log(torch.tensor(0.5)))) # 0.8
        #self.M_ = torch.nn.Parameter(torch.log(torch.tensor(0.01))) # 0.04
        #self.p_ = torch.nn.Parameter(torch.log(torch.tensor(200.))) # 220.

        # Enable gradient for trainable parameters
        self.e_requires_grad = True
        self.mu_.requires_grad = True
        self.M_.requires_grad = True
        #self.p_.requires_grad = True

    def forward(self, x):
        # Output u

        mu = self.get_mu()#.item()
        #p = self.get_p()
        M  = self.get_M()
        e  = self.get_e()#.item()

        u = mu/M * (1 + e*torch.cos(self.nn(x)))
        return u
    
    def get_e(self):
        return torch.abs(self.e_)
        #return torch.tanh(torch.exp(self.e_))
    
    def get_M(self):
        return torch.abs(self.M_)
        #return torch.exp(self.M_)
    
    def get_mu(self):
        return torch.abs(self.mu_)
        #return torch.exp(self.mu_)
    
    def get_p(self):
        return self.get_M()/self.get_mu()
        #return self.M_/self.mu_
        #return torch.exp(self.p_)

    def physical_loss(self, phi, lambda1=1., lambda2=1.):
        chi = self.nn(phi)

        mu = self.get_mu()#.item()
        #p  = self.get_p()
        #M  = self.get_M()
        e  = self.get_e()#.item()

        dchi = grad(chi, phi)[0] # dchi/dphi = dchi/du * du/dphi = du/dphi * du/dchi
        ddchi = grad(dchi, phi)[0]

        # u = mu/M * (1 + e*torch.cos(self.nn(x)))
        # dchi = grad(self.forward(phi), phi)[0] * grad(mu/M * (1 + e*torch.cos(chi)), chi)[0]

        ode1 = dchi**2 - (1 - 2*mu*(3 + e*torch.cos(chi)))
        ode2 = ddchi - mu*e*torch.sin(chi)

        loss1 = torch.mean(ode1**2)
        loss2 = torch.mean(ode2**2)
        
        # L1 Loss
        #loss1 = torch.mean(torch.abs(ode1))
        #loss2 = torch.mean(torch.abs(ode2))

        loss1 *= lambda1
        loss2 *= lambda2

        #print(loss1, loss2)
        return loss1 + loss2
    

if __name__ == "__main__":
    nn = PINN(1, 1)
    
    x = torch.randn(32, 1).requires_grad_(True)
    print(nn.physical_loss(x))

    y = nn.nn(torch.linspace(-3, 3, steps=100).view(-1,1)).detach().numpy()  
    print(y)
    #print(y.shape)


