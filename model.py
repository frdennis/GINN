import torch
import torch.nn as nn



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
        super().__init_()

        self.nn = NNBlock(in_chan, out_chan)

        # TODO: Implement trainable model parameters here
        self.e = torch.nn.Parameter(torch.randn(1))

        # Enable gradient for trainable parameters
        self.e.requires_grad = True

    def forward(self, x):
        
        return self.nn(x)

    def physical_loss(self, y):
        # TODO: Implement physical loss based on differential equations
        
        return y

    

if __name__ == "__main__":
    nn = NNBlock(1, 1)
    
    x = torch.randn(1, 32, 1)
    y = nn(x)
    print(y.shape)


