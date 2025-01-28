import torch
import torch.nn as nn
from torch.utils.data import Dataset


class dataloader(Dataset):
    def __init__(self, dir, in_norms=None, tgt_norms=None):
        self.dir = dir
        self.in_norms = in_norms 
        self.tgt_norms = tgt_norms

        # TODO: Count number of data points in all files in dir
        self.nsample = None

    def __len__(self):
        """
            Returns number of data-points in data set.
        """
        return self.nsample
    
    def __getitem__(self, idx):
        # TODO: * for index idx get part of data...
        #       * depending on the input of the model, we might want to 
        #           sample a couple of points from the files...
        #       * load input and target file
        #       * normalization of data?


        return input, target



