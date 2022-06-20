import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
from torch.utils import data
from dataloader import HDataSet


class CrossEntropy2d(nn.Module):

    def __init__(self, size_average=True, ignore_label=255):
        super(CrossEntropy2d, self).__init__()
        self.size_average = size_average
        self.ignore_label = ignore_label

    def forward(self, predict, target, weight=None):
        """
            Args:
                predict:(n, c, h, w)
                target:(n, h, w)
                weight (Tensor, optional): a manual rescaling weight given to each class.
                                           If given, has to be a Tensor of size "nclasses"
        """
        assert not target.requires_grad
        assert predict.dim() == 3
        assert target.dim() == 3
        assert predict.size(0) == target.size(0), "{0} vs {1} ".format(predict.size(0), target.size(0))
        assert predict.size(1) == target.size(1), "{0} vs {1} ".format(predict.size(2), target.size(1))
        assert predict.size(2) == target.size(2), "{0} vs {1} ".format(predict.size(3), target.size(3))

        loss = F.smooth_l1_loss(predict, target)
        return loss

# def main():
   
if __name__ == '__main__':
    dst = HDataSet('./', './train.txt', mean=(0))
    trainloader = data.DataLoader(dst, batch_size=4)
    for i, data in enumerate(trainloader):
        imgs, labs, _ = data
        print(imgs.shape)
        print(labs.shape)
        cal_loss = CrossEntropy2d()
        loss = cal_loss(imgs,labs)
        print(loss)
        break