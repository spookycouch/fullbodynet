import torch
from .resnet import resnet50

class FullBodyNet(torch.nn.Module):
    def __init__(self, train=False):
        super(FullBodyNet, self).__init__()
        self.resnet50 = resnet50(pretrained=train)
        self.resnet50.fc = torch.nn.Linear(self.resnet50.fc.in_features, 128)

    # this just applies L2 norm to resnet50
    def forward(self, x):
        resnet_outputs = self.resnet50(x)
        return resnet_outputs/torch.norm(resnet_outputs, p=2, dim=1, keepdim=True)