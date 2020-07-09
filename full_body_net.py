import torch

# facenet repurposed for people recognition
class FullBodyNet(torch.nn.Module):
    def __init__(self):
        super(FullBodyNet, self).__init__()
        inception_v3 = torch.hub.load('pytorch/vision:v0.6.0', 'inception_v3', pretrained=True)
        inception_v3.fc = torch.nn.Linear(inception_v3.fc.in_features, 128)
        self.inception_v3 = inception_v3

    # this just applies L2 norm to inception_v3
    def forward(self, x):
        inception_outputs = self.inception_v3(x)
        return torch.abs(inception_outputs[0]), inception_outputs[1]