import torch
import torchvision

class MakeModel(torch.nn.Module):
    def __init__(self,pretrained=False):
        super(MakeModel,self).__init__()
        self.model = torchvision.models.resnet50(pretrained=pretrained)
        self.regression = torch.nn.Sequential(torch.nn.Linear(1000,4))
        self.IScrosswalk_output = torch.nn.Sequential(torch.nn.Linear(1000,1),torch.nn.Sigmoid())
        self.semaphore_output = torch.nn.Sequential(torch.nn.Linear(1000,3),torch.nn.Sigmoid())


    def forward(self,x):
        x = self.model(x)

        top_features = {}
        top_features['coordinates'] = self.regression(x)
        top_features['IScrosswalk'] = self.IScrosswalk_output(x)
        top_features['light_class'] = self.semaphore_output(x)

        return top_features
