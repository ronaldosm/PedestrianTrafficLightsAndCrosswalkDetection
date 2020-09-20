import torch

class MakeModel(torch.nn.Module):
    def __init__(self,net_type='efficientnet_b0',pretrained=False):
        super(MakeModel,self).__init__()
        self.original_model = torch.hub.load('narumiruna/efficientnet-pytorch', net_type, pretrained=pretrained)

        self.regression = torch.nn.Sequential(torch.nn.Linear(1000,4))
        self.IScrosswalk_output = torch.nn.Sequential(torch.nn.Linear(1000,1),torch.nn.Sigmoid())
        self.semaphore_output = torch.nn.Sequential(torch.nn.Linear(1000,3),torch.nn.Sigmoid())

        if not pretrained: self._initialize_weights()


    def forward(self,x):
        x = self.original_model(x)

        top_features = {}
        top_features['coordinates'] = self.regression(x)
        top_features['IScrosswalk'] = self.IScrosswalk_output(x)
        top_features['light_class'] = self.semaphore_output(x)

        return top_features

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                if m.bias is not None:
                    m.bias.data.zero_()
                torch.nn.init.xavier_normal_(m.weight.data)
            elif isinstance(m, torch.nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, torch.nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, torch.nn.Linear):
                m.weight.data.normal_(0, 0.01)

