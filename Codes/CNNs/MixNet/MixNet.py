import torch

# ------------------- Split Channels -------------------
def split_channels(N_channels, N_groups):
    split_channels = [N_channels//N_groups]*N_groups
    split_channels[0] += N_channels - sum(split_channels)
    return split_channels


# ---------------- Activation Functions ----------------
class hard_swish(torch.nn.Module):
    def __init__(self,inplace=True):
        super(hard_swish,self).__init__()
        self.inplace = inplace

    def forward(self,x):
        return x*torch.nn.functional.relu6(x+3.0, inplace=self.inplace)/6.0


class hard_sigmoid(torch.nn.Module):
    def __init__(self,inplace=True):
        super(hard_sigmoid, self).__init__()
        self.inplace = inplace

    def forward(self,x):
        return torch.nn.functional.relu6(x+3.0,inplace=self.inplace)/6.0


activation_dict = {'ReLU': torch.nn.ReLU(inplace=True),'HSwish': hard_swish()}


# -------------------- Base Layers ---------------------
def batchnorm_conv(in_channels, out_channels, strides):
    return torch.nn.Sequential(torch.nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=strides,padding=1,bias=False),
                               torch.nn.BatchNorm2d(out_channels),
                               torch.nn.ReLU(inplace=True))


def batchnorm_1x1_conv(in_channels, out_channels):
    return torch.nn.Sequential(torch.nn.Conv2d(in_channels,out_channels,kernel_size=1,stride=1,padding=0,bias=False),
                               torch.nn.BatchNorm2d(out_channels),
                               torch.nn.ReLU(inplace=True))


class squeeze_excitation(torch.nn.Module):
    def __init__(self,in_channels,reduction_channels):
        super(squeeze_excitation,self).__init__()
        self.FC = torch.nn.Sequential(torch.nn.Linear(in_channels,reduction_channels,bias=False),
                                      torch.nn.ReLU(inplace=True),
                                      torch.nn.Linear(reduction_channels,in_channels,bias=False),
                                      hard_sigmoid())

    def forward(self,x):
        y = torch.nn.AdaptiveAvgPool2d(1)(x)
        y = torch.flatten(y,start_dim=1)
        y = self.FC(y).unsqueeze(2)
        return x*y.unsqueeze(3).expand_as(x)


# ----------------- Group Convolutions -----------------
class grouped_Conv2D(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, strides=1, padding=0):
        super(grouped_Conv2D, self).__init__()

        self.N_groups = len(kernel_size)
        self.split_in_channels = split_channels(in_channels, self.N_groups)
        self.split_out_channels = split_channels(out_channels, self.N_groups)

        self.grouped_conv = torch.nn.ModuleList()
        for i in range(self.N_groups):
            self.grouped_conv.append(torch.nn.Conv2d(self.split_in_channels[i],self.split_out_channels[i],kernel_size[i],stride=strides,padding=padding,bias=False) )

    def forward(self, x):
        if self.N_groups == 1: return self.grouped_conv[0](x)
        else:
            x_split = torch.split(x, self.split_in_channels, dim=1)
            x = [conv(group) for conv, group in zip(self.grouped_conv, x_split)]
            return torch.cat(x, dim=1)


class mixed_DW_Conv2D(torch.nn.Module):
    def __init__(self, N_channels, kernel_size, strides):
        super(mixed_DW_Conv2D, self).__init__()

        self.N_groups = len(kernel_size)
        self.splited_channels = split_channels(N_channels, self.N_groups)

        self.mixed_depthwise_conv = torch.nn.ModuleList()
        for i in range(self.N_groups):
            self.mixed_depthwise_conv.append(torch.nn.Conv2d(self.splited_channels[i],
                                                             self.splited_channels[i],
                                                             kernel_size[i],
                                                             stride=strides,
                                                             padding=kernel_size[i]//2,
                                                             groups=self.splited_channels[i],
                                                             bias=False ))

    def forward(self, x):
        if self.N_groups == 1: return self.mixed_depthwise_conv[0](x)
        else:
            x_split = torch.split(x, self.splited_channels, dim=1)
            x = [conv(group) for conv, group in zip(self.mixed_depthwise_conv, x_split)]
            return torch.cat(x, dim=1)


# -------------------- MixNet Block --------------------
class MixNet_block(torch.nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=[3],exp_kernel_size=[1],project_kernel_size=[1],strides=1,exp_ratio=1,activation='ReLU',SE_ratio=0.0):
        super(MixNet_block, self).__init__()

        exp_layer = exp_ratio != 1
        expand_channels = in_channels*exp_ratio
        SE_layer = SE_ratio != 0.0
        self.residual_connection = strides == 1 and in_channels == out_channels

        conv_list = []

        # Expansion Phase
        if exp_layer:
            exp_layer = torch.nn.Sequential(grouped_Conv2D(in_channels, expand_channels, exp_kernel_size),
                        torch.nn.BatchNorm2d(expand_channels),
                        activation_dict[activation] )
            conv_list.append(exp_layer)

        # Depthwise Convolution Phase
        DWC_layer = torch.nn.Sequential(mixed_DW_Conv2D(expand_channels, kernel_size, strides),
                                        torch.nn.BatchNorm2d(expand_channels),
                                        activation_dict[activation])
        conv_list.append(DWC_layer)

        # Squeeze-Excitation Phase
        if SE_layer:
            SE_layer = squeeze_excitation(expand_channels, int(in_channels*SE_ratio))
            conv_list.append(SE_layer)

        # Projection Phase
        projection = torch.nn.Sequential(grouped_Conv2D(expand_channels, out_channels, project_kernel_size),
                                        torch.nn.BatchNorm2d(out_channels) )
        conv_list.append(projection)

        self.conv_list = torch.nn.Sequential(*conv_list)

    def forward(self, x):
        if self.residual_connection: return x + self.conv_list(x)
        else: return self.conv_list(x)



# -------------------- MixNet Model --------------------
class MakeModel(torch.nn.Module):
    def __init__(self,include_top=True,semaphore_output=True):
        super(MakeModel, self).__init__()
        self.include_top = include_top
        botton = []

        # CNN Structure
        botton.append(batchnorm_conv(in_channels=3,out_channels=16,strides=2))

        botton.append(MixNet_block(16, 16, kernel_size=[3],         exp_kernel_size=[1],  project_kernel_size=[1],  strides=1,exp_ratio=1,activation='ReLU',  SE_ratio=0.0 ))
        botton.append(MixNet_block(16, 24, kernel_size=[3],         exp_kernel_size=[1,1],project_kernel_size=[1,1],strides=2,exp_ratio=6,activation='ReLU',  SE_ratio=0.0 ))
        botton.append(MixNet_block(24, 24, kernel_size=[3],         exp_kernel_size=[1,1],project_kernel_size=[1,1],strides=1,exp_ratio=3,activation='ReLU',  SE_ratio=0.0 ))
        botton.append(MixNet_block(24, 40, kernel_size=[3,5,7],     exp_kernel_size=[1],  project_kernel_size=[1],  strides=2,exp_ratio=6,activation='HSwish',SE_ratio=0.5 ))
        botton.append(MixNet_block(40, 40, kernel_size=[3,5],       exp_kernel_size=[1,1],project_kernel_size=[1,1],strides=1,exp_ratio=6,activation='HSwish',SE_ratio=0.5 ))
        botton.append(MixNet_block(40, 40, kernel_size=[3,5],       exp_kernel_size=[1,1],project_kernel_size=[1,1],strides=1,exp_ratio=6,activation='HSwish',SE_ratio=0.5 ))
        botton.append(MixNet_block(40, 40, kernel_size=[3,5],       exp_kernel_size=[1,1],project_kernel_size=[1,1],strides=1,exp_ratio=6,activation='HSwish',SE_ratio=0.5 ))
        botton.append(MixNet_block(40, 80, kernel_size=[3,5,7],     exp_kernel_size=[1],  project_kernel_size=[1,1],strides=2,exp_ratio=6,activation='HSwish',SE_ratio=0.25))
        botton.append(MixNet_block(80, 80, kernel_size=[3,5],       exp_kernel_size=[1],  project_kernel_size=[1,1],strides=1,exp_ratio=6,activation='HSwish',SE_ratio=0.25))
        botton.append(MixNet_block(80, 80, kernel_size=[3,5],       exp_kernel_size=[1],  project_kernel_size=[1,1],strides=1,exp_ratio=6,activation='HSwish',SE_ratio=0.25))
        botton.append(MixNet_block(80, 120,kernel_size=[3,5,7],     exp_kernel_size=[1,1],project_kernel_size=[1,1],strides=1,exp_ratio=6,activation='HSwish',SE_ratio=0.5 ))
        botton.append(MixNet_block(120,120,kernel_size=[3,5,7],     exp_kernel_size=[1,1],project_kernel_size=[1,1],strides=1,exp_ratio=3,activation='HSwish',SE_ratio=0.5 ))
        botton.append(MixNet_block(120,120,kernel_size=[3,5,7],     exp_kernel_size=[1,1],project_kernel_size=[1,1],strides=1,exp_ratio=3,activation='HSwish',SE_ratio=0.5 ))
        botton.append(MixNet_block(120,200,kernel_size=[3,5,7,9,11],exp_kernel_size=[1],  project_kernel_size=[1],  strides=2,exp_ratio=6,activation='HSwish',SE_ratio=0.5 ))
        botton.append(MixNet_block(200,200,kernel_size=[3,5,7,9],   exp_kernel_size=[1],  project_kernel_size=[1,1],strides=1,exp_ratio=6,activation='HSwish',SE_ratio=0.5 ))
        botton.append(MixNet_block(200,200,kernel_size=[3,5,7,9],   exp_kernel_size=[1],  project_kernel_size=[1,1],strides=1,exp_ratio=6,activation='HSwish',SE_ratio=0.5 ))

        botton.append(batchnorm_1x1_conv(200,960))
        botton.append(torch.nn.AdaptiveAvgPool2d(1))
        botton.append(torch.nn.Conv2d(in_channels=960,out_channels=1280,kernel_size=1,stride=1,padding=0))
        botton.append(torch.nn.Dropout(0.2))

        self.botton = torch.nn.Sequential(*botton)

        if self.include_top:
            self.coordinates_output = torch.nn.Sequential(torch.nn.Linear(1280,4))
            self.IScrosswalk_output = torch.nn.Sequential(torch.nn.Linear(1280,1),hard_sigmoid())
            if semaphore_output: self.semaphore_output = torch.nn.Sequential(torch.nn.Linear(1280,3),hard_sigmoid())

        self._initialize_weights()


    def forward(self, x):
        x = self.botton(x)
        x = x.view(x.size(0),-1)
        if self.include_top:
            top_features = {}
            top_features['coordinates'] = self.coordinates_output(x)
            top_features['IScrosswalk'] = self.IScrosswalk_output(x)
            if self.semaphore_output: top_features['light_class'] = self.semaphore_output(x)

            return top_features
        else: return x


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
