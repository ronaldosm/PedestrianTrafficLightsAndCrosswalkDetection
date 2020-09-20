import torch

class FCN():
    def __init__(self,loss_types,cuda=False):
        # Loss functions that can be used:
        loss_functions_list = {'MSE':torch.nn.MSELoss(),
                               'CrossEntropy':torch.nn.CrossEntropyLoss(),
                               'BCE':torch.nn.BCELoss(),
                               'SmoothL1':torch.nn.SmoothL1Loss()}
        # Chosen loss functions:
        self.coord_loss_fcn = loss_functions_list[loss_types[0]]
        self.IScrw_loss_fcn = loss_functions_list[loss_types[1]]
        self.LClass_loss_fcn= loss_functions_list[loss_types[2]]

        self.one_hot = loss_types[2] == 'MSE'
        self.cuda = cuda

    def compute(self,targets,predictions):
        true_coordinates,true_IScrosswalk,true_light_class = targets
        true_light_class = true_light_class.long()

        # Identify which indexes contain Crosswalk coordinates
        IScoordinates = [i for i, is_crosswalk  in enumerate(true_IScrosswalk) if is_crosswalk   != 0]

        # If needed, convert the target Light-Class variable to One-Hot Encoding
        if self.one_hot:
            one_hot_true_light_class = torch.tensor([[0.0,0.0,0.0]]*len(true_light_class))
            for (array,Class) in zip(one_hot_true_light_class,true_light_class): array[Class] = 1
            true_light_class = one_hot_true_light_class
            if self.cuda: true_light_class = true_light_class.cuda()

        # Calculation of losses:
        coordinates_loss = self.coord_loss_fcn(predictions['coordinates'][IScoordinates],true_coordinates[IScoordinates]) if IScoordinates else 0.0
        IScrosswalk_loss = self.IScrw_loss_fcn(predictions['IScrosswalk'],true_IScrosswalk)
        light_class_loss = self.LClass_loss_fcn(predictions['light_class'],true_light_class)

        loss = (coordinates_loss+IScrosswalk_loss+light_class_loss)/((1.0 if IScoordinates else 0.0)+2.0)

        return loss, coordinates_loss, IScrosswalk_loss, light_class_loss
