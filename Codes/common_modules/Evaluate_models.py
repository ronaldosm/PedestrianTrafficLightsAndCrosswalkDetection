# Imports
import os
import sys
import torch

import Dataset
import Statistics

home_dir = os.path.expanduser('~')
sys.path.append(home_dir + 'Path to: /Codes/CNNs/EfficientNet')
sys.path.append(home_dir + 'Path to: /Codes/CNNs/LytNetV2')
sys.path.append(home_dir + 'Path to: /Codes/CNNs/ResNet')
sys.path.append(home_dir + 'Path to: /Codes/CNNs/MixNet')

import EfficientNet
import my_lytnetV2
import ResNet50
import MixNet

# Parameters
batch_size = 8
images_size = [768,576]
models_list = {'ResNet' : 0, # 1 -> Use this model, 0 -> Don't use it
               'MixNet' : 0,
               'LytNet' : 1,
               'EffNet2': 1,
               'EffNet1': 0,
               'EffNet0': 0
               }

# Paths
PTL_train_ann_path = home_dir + 'Path to: /Datasets/Annotations/PTL_Dataset/PTLR_training.csv'   # Lytnet Dataset Training Annotations
PTL_train_img_path = home_dir + 'Path to: /Datasets/Images/PTL_Dataset_876x657/'                 # Lytnet Dataset Training Images

Sec_train_ann_path = home_dir + 'Path to: /Datasets/Annotations/second_dataset/training.csv'     # Second Dataset Training Annotaions
Sec_train_img_path = home_dir + 'Path to: /Datasets/Images/second_dataset/'                      # Second Dataset Training Images

PTL_valid_ann_path = home_dir + 'Path to: /Datasets/Annotations/PTL_Dataset/PTLR_validation.csv' # Lytnet Dataset Validation Annotations
PTL_valid_img_path = home_dir + 'Path to: /Datasets/Images/PTL_Dataset_768x576/'                 # Lytnet Dataset Validation Images

Sec_valid_ann_path = home_dir + 'Path to: /Datasets/Annotations/second_Dataset/testing.csv'      # Second Dataset Validation Annotaions
Sec_valid_img_path = home_dir + 'Path to: /Datasets/Images/second_dataset/'                      # Second Dataset Validation Images

PTL_tests_ann_path = home_dir + 'Path to: /Datasets/Annotations/PTL_Dataset/testing_file.csv'    # Lytnet Dataset Testing Annotations
PTL_tests_img_path = home_dir + 'Path to: /Datasets/Images/PTL_Dataset_768x576/'                 # Lytnet Dataset Testing Images

ResNet_StateDict  = 'Path to: /Codes/CNNs/ResNet/my_model.pth'          # ResNet 50 state-dict path
MixNet_StateDict  = 'Path to: /Codes/CNNs/MixNet/my_model.pth'          # MixNet state-dict path
LytNet_StateDict  = 'Path to: /Codes/CNNs/LytNetV2/my_model.pth'        # LytnetV2 state-dict path
EffNet0_StateDict = 'Path to: /Codes/CNNs/EfficientNet/B0/my_model.pth' # EfficientNet B0 state-dict path
EffNet1_StateDict = 'Path to: /Codes/CNNs/EfficientNet/B1/my_model.pth' # EfficientNet B1 state-dict path
EffNet2_StateDict = 'Path to: /Codes/CNNs/EfficientNet/B2/my_model.pth' # EfficientNet B2 state-dict path

# --------------------------------- Facilitate dataset choice ---------------------------------
dataset_dict = {'train':{'PTL':{'img_path':PTL_train_img_path,'ann_path':PTL_train_ann_path},
                      'second':{'img_path':Sec_train_img_path,'ann_path':Sec_train_ann_path}},
                'valid':{'PTL':{'img_path':PTL_valid_img_path,'ann_path':PTL_valid_ann_path},
                      'second':{'img_path':Sec_valid_img_path,'ann_path':Sec_valid_ann_path}},
                'tests':{'PTL':{'img_path':PTL_tests_img_path,'ann_path':PTL_tests_ann_path},
                      'second':{'img_path':Sec_valid_img_path,'ann_path':Sec_valid_ann_path}}}

def get_dataset_array(Type,Sources):
    global dataset_dict
    img_path_arr = []
    ann_path_arr = []

    Sources = [Sources] if Sources != 'complete' else ['PTL','second']

    for dataset_source in Sources:
        img_path_arr.append(dataset_dict[Type][dataset_source]['img_path'])
        ann_path_arr.append(dataset_dict[Type][dataset_source]['ann_path'])

    return img_path_arr,ann_path_arr,Sources
# ---------------------------------------------------------------------------------------------

if __name__ == '__main__':
    # Datasets and dataloaders:
    img_path, ann_path, dtset_types = get_dataset_array(Type='tests', Sources='complete')
    Dataset = Dataset.make(img_path,ann_path,output_size=images_size,dataset_types=dtset_types,transformation=False)
    dataloader = torch.utils.data.DataLoader(Dataset,batch_size=batch_size,shuffle=False, num_workers=0)

    # Models and state-dicts
    ResNet, Mixnet, LytNet,EffNet0,EffNet1,EffNet2 = [None,None,None,None,None,None]
    using_models, names = [],[]
    if models_list['EffNet2']:
        EffNet2 = EfficientNet.MakeModel('efficientnet_b2').cuda()
        EffNet2.load_state_dict(torch.load(EffNet2_StateDict))
        using_models.append(EffNet2.eval())
        names.append('EfficientNet B2')
    if models_list['EffNet1']:
        EffNet1 = EfficientNet.MakeModel('efficientnet_b1').cuda()
        EffNet1.load_state_dict(torch.load(EffNet1_StateDict))
        using_models.append(EffNet1.eval())
        names.append('EfficientNet B1')
    if models_list['EffNet0']:
        EffNet0 = EfficientNet.MakeModel('efficientnet_b0').cuda()
        EffNet0.load_state_dict(torch.load(EffNet0_StateDict))
        using_models.append(EffNet0.eval())
        names.append('EfficientNet B0')
    if models_list['LytNet']:
        LytNet = my_lytnetV2.MakeModel().cuda()
        LytNet.load_state_dict(torch.load(LytNet_StateDict))
        using_models.append(LytNet.eval())
        names.append('LytNet V2')
    if models_list['ResNet']:
        ResNet = ResNet50.MakeModel().cuda()
        ResNet.load_state_dict(torch.load(ResNet_StateDict))
        using_models.append(ResNet.eval())
        names.append('ResNet 50')
    if models_list['MixNet']:
        Mixnet = MixNet.MakeModel().cuda()
        Mixnet.load_state_dict(torch.load(MixNet_StateDict))
        using_models.append(Mixnet.eval())
        names.append('MixNet')

    # Measures
    Mi = {} # Model IScrosswalk variable
    Ti = [] # True IScrosswalk variable
    Mc = {} # Model Coordinates
    Tc = [] # True Coordinates
    Ml = {} # Model Light Class
    Tl = [] # True Light Class

    for i in range(len(using_models)):
        Mi[str(i)] = []
        Mc[str(i)] = []
        Ml[str(i)] = []

    # Iteration over the dataset
    with torch.no_grad():
        for i,data in enumerate(dataloader):
            # Images and original labels
            images = data['image'].cuda()
            true_coordinates = data['coordinates']
            true_IScrosswalk = data['IScrosswalk']
            true_light_class = data['light_class']

            Ti.extend(true_IScrosswalk)
            Tc.extend(true_coordinates)
            Tl.extend(true_light_class)
            # Predictions
            for N,model in enumerate(using_models,0):
                # Outputs
                outputs = model(images)
                output_coordinates = outputs['coordinates'].cpu() 
                output_coordinates = [[min(max(0,float(val)),1) for val in array] for array in output_coordinates] # Restricts coordinates values between 0 and 1
                output_IScrosswalk = outputs['IScrosswalk'].cpu().squeeze()
                output_light_class = [torch.argmax(item) for item in outputs['light_class'].cpu()]

                # Extend output arrays
                Mi[str(N)].extend(output_IScrosswalk)
                Mc[str(N)].extend(output_coordinates)
                Ml[str(N)].extend(output_light_class)

            print('\r',end='')
            print(f'Obtaining Predictions: {(i/(len(dataloader)-1))*100:.2f} %',end='')
    print('\r',end='')


    # Statistics Calc.
    for N,model in enumerate(using_models,0):
        # Start-point And End-point Error
        start_error,end_error = Statistics.EQM(Mc[str(N)],Tc,Ti)
        print('\n\n\n\033[1mModel: '+names[N]+'\033[0m')
        print(f'Start Error: {start_error:.4f}, End Error: {end_error:.4f}',end='\n\n')

        # IScrosswalk Variable: Precision, Recall, And Confusion Matrix
        IScrw_confusion_matrix= Statistics.confusion_matrix(Mi[str(N)],Ti,n_classes=2)
        IScrw_precision,IScrw_recall = Statistics.precision_recall(IScrw_confusion_matrix)
        print(f'IScrosswalk Precision: {IScrw_precision:.4f}, IScrosswalk Recall: {IScrw_recall:.4f}')
        print('IScrosswalk Confusion Matrix:')
        print(IScrw_confusion_matrix.numpy(),end='\n\n')

        # Light-Class Variable: Precision, Recall, And Confusion Matrix
        LClass_confusion_matrix= Statistics.confusion_matrix(Ml[str(N)],Tl,n_classes=3)
        LClass_precision,LClass_recall = Statistics.precision_recall(LClass_confusion_matrix)
        print(f'Light-Class Precision: {LClass_precision[0]:.2f}, {LClass_precision[1]:.2f}, {LClass_precision[2]:.2f}, ',end='')
        print(f'Light-Class Recall: {LClass_recall[0]:.2f}, {LClass_recall[1]:.2f}, {LClass_recall[2]:.2f}')
        print('Light-Class Confusion Matrix:')
        print(LClass_confusion_matrix.numpy())