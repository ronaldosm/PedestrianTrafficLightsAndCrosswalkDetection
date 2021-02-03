# Imports
import os
import sys
import json
import torch
import ResNet50_M

home_dir = os.path.expanduser('~')
sys.path.append(home_dir + 'Path to: /Codes/common_modules')

import Dataset
import Loss

# Parameters
batch_size = 10
max_epochs = 400
learning_rate = 0.0001
images_size = [768,576]
load_state_dict = False
pretrained=True
loss_functions = ['MSE','MSE','CrossEntropy']
cuda = torch.cuda.is_available()

# Paths
PTL_train_ann_path = home_dir + 'Path to: /Datasets/Annotations/PTL Dataset/PTLR_training.csv'                    # PTL Dataset Training Annotations
PTL_train_img_path = home_dir + 'Path to: /Datasets/Images/PTL Dataset/PTL_Dataset_876x657/'                      # PTL Dataset Training Images

PTL_valid_ann_path = home_dir + 'Path to: /Datasets/Annotations/PTL Dataset/PTLR_validation.csv'                  # PTL Dataset Validation Annotations
PTL_valid_img_path = home_dir + 'Path to: /Datasets/Images/PTL Dataset/PTL_Dataset_768x576/'                      # PTL Dataset Validation Images

PTL_Crosswalk_img_path = home_dir + 'Path to: /Datasets/Images/PTL-Crosswalk Dataset/'                            # PTL-Crosswalk Dataset Images

PTL_Crosswalk_train_ann_path = home_dir + 'Path to: /Datasets/Annotations/PTL-Crosswalk Dataset/training.csv'     # PTL-Crosswalk Dataset Training Annotaions
PTL_Crosswalk_valid_ann_path = home_dir + 'Path to: /Datasets/Annotations/PTL-Crosswalk Dataset/validation.csv'   # PTL-Crosswalk Dataset Validation Annotaions

model_state_dict_path = home_dir + 'Path to: /Codes/CNNs/ResNet50-M/state_dict.pth'
train_history_save_path = home_dir + 'Path to: /Codes/CNNs/ResNet50-M/history.json'

# --------------------------------- Facilitate dataset choice ---------------------------------
dataset_dict = {'train':{'PTL':{'img_path':PTL_train_img_path,    'ann_path':PTL_train_ann_path},
               'PTL-Crosswalk':{'img_path':PTL_Crosswalk_img_path,'ann_path':PTL_Crosswalk_train_ann_path}},
                'valid':{'PTL':{'img_path':PTL_valid_img_path,    'ann_path':PTL_valid_ann_path},
               'PTL-Crosswalk':{'img_path':PTL_Crosswalk_img_path,'ann_path':PTL_Crosswalk_valid_ann_path}}}

def get_dataset_array(Type,Sources):
    global dataset_dict
    img_path_arr = []
    ann_path_arr = []

    Sources = [Sources] if Sources != 'complete' else ['PTL','PTL-Crosswalk']

    for dataset_source in Sources:
        img_path_arr.append(dataset_dict[Type][dataset_source]['img_path'])
        ann_path_arr.append(dataset_dict[Type][dataset_source]['ann_path'])

    return img_path_arr,ann_path_arr,Sources
# ---------------------------------------------------------------------------------------------


if __name__ == '__main__':
    # Training and validation datasets array
    train_img_path, train_ann_path, train_dtset_types = get_dataset_array(Type='train', Sources='complete')
    valid_img_path, valid_ann_path, valid_dtset_types = get_dataset_array(Type='valid', Sources='complete')

    # Datasets and dataloaders
    train_dataset = Dataset.make(train_img_path,train_ann_path,output_size=images_size,dataset_types=train_dtset_types,transformation=True )
    valid_dataset = Dataset.make(valid_img_path,valid_ann_path,output_size=images_size,dataset_types=valid_dtset_types,transformation=False)
    train_dataloader = torch.utils.data.DataLoader(train_dataset,batch_size=batch_size,shuffle=True, num_workers=2)
    valid_dataloader = torch.utils.data.DataLoader(valid_dataset,batch_size=batch_size,shuffle=False,num_workers=1)

    # Model
    model = ResNet50_M.MakeModel(pretrained=pretrained)
    if load_state_dict:
        model.load_state_dict(torch.load(model_state_dict_path))
        print('State-Dict Loaded Successfuly!')
    if cuda: model = model.cuda()

    # Loss and optimizer
    My_loss = Loss.FCN(loss_types=loss_functions,cuda=cuda)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=0.00001)

    # Train history arrays
    train_history = []
    valid_history = []

    # Print header
    print('Epoch | Coordinates loss | IScrosswalk loss | Light class loss | Train Loss | Vl. Coordinates loss | Vl. IScrosswalk loss | Vl. Light class loss | Valid Loss')

    # Main loop
    for epoch in range(max_epochs):
        # Train loop
        model.train()
        train_history.append([0.0, 0.0, 0.0])
        for i,data in enumerate(train_dataloader,0):
            # Targets and predictions
            images = data['image']
            true_coordinates = data['coordinates']
            true_IScrosswalk = data['IScrosswalk'].unsqueeze(1)
            true_light_class = data['light_class']
            if cuda:
                images = images.cuda()
                true_coordinates = true_coordinates.cuda()
                true_IScrosswalk = true_IScrosswalk.cuda()
                true_light_class = true_light_class.cuda()

            targets = [true_coordinates,true_IScrosswalk,true_light_class]
            predictions = model(images)

            # Loss calc.
            optimizer.zero_grad()
            loss, coordinates_loss, IScrosswalk_loss, light_class_loss = My_loss.compute(targets,predictions)
            loss.backward()
            optimizer.step()

            # Loss history
            train_history[epoch][0] += float(coordinates_loss)
            train_history[epoch][1] += float(IScrosswalk_loss)
            train_history[epoch][2] += float(light_class_loss)
        train_history[epoch][0] /= i+1
        train_history[epoch][1] /= i+1
        train_history[epoch][2] /= i+1

        # Validation loop
        model.eval()
        optimizer.zero_grad()
        with torch.no_grad():
            valid_history.append([0.0, 0.0, 0.0])
            for i,data in enumerate(valid_dataloader,0):
                # Targets and predictions
                images = data['image']
                true_coordinates = data['coordinates']
                true_IScrosswalk = data['IScrosswalk'].unsqueeze(1)
                true_light_class = data['light_class']
                if cuda:
                    images = images.cuda()
                    true_coordinates = true_coordinates.cuda()
                    true_IScrosswalk = true_IScrosswalk.cuda()
                    true_light_class = true_light_class.cuda()

                targets = [true_coordinates,true_IScrosswalk,true_light_class]
                predictions = model(images)

                # Loss calc.
                loss, coordinates_loss, IScrosswalk_loss, light_class_loss = My_loss.compute(targets,predictions)

                # Loss history
                valid_history[epoch][0] += float(coordinates_loss)
                valid_history[epoch][1] += float(IScrosswalk_loss)
                valid_history[epoch][2] += float(light_class_loss)
            valid_history[epoch][0] /= i+1
            valid_history[epoch][1] /= i+1
            valid_history[epoch][2] /= i+1

        # Prints
        print('\r',end='')
        print(' '+str(epoch+1).zfill(3)+'  | ',end='')
        print(f'     {train_history[epoch][0]:.4f}      | ',end='')
        print(f'     {train_history[epoch][1]:.4f}      | ',end='')
        print(f'     {train_history[epoch][2]:.4f}      | ',end='')
        print(f'  {sum(train_history[epoch])/3:.4f}   | ',  end='')
        print(f'       {valid_history[epoch][0]:.4f}        | ',end='')
        print(f'       {valid_history[epoch][1]:.4f}        | ',end='')
        print(f'       {valid_history[epoch][2]:.4f}        | ',end='')
        print(f'  {sum(valid_history[epoch])/3:.4f}')

        # Save model
        torch.save(model.state_dict(), model_state_dict_path)

        # Save train history as a json file
        with open(train_history_save_path,'w') as File:
            json.dump({'train':train_history,'valid':valid_history},File)
            File.close()
