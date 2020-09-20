import torch
import random
import torchvision
import pandas as pd
from PIL import Image,ImageFile,ImageFilter


def make_annotations(images_path,annotations_path,dataset_types):
    annotations_list = []

    if type(dataset_types) is str: dataset_types,images_path,annotations_path = [dataset_types],[images_path],[annotations_path]

    for DTtype,IMGpath,ANNpath in zip(dataset_types,images_path,annotations_path):
        csv_file = pd.read_csv(ANNpath)
        if DTtype == 'PTL':
            for _,line in csv_file.iterrows():
                img_path = IMGpath+line['file']
                coordinates = [line['x1']/4032,line['y1']/3024,line['x2']/4032,line['y2']/3024]
                light_class = line['mode']
                if light_class == 2: light_class = 1
                elif light_class == 3: light_class = 0
                elif light_class == 4: light_class = 2
                else: pass
                annotations_list.append([img_path,coordinates,light_class,1])

        elif DTtype == 'second':
            for _,line in csv_file.iterrows():
                img_path = IMGpath+line['image_file']
                coordinates = line['coordinates'].replace('[','').replace(']','').split(',')
                coordinates = [float(item) for item in coordinates]
                IScrosswalk = line['IScrosswalk']
                light_class = line['light_class']
                annotations_list.append([img_path,coordinates,light_class,IScrosswalk])

        else: raise NotImplementedError

    return annotations_list


class make(torch.utils.data.Dataset):
    """
    images_path: path to your images folder
    annotations_path: path to your annotations.csv
    output_size: size of output images
    dataset_types: PTL -> original dataset of LytNet CNN. second -> secondary dataset composed of images with and without crosswalk
    transformation: [probability of 50%] -> gaussian blur, change of brightness, contrast and saturation
    """
    def __init__(self,images_path, annotations_path, output_size=[768,768], dataset_types='lytnet', transformation=True):
        self.transformation = transformation
        self.output_size = output_size
        self.annotations_list = make_annotations(images_path,annotations_path,dataset_types)

    def __len__(self): return len(self.annotations_list)

    def __getitem__(self,index):
        ImageFile.LOAD_TRUNCATED_IMAGES = True

        image_path = self.annotations_list[index][0]
        coordinates = self.annotations_list[index][1]

        image = Image.open(image_path)

        if self.transformation:
            if torch.rand(1) > 0.5: image = image.filter(ImageFilter.GaussianBlur(radius=random.randint(1,3)))

        if self.output_size is not None: image = image.resize((self.output_size[0],self.output_size[1]), Image.BILINEAR)
        if self.transformation:
            transforms = torchvision.transforms.Compose([torchvision.transforms.ColorJitter(0.1, 0.1, 0.1, 0.02), torchvision.transforms.ToTensor()])
            image = transforms(image)
        else: image = torchvision.transforms.functional.to_tensor(image)
        coordinates = torch.tensor(coordinates).type(torch.float32)
        light_class = torch.tensor(self.annotations_list[index][2]).type(torch.float32)
        IScrosswalk = torch.tensor(self.annotations_list[index][3]).type(torch.float32)

        return {'image':image*255,'coordinates':coordinates,'light_class':light_class,'IScrosswalk':IScrosswalk}
