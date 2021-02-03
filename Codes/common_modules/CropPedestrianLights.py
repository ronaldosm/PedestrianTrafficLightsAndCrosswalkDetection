import glob
import random
from PIL import Image

original_images_dir = 'Path to: /pedestrianlights/download/pedestrianlights_large/'
original_ann_dir = 'Path to: /pedestrianlights/download/pedestrianlights_large/groundtruth_large.txt'
cropped_images_dir = 'Path to: /Datasets/Images/PTL-Crosswalk Dataset/pedestrianlights_large/'

with open(original_ann_dir,'r') as File:
    original_ann = File.readlines()
    File.close()

# Convert the TXT file to a list of ints
original_ann = [item.replace('\n','').split(' ') for item in original_ann]
original_ann = [[int(item) for item in List] for List in original_ann]

# Crop Images
for i,item in enumerate(glob.glob(original_images_dir+'*.jpg'),0):
    name = item.split('/')[-1:][0].split('.')[0]
    y_candidates = [ann for ann in original_ann if int(name) == ann[0]]
    if y_candidates == []:
        light_y_coord = 160
    else:
        flag = 0
        for candidate in y_candidates:
            if candidate[6] == 1:
                light_y_coord = candidate[1]
                flag = 1
        if not flag:
            light_y_coord = y_candidates[0][1]

    if light_y_coord>160: light_y_coord = 160
    y1 = random.randint(0,light_y_coord)
    y2 = y1+480

    box = (0,y1,480,y2)
    save_name = cropped_images_dir+name+'.jpg'
    Image.open(original_images_dir+name+'.jpg').crop(box).save(save_name)
