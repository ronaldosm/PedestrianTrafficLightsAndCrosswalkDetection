# Pedestrian Traffic Lights And Crosswalk Identification

### Introduction
This repository hosts a project that consists of the training and evaluation of state-of-the-art convolutional neural network (CNN) architectures for the real-time identification of the presence and coordinates of zebra crosswalks and the state of pedestrian traffic lights (PTL), with the use of a new large and culturally diverse image dataset. Paper link: https://link.springer.com/article/10.1007/s11042-022-12222-6

## CNN Architectures
For the first tests, we used the CNN [LytNetV2](https://github.com/samuelyu2002/ImVisible) since its development attends a very similar purpose. Nevertheless, its last layer was replaced by another that includes a new output variable to indicate the presence of crosswalks in the image and change the PTL class output from five possible classes to three. We name the modified CNN LytNetV2-M.

Then, we carried research to choose light and efficient state-of-the-art CNNs commonly used for object detection tasks to adopt them in this work. The selected architectures were ResNet50 and EfficientNet, in its B0, B1, and B2 variants. For each of these architectures, we attached an extra linear layer to its final one so that the model outputs only the information about the crosswalks and the PTLs, instead of its original one-thousand classes output designed for classification on the ImageNet database. We also name the modified architectures Resnet50-M and EfficientNet-M.

### Datasets
Our proposed dataset, named as PTL-Crosswalk Dataset, consists of two parts. Firstly, we used a smartphone in a horizontal position, with a resolution set to 1920x1080, to make 100 footages of different street intersections, and also received two videos from the ADAPT Project team, containing several street intersections in each video. We split these videos into 2711 images, including 687 made at night and 748 on rainy days.

For the second part, we attached three datasets available online to ours, as shown in the table below, to achieve the goal of adding a decent quantity of images from various locations, which is required as PTLs differ in shape.

| Dataset Name               | Author             | Location       | Number of images   | Link |
|      ---                   |  ---               |   ---          | ---                | ---  |
| Crosswalk Dataset          | Ruiqi Cheng        | China, Italy   | 833                | [Link](http://wangkaiwei.org/downloadeg.html) |
| PTLR Dataset               | Ruiqi Cheng        | China, Italy   | 1135               | [Link](http://wangkaiwei.org/downloadeg.html) |
| PedestrianLights - Large   | Jan Roters         | Germany        | 501                | [Link](https://www.uni-muenster.de/PRIA/en/forschung/index.shtml) |
| PTL-Crosswalk Base Dataset | Ours, ADAPT Project| Brazil, France | 2711               | [Link](https://drive.google.com/file/d/1xi4LJnp7pTeP0lonrSqkhCbCS_IbytQe/view?usp=sharing) |

For all images, we manually generated and conferred new labels, composed of the following parts:
* **image_file:** Path to the referred image;
* **Coordinates:** (X, Y) coordinates, comprised between 0 and 1, of the start point and the endpoint of the midline of the crosswalk. In the cases where there is no crosswalk in the image, the coordinates are an array of zeros;
* **IScrosswalk:** Equal to 1 if the image contains a crosswalk, and 0 otherwise;
* **light_class:** PTL class indicator. 0 correspond to red PTL, 1 to green PTL, and 2 to nonexistent PTL.

The next tables contains statistics about our dataset:

|            | Images with crosswalks| Images without crosswalks |
| ---        | ---                   | ---                       |
| Training   | 1951 (60.7%)          | 1263 (39.3%)              |
| Validation | 607 (61.4%)           | 382 (38.6%)               |
| Testing    | 589 (60.3%)           | 388 (39.7%)               |

|            | Green PTL    | Red PTL      | No PTL       |
| ---        | ---          | ---          | ---          |
| Training   | 678 (21.1%)  | 854 (26.6%)  | 1682 (52.3%) |
| Validation | 185 (18.7%)  | 396 (40.0%)  | 408 (41.3%)  |
| Testing    | 193 (19.7%)  | 396 (40.5%)  | 388 (39.8%)  |

After the end of the training process with our proposed dataset, to further increase the CNNs performance for PTL class identification, we made a second training step, where we appended the PTL dataset from Yu, Lee, and Kim [(Link)](https://github.com/samuelyu2002/ImVisible). For this specific dataset, since the image labels contain all the necessary information, it was not required to redo them. However, as our CNNs outputs have only three classes, our code automatically merges the following ones: **green** and **countdown green**, and **red** and **countdown blank**. The table below contains statistics about the union of ours and this dataset:

|            | Green PTL     | Red PTL       | No PTL        |
| ---        | ---           | ---           | ---           |
| Training   | 2205 (33.0%)  | 2491 (37.3%)  | 1974 (29.7%)  |
| Validation | 595 (32.1%)   | 784 (42.3%)   | 474 (25.6%)   |
| Testing    | 522  (30.4%)  | 752 (43.8%)   | 442 (25.8%)   |

### Results
The tables below show the results obtained by each tested CNN architecture:
|               | **CNN Model**     | **Red PTL**   | **Green PTL** | **No PTL**   | **CP**    |
|  :---:        | :---:             | :---:         | :---:         | :---:        | :---:     |
| **Precision** | EfficientNet-M B0 | **0.979**     |   0.984       | **0.943**    |   0.977   |
|               | EfficientNet-M B1 |   0.976       |   0.988       |   0.929      | **0.986** |
|               | EfficientNet-M B2 |   0.977       |   0.986       |   0.935      |   0.984   |
|               | LytNetV2-M        |   0.966       |   0.976       |   0.868      |   0.973   |
|               | ResNet-50-M       |   0.977       | **0.992**     |   0.915      |   0.985   |
|               |                   |               |               |              |           |
| **Recall**    | EfficientNet-M B0 |   0.976       | **0.954**     | **0.982**    |   0.993   |
|               | EfficientNet-M B1 |   0.971       |   0.952       |   0.977      | **0.995** |
|               | EfficientNet-M B2 | **0.980**     |   0.950       |   0.971      |   0.992   |
|               | LytNetV2-M        |   0.947       |   0.902	      |   0.964      |   0.980   |
|               | ResNet-50-M       |   0.968       |   0.946	      |   0.980      |   0.994   |
|               |                   |               |               |              |           |
| **F1 Score**  | EfficientNet-M B0 |   0.977       |   0.969       | **0.962**    | **0.980** |
|               | EfficientNet-M B1 |   0.973       | **0.970**     |   0.952      | **0.990** |
|               | EfficientNet-M B2 | **0.978**     |   0.968       |   0.953      |   0.988   |
|               | LytNetV2-M        |   0.956       |   0.937	      |   0.913      | **0.976** |
|               | ResNet-50-M       |   0.972       |   0.968	      |   0.946      |   0.989   |

**CP denotes the crosswalk presence indication variable**

| **CNN Model**   | **Multi-Class Accuracy (CP + PTL Class)** |
| :---:           | :---:                                     |
| EfficientNet B0 |   0.950                                   |
| EfficientNet B1 | **0.953**                                 |
| EfficientNet B2 | **0.953**                                 |
| ResNet-50       |   0.904	                                  |
| LytNetV2        |   0.945	                                  |

| **CNN Model**   | **Start point error {STD}** | **Endpoint error {STD}** |
| :---:           | :---:                       | :---:                    |
| EfficientNet B0 | 0.0851 {0.0651}             | 0.1031 {0.0821}          |
| EfficientNet B1 | **0.0725 {0.0616}**         | **0.0916 {0.0738}**      |
| EfficientNet B2 | 0.0830 {0.0648}             | 0.0954 {0.0785}          |
| LytNetV2        | 0.0787 {0.0749}             | 0.1098 {0.0983}          |
| ResNet-50       | 0.0970 {0.0780}             | 0.1187 {0.0990}          |

### Credits
1. Ruiqi Cheng. 2018. Image Data Sets.   http://wangkaiwei.org/downloadeg.html
2. Ruiqi Cheng, Kaiwei Wang, Kailun Yang, Ningbo Long, Jian Bai, and Dong Liu. 2017. Real-time pedestrian crossing lights detection algorithm for the visually impaired. Multimedia Tools and Applications 77 (12 2017).   https://doi.org/10.1007/s11042-017-5472-5
3. Jan Roters. 2019. Databases.   https://www.uni-muenster.de/PRIA/en/forschung/index.shtml
4. Samuel Yu, Heon Lee, and Junghoon Kim. 2019. Street Crossing Aid Using Light-Weight CNNs for the Visually Impaired. In: The IEEE International Conference on Computer Vision (ICCV) Workshops.
