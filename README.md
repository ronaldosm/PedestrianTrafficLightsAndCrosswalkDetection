# Crosswalks And Pedestrian Traffic Lights Identification

### Introduction
This repository hosts project that aims to develop software for real-time pedestrian traffic lights (PTLs) and zebra crosswalks identification.
The approach to performing these identifications consists of the adaptation of known and pre-trained CNN architectures, followed by their training with the datasets proposed bellow.

### Datasets
To achieve the goal of adding a decent quantity of images from various locations, which is required as PTLs differ in shape, we used images from three datasets available online, as shown in Table 1. To do this, we remade all their labels, as the originals were not suitable for our task. Also, we registered images of several street intersections, with some of them being at night or on raining days.

| Dataset Name             | Author      | Location     | TD   | TU   | Link |
|      ---                 |  ---        |   ---        | ---  | ---  |  --- |
| Crosswalk Dataset        | Ruiqi Cheng | China, Italy | 833  | 744  | [Link](http://wangkaiwei.org/downloadeg.html) |
| PTLR Dataset             | Ruiqi Cheng | China, Italy | 1135 | 1106 | [Link](http://wangkaiwei.org/downloadeg.html) |
| PedestrianLights - Large | Jan Roters  | Germany      | 501  | 501  | [Link](https://www.uni-muenster.de/PRIA/en/forschung/index.shtml) |
| Complementary Dataset    | Ours        | Brazil       | 1215 | 1215 | [Link](https://dl.orangedox.com/FeYAju) |

 **TD denotes the total number of images in its original dataset, and TU the number of images that we appended to make ours**

To use this dataset, you must perform the following steps:
1. Download and extract its four parts, shown in table 1. For the PedestrianLights dataset, only the following folder needs extraction: *pedestrianlights/download/pedestrianlights_large.zip*;
2. Execute the code "CropPedestrianLights.py", informing the path to the original images and annotations of the PedestrianLights dataset, by author Jan Roters. This code performs a random crop in each image, as the original has a much different aspect ratio from the ones of other datasets;
3. Organize the extracted parts as follows:

![alt text](https://github.com/ronaldosm/LightsAndCrosswalkDetect/blob/master/Figures/dataset_structure.png?raw=true)

Annotations of the resulting dataset are composed of the following parts:

* **image_file:** Path to the referred image;
* **Coordinates:** (X, Y) coordinates, comprised between 0 and 1, of the start point and the endpoint of the midline of the crosswalk. In the cases where there is no crosswalk in the image, the coordinates are an array of zeros;
* **IScrosswalk:** Equal to 1 if the image contains a crosswalk, and 0 otherwise;
* **light_class:** PTL class indicator. 0 correspond to red PTL, 1 to green PTL, and 2 to nonexistent PTL.

The next tables contains statistics about this dataset:

|          | Images with crosswalks| Images without crosswalks |
| ---      | ---                   | ---                       |
| Training | 1056 (50.94%)         | 1017 (49.06%)             |
| Testing  | 867 (58.07%)          | 626 (41.93%)              |

|          | Green PTL    | Red PTL      | No PTL       |
| ---      | ---          | ---          | ---          |
| Training | 532 (25.66%) | 702 (33.86%) | 839 (40.48%) |
| Testing  | 339 (22.71%) | 722 (48.36%) | 432 (28.93%) |

To further increase the CNNs performance for PTL class identification, we also used the PTL dataset from Yu; Lee; Kim [(Link)](https://github.com/samuelyu2002/ImVisible). For this specific dataset, since the image labels in this dataset contain all the necessary information for training, it was not required to redo them. However, as our CNNs outputs have only three classes, our code automatically merges the following ones: **green** and **countdown green**, and **red** and **countdown blank**. The table below contains statistics about the union of ours and this dataset:

|          | Green PTL     | Red PTL       | No PTL        |
| ---      | ---           | ---           | ---           |
| Training | 2059 (37.24%) | 2339 (42.30%) | 1131 (20.46%) |
| Testing  | 668  (29.93%) | 48.30 (48.36%) | 486 (21.77%) |

### Results
The tables below show the results obtained by each tested CNN architecture:
|               | **CNN Model**   | **Red PTL**  | **Green PTL** | **No PTL**   | **CP**   |
|  :---:        | :---:           | :---:        | :---:         | :---:        | :---:    |
| **Precision** | EfficientNet B2 | **0.98**     | **0.98**      | 0.90         | **0.99** |
|               | EfficientNet B1 | 0.97         | **0.98**      | 0.83         | **0.99** |
|               | EfficientNet B0 | **0.98**     | **0.98**      | **0.91**     | **0.99** |
|               | ResNet-50       | 0.91         | 0.97          | 0.81         | **0.99** |
|               | LytNetV2        | 0.96         | 0.97          | 0.82         | **0.99** |
|               |                 |              |               |              |          |
| **Recall**    | EfficientNet B2 | 0.97         | **0.93**      | 0.97         | **0.98** |
|               | EfficientNet B1 | 0.95         | 0.90          | 0.96         | 0.97     |
|               | EfficientNet B0 | **0.98**     | 0.92	         | **0.98**     | **0.98** |
|               | ResNet-50       | 0.93         | 0.87	         | 0.89         | 0.95     |
|               | LytNetV2        | 0.95         | 0.88	         | 0.93         | 0.97     |
|               |                 |              |               |              |          |
| **F1 Score**  | EfficientNet B2 | 0.97         | **0.95**      | 0.93         | **0.98** |
|               | EfficientNet B1 | 0.96         | 0.94	         | 0.89         | **0.98** |
|               | EfficientNet B0 | **0.98**     | **0.95**      | **0.94**     | **0.98** |
|               | ResNet-50       | 0.92         | 0.92	         | 0.85         | 0.97     |
|               | LytNetV2        | 0.95         | 0.92	         | 0.87         | **0.98** |

**CP denotes the crosswalk presence indication variable**

| **CNN Model**   | **Start point error** | **Endpoint error** |
| :---:           | :---:                 | :---:              |
| EfficientNet B2 | **0.0662**            | **0.0988**         |
| EfficientNet B1 | 0.0723	              | 0.1025             |
| EfficientNet B0 | 0.0902	              | 0.1082             |
| ResNet-50       | 0.0965	              | 0.1346             |
| LytNetV2        | 0.0734	              | 0.1088             |

### Credits
1. Ruiqi Cheng. 2018. Image Data Sets.   http://wangkaiwei.org/downloadeg.html
2. Ruiqi Cheng, Kaiwei Wang, Kailun Yang, Ningbo Long, Jian Bai, and Dong Liu. 2017. Real-time pedestrian crossing lights detection algorithm for the visually impaired. Multimedia Tools and Applications 77 (12 2017).   https://doi.org/10.1007/s11042-017-5472-5
3. Jan Roters. 2019. Databases.   https://www.uni-muenster.de/PRIA/en/forschung/index.shtml
4. Samuel Yu, Heon Lee, and Junghoon Kim. 2019. Street Crossing Aid Using Light-Weight CNNs for the Visually Impaired. In: The IEEE International Conference on Computer Vision (ICCV) Workshops.
