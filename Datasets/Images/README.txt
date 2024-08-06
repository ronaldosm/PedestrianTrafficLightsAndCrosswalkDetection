How to configure the datasets:
    PTL-Crosswalk Dataset:
        Download the PTL-Crosswalk-Base subset from < https://1drv.ms/f/c/ac5426c77d3b6a62/ElwZ_7ElwzxDv8V8zd7daCQBfp91XE85Rd86yMl1ctgo-w?e=1sfrqy >. Extract the files. Put the < PTL-Crosswalk-Base > folder inside < /Datasets/Images/PTL-Crosswalk Dataset/ >.
        Download the PTLR and Crosswalk subsets from < http://wangkaiwei.org/downloadeg.html >. Extract the files. Put the < PTLR dataset > and < crosswalk DATASET > folders inside < /Datasets/Images/PTL-Crosswalk Dataset/ >.
        Download the PedestrianLights subset from < https://www.uni-muenster.de/PRIA/en/forschung/index.shtml >. Extract the < pedestrianlights_large > folder. Run < /Codes/common_modules/CropPedestrianLights.py >, indicating the path to that folder, as well as the save_path of the cropped images. This save_path must be: < /Datasets/Images/PTL-Crosswalk Dataset/pedestrianlights_large/ >
    (Yu; Lee; Kim) PTL Dataset:
        Download the PTL Dataset from < https://github.com/samuelyu2002/ImVisible >. Extract the files. Put the < PTL_Dataset_768x576 > and < PTL_Dataset_876x657 > folders inside < /Datasets/Images/PTL Dataset >.
        Put the annotations files inside < /Datasets/Annotations/PTL_Dataset/ >.
