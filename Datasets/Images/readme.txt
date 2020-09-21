How to configure the datasets:
1. Download the LytNet PTL Dataset from < https://github.com/samuelyu2002/ImVisible >
2. Extract the files. Put the <PTL_Dataset_768x576> and <PTL_Dataset_876x657> folders inside </Datasets/Images>. Put the annotations files inside </Datasets/Annotations/PTL_Dataset/>
3. Download the PTLR and Crosswalk Datasets from < http://wangkaiwei.org/downloadeg.html >
4. Rename the PTLR Dataset folder to <PTLR_Lights>, and the Crosswalk Dataset folder to <PTLR_Crosswalk>. Then, put the renamed folders inside </Datasets/Images/second_dataset/>
5. Download the PedestrianLights dataset from < https://www.uni-muenster.de/PRIA/en/forschung/index.shtml >
6. Extract the files. Run < /Codes/common_modules/CropPedestrianLights.py >, indicating the path to the extracted dataset, as well as the save_path of the cropped images. This save_path must be: <(Your_repository_location)/Datasets/Images/second_dataset/PedestrianLights/>
7. Download the complementary dataset from < https://dl.orangedox.com/k1Oi37 >
8. Extract the files. Put the images inside </Datasets/Images/second_dataset/Complementary Dataset>
