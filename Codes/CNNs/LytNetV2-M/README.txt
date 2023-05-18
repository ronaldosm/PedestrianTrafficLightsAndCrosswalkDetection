Before running <training.py>, the following files must be added to this directory:
* LYTNetV2.py
* LytNetV2_weights
These files can be found at <https://github.com/samuelyu2002/ImVisible>

Besides that, in LYTNetV2.py, the line 153 must be changed from "self.features.append(nn.AvgPool2d(12,9))" to "self.features.append(nn.AvgPool2d(9,12))".
