## Helper GUI for manual tagging/cropping
--------
A GUI to help with manual tagging and cropping. Written in Python/Qt5.
![example](https://github.com/arenatemp/sd-tagging-helper/raw/master/screenshot.png)

### Requirements
Pillow and PyQt5
```
pip install pillow pyqt5
```

### Workings
Operates on a folder of input images, each image can optionally have a tag file:
```
img0000.png
WITH img0000.txt OR img0000.png.txt
```
The txt file should contain comma seperated tags.
These inputs are left untouched, all changes made are saved into the metadata folder as json files. So changes are persistant but also no data loss is possible.

When ready the inputs can be packaged into the output folder, in this folder the images will be cropped/resized and the modified tags written.
The two modes of packaging are:
- Single image
- Image/prompt pairs

Single image will save an image with the filename set to the prompt.
Image/prompt pairs will save the image with its original name and a corresponding text file containing the prompt (like the input folder format)

The square overlay shows the cropped view, position the image how you would like it cropped. Red indicates that the position has been modified but not saved.

### Usage
The minimal usage is just specifying the input folder. A metadata and output folder will be created in the current directory, and the output dimension will be 1024x1024:
```
python helper.py --input "path/to/input/folder"
```

### Controls
#### Mouse
```
Middle mouse - drag the image
Scroll - zoom the image
Double click - removes an active tag or adds an inactive tag
Left click - dividers can be dragged to resize the tag lists
```
#### Keyboard
```
F - save metadata
R - reset image position to last saved state
N - auto position image. contain mode
M - auto position image. fill mode
WASD - move image

LEFT  - previous image
RIGHT - next image
UP    - move selected tag up
DOWN  - move selected tag down
```

### Compiling
If you change the GUI `.qml` files the `qml_rc.py` will need to be recompiled:
```
pyrcc5 -o qml_rc.py qml/qml.qrc
```