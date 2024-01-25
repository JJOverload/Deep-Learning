# Herb Detector by JJOverload

This project aims to help create a dataset, as well as train an image classification model, which can potentially recognize different types of herbs (E.g. Rosemary, Sage, Thyme).

-----------------------------------

**Step 1: Gathering Unfiltered Google Images for Dataset**

Check out this repo for more information of how to using "simple_image_download":<br>
https://github.com/RiddlerQ/simple_image_download

(However, I did find that the quick start code needed some updating/fixing in order to run. 'get-herb-images.py' has worked with this so far.)

From scratch, get images by running:<br>
`python3 get-herb-images.py`

(Be sure to remove existing images from directory before running again... Unless you want duplicates!)

----------------------------------------------

**Step 2: Filter Out Bad Training Images**

Would have to manually take out certain images from the dataset since Google Images has both good and bad images for training.

Resizing will be done through the pipeline, so don't have to worry too much about that (from my experience).

Terminal command for converting png images to jpg. (Note: You might be overwriting your data for this.):<br>
`mogrify -format jpg *.png`


-----------------------------------------------


**Step 3: Expand Image Set with More Rotations**

Make another copy of the folder(s), and then rotate new images as desired. Repeat copy and rotate steps as desired to expand image set.

Terminal command for rotating images in present working directory (90 degrees counter-clockwise, in this example). (Note: Again, you will overwrite your images in the folder):<br>
`mogrify -rotate -90 *jpeg`


TODO:
Bash script to help with trimming spaces from directories.
trimmed_string=$(echo $string | tr -d ' ')



-----------------------------------------------

**Getting Code for Plot Model**

`pip install pydot-ng`

According to (https://graphviz.gitlab.io/download/):<br>
`sudo apt install graphviz`

-----------------------------------------------
TODO:
so far 60 percent before heavy fluctuations in val loss. Next goal, either be able to train in stable manner for more epoches, or increase val accuracy before heavy val loss fluctuations, or both. Might need more images.


**Useful Links**
https://medium.com/@bdhuma/which-pooling-method-is-better-maxpooling-vs-minpooling-vs-average-pooling-95fb03f45a9

https://datascience.stackexchange.com/questions/65471/validation-loss-much-higher-than-training-loss

https://stackoverflow.com/questions/66785014/how-to-plot-the-accuracy-and-and-loss-from-this-keras-cnn-model

https://chartio.com/resources/tutorials/how-to-save-a-plot-to-a-file-using-matplotlib/


NOTE: DO NOT USE GOOGLE IMAGES FOR COMMERCIAL USE!
