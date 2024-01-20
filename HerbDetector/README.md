# Herb Detector by JJOverload

This project aims to help create a dataset for training an image classification model, which can potentially recognize different types of herbs.

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

Terminal command for converting png images to jpg:<br>
`mogrify -format jpg *.png`


-----------------------------------------------

NOTE: DO NOT USE GOOGLE IMAGES FOR COMMERCIAL USE!
