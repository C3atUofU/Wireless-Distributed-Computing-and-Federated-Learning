from PIL import Image
# PIL is the Python Imaging Library, now defunct but Pillow is the one that replaced it and is backwards compatible
# Image is a module that provides functions and classes to represent PIL images

import glob
# glob is a module that finds path names matching a specified pattern, able to manipulate directories and sub-directories

# for the test folder file path, only has one sub-folder (0) with 115,462 images
# for filename in glob.glob(r'C:\Users\...\*\*.jpg'):

# for the train folder file path, iterates through the sub-folders (from 0-14950) with a variable amount of images per folder
# note: goes through folders that start with 0 first, then 1 (1, 10, 11, 12, .., 1000, 1001, .., etc), following the same pattern for 2-9
# enter file path in ...

for filename in glob.glob(r'C:\...\*\*.jpg'):
    # identifies and opens file; convert image from RGB ('RGB') to greyscale ('L'); keeps file open
    img = Image.open(filename).convert('L')
    # save as original filename
    img.save(filename)
    # see which file is being converted, unnecessary, but helpful when processing large quantities of images
    print(filename)

    # loads the pixel data and closes the file associated with the image to prevent having too many open processes creating errors
    img.load()
