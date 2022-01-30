# Convert images to another format and delete duplicates

With script **preprocess_img/preprocess_imgs.py** 
you can preprocess you image datasets:
 - Convert images to another format
 - Reduce images sizes
 - Find similar images and delete image duplicates 

You can use script **preprocess_imgs.py** with parameters:

For example, to convert all images in folder `/datadir` convert to **jpeg** and the result size of images
equal to 80 percents of original image size. Delete source images if can save it in target format.
`python preprocess_imgs.py --images-dir /datadir --target-ext jpeg`

OR result size of images equal to 80 percents of original image size. 
`python preprocess_imgs.py --images-dir /datadir --target-ext jpeg --ratio-to-compress 0.8`

OR result size of images is not greater than 128000 bytes. 
`python preprocess_imgs.py --images-dir /datadir --target-ext jpeg --max-img-size 128000`

And run `python preprocess_imgs.py -h` to show help:

```
usage: preprocess_imgs.py [-h] --images-dir DIR --target-ext EXT
                          [--ratio-to-compress RATIO | --max-img-size NUM_BYTES]
                          [--no-delete-original] [--no-rename]
                          [--no-delete-duplicates] [--no-show-duplicates]

Convert images with size compressing and deleting duplicates in selected folder.
If folder have subfolders, then convert images in subfolders.

optional arguments:
  -h, --help            show this help message and exit
  --images-dir DIR      Path to folder with images (may contain subfolders)
                        (default: None)
  --target-ext EXT      Target extension to convert. (default: None)
  --ratio-to-compress RATIO
                        Ratio to compress size of image. Float number should
                        be between 0. and 1. (default: None)
  --max-img-size NUM_BYTES
                        Maximum size of each image in bytes. (default: None)
  --no-delete-original  Not delete original images after converting and
                        compressing. (default: True)
  --no-rename           Rename images according folder name and numerate them
                        (default: True)
  --no-delete-duplicates
                        Optional delete duplicates except one.Find for image
                        duplicates and optionally delete all but one.
                        (default: True)
  --no-show-duplicates  Optional show duplicates on the screen. By default
                        show all duplicates. (default: True)
```