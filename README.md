## Convert images to another format

To convert images you can use script **convert_img.py** with parameters.


For example, to convert all images in folder `/Users/user/dataset/` convert to **jpeg** and the result size of images
equal to 80 percents of original image size. Delete source images if can save it in target format.

`python convert_img.py --images_dir datadir/ --target_ext jpeg --is_delete_original True --ratio_to_compress 0.8`

And run `python convert_img.py -h` to show help:

```
usage: convert_img.py [-h] --images_dir DIR --target_ext EXT
                            [--is_delete_original False]
                            [--ratio_to_compress RATIO | --max_img_size NUM_BYTES]

Convert images with optional size compressing.

optional arguments:
  -h, --help            show this help message and exit
  --images_dir DIR      Path to folder with images (may contain subfolders)
  --target_ext EXT      Target extension to convert.
  --is_delete_original False
                        Delete original images after converting and
                        compressing?
  --ratio_to_compress RATIO
                        Ratio to compress size of image. Float number should
                        be between 0. and 1.
  --max_img_size NUM_BYTES
                        Maximum size of each image in bytes.

```