# Arch style detection project

You can try it in [telegram bot](https://t.me/arch_styles_bot).

Computer vision model based on neural networks for detection
several architectural styles. 

Current architecture styles classification is relevant for Russia. 
In European or American architecture classification consist many styles
not present in Russia and vice versa.

There are 21 arch styles that our model can be classify:
1. Авангард (The avant-garde)                                               
1. Ар-деко (Art Deco)                                                
1. Барокко (Baroque)                                                
1. Брутализм (Brutalism)                                             
1. Готика (Gothic)                                                  
1. Деконструктивизм (Deconstructivism)                                        
1. Древнерусская архитектура (Ancient Russian architecture)                             
1. Капиталистический романтизм (Capitalist Romanticism)                           
1. Кирпичный стиль (Brick style)                                        
1. Классицизм (Classicism)    
1. Модерн (Modern)                                         
1. Модернизм (Modernism)
1. Неоклассицизм (Neoclassicism)
1. Неорусский стиль (Neo-Russian style)
1. Постмодернизм (Postmodernism)
1. Ренессанс (Renaissance)
1. Романика (Romance)
1. Русское барокко (Russian Baroque)
1. Русское деревянное зодчество (Russian wooden architecture)
1. Современная архитектура (Modern architecture)
1. Узорочье (Uzorochye)

## Convert images to another format and delete duplicates

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