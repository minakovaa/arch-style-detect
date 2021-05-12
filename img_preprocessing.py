import io
import math
import sys
import os
from PIL import Image
import argparse


def save_img_with_target_size(img, filename, format='webp', target_size=None):
    """Save the image as .format with the given name at best quality that makes less than "target_size" bytes"""
    # Min and Max quality

    Qmin, Qmax = 25, 98
    # Highest acceptable quality found
    Qacc = -1

    while Qmin <= Qmax and target_size is not None:
        m = math.floor((Qmin + Qmax) / 2)

        # Encode into memory and get size
        buffer = io.BytesIO()
        img.save(buffer, format=format, quality=m)
        s = buffer.getbuffer().nbytes

        if s <= target_size:
            Qacc = m
            Qmin = m + 1
        elif s > target_size:
            Qmax = m - 1

    # Write to disk at the defined quality
    if Qacc > -1:
        img.save(filename, format=format, quality=Qacc)
    elif target_size is None:
        img.save(filename, format=format)
    else:
        print("ERROR: No acceptble quality factor found", file=sys.stderr)


def convert_img(fname, to_ext='webp', is_delete_src_img=False, from_ext=None, target_size=None):
    prefix, ext = os.path.splitext(fname)

    if not ext:
        print(f'File "{fname}" has no extension!')
        return

    if from_ext is None:
        from_ext = ext.strip('.')
        from_ext = from_ext.lower()

    to_ext = to_ext.lower()

    # If the same extensions and should not compress
    if (from_ext == to_ext or
        (from_ext.endswith(('jpg', 'jpeg')) and to_ext.endswith(('jpg', 'jpeg')))) \
            and target_size is None:
        return

    if ext.endswith(from_ext):
        img = Image.open(fname).convert('RGB')

        # img.save(prefix+'.'+to_ext, to_ext, quality=80, optimize=True, progressive=True, )
        out_filename = prefix + '.' + to_ext

        save_img_with_target_size(img, out_filename, format=to_ext, target_size=target_size)

        if is_delete_src_img:
            os.remove(fname)

    else:
        print(f'Extention of image "{fname}" is not "*.{from_ext}"')


def rename_files_in_folder(folder_path: str, common_name: str):
    """
    Rename files in folder 'folder_path' to {common_name}_0, {common_name}_1, ...
    with saving their extensions.
    """

    for num, file_name in enumerate(os.listdir(folder_path), 1):
        name, ext = os.path.splitext(file_name)

        if not ext:
            continue

        os.rename(
            src=os.path.join(folder_path, file_name),
            dst=os.path.join(folder_path, ''.join((common_name, '_', str(num), ext))),
        )


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Convert images with optional size compressing.')
    parser.add_argument('--images_dir', metavar='path', required=True,
                        help='Path to folder with images (may contain subfolders)')

    parser.add_argument('--target_ext', type=str, required=True,
                        help='Target extension to convert.')

    parser.add_argument('--ratio_to_compress', type=float, default=None, required=False,
                        help='Ratio to compress size of image. Float number should be between 0. and 1.')

    args = parser.parse_args()
    images_dir = args.images_dir
    target_ext = args.target_ext
    ratio_to_compress = args.ratio_to_compress

    for subdir, dirs, files in os.walk(images_dir):
        for file in files:
            file_path = os.path.join(subdir, file)

            target_size = None
            if ratio_to_compress is not None:
                target_size = int(os.stat(file_path).st_size * ratio_to_compress)

            convert_img(file_path, is_delete_src_img=False, to_ext=target_ext, target_size=target_size)
