import io
import math
import sys
import os

from PIL import Image


def save_img_with_target_size(img: Image, filename: str, format: str = 'webp', target_size: int = None) -> bool:
    """
    Save the image as .format with the given name at best quality that makes less than "target_size" bytes.

    return True if compressed image saved successfully and False otherwise.
    """
    # Min and Max quality

    Qmin, Qmax = 25, 98
    # Highest acceptable quality found
    Qacc = -1

    while Qmin <= Qmax and target_size is not None:
        m = math.floor((Qmin + Qmax) / 2)

        # Encode into memory and get size
        buffer = io.BytesIO()
        img.save(buffer, format=format, quality=m)
        img_bytes = buffer.getbuffer().nbytes

        if img_bytes <= target_size:
            Qacc = m
            Qmin = m + 1
        elif img_bytes > target_size:
            Qmax = m - 1

    # Write to disk at the defined quality
    if Qacc > -1:
        img.save(filename, format=format, quality=Qacc)
    elif target_size is None:
        img.save(filename, format=format)
    else:
        print(f"ERROR: No acceptble quality factor found for {filename}", file=sys.stderr)
        return False

    return True


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
        out_filename = prefix + '.' + to_ext
        is_compressed_saved = save_img_with_target_size(img, out_filename,
                                                        format=to_ext,
                                                        target_size=target_size)

        if is_delete_src_img and is_compressed_saved:
            os.remove(fname)

    else:
        print(f'Extention of image "{fname}" is not "*.{from_ext}"')


def rename_files_in_folder(folder_path: str, common_name: str):
    """
    Rename files in folder 'folder_path' to {common_name}_0, {common_name}_1, ...
    with saving their extensions.
    """
    sub = 0
    add = 0

    if os.path.exists(os.path.join(folder_path, '.DS_Store')):
        os.remove(os.path.join(folder_path, '.DS_Store'))

    for num, file_name in enumerate(os.listdir(folder_path), 1):
        name, ext = os.path.splitext(file_name)

        if not ext:
            sub += 1
            continue

        src_name = os.path.join(folder_path, file_name)
        dst_name = os.path.join(folder_path, ''.join((common_name, '_', str(num-sub+add), ext)))

        # If file with this 'dst_name' exists then change file name index
        while os.path.exists(dst_name):
            add += 1
            dst_name = os.path.join(folder_path, ''.join((common_name, '_', str(num-sub+add), ext)))

        os.rename(src=src_name, dst=dst_name)
