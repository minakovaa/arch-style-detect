import os
from typing import List
from .convert_img import rename_files_in_folder, convert_img


def convert_all_images_to_jpeg(dataset_path: str, convert=True, rename=True, only_for_classes: List[str] = None):
    """
    Rename images in dataset according class name.

    folder structure in 'dataset_path' must be:
        > folders with classes names
            > inside each folder images with this class

    If 'only_for_classes' defined, rename and convert images
    only for specified classes
    """
    for arhc_style_name in os.listdir(dataset_path):

        arch_style_folder = os.path.join(dataset_path, arhc_style_name)
        if not os.path.isdir(arch_style_folder):
            continue

        if only_for_classes is None or arhc_style_name in only_for_classes:

            if rename:
                rename_files_in_folder(folder_path=arch_style_folder,
                                       common_name=arhc_style_name)

            if convert:
                for image in os.listdir(arch_style_folder):
                    convert_img(fname=os.path.join(arch_style_folder, image),
                                to_ext='jpeg',
                                is_delete_src_img=True)


if __name__ == '__main__':
    dataset_path = '../dataset'

    convert_all_images_to_jpeg(dataset_path, convert=True, rename=True,
                               only_for_classes=['барокко']
                               )
