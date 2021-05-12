import os
from img_preprocessing import rename_files_in_folder, convert_img


def convert_all_images_to_jpeg(dataset_path):
    for arhc_style_name in os.listdir(dataset_path):

        arch_style_folder = os.path.join(dataset_path, arhc_style_name)
        if not os.path.isdir(arch_style_folder):
            continue

        rename_files_in_folder(folder_path=arch_style_folder,
                               common_name=arhc_style_name)

        for image in os.listdir(arch_style_folder):
            convert_img(fname=os.path.join(arch_style_folder, image),
                        to_ext='jpeg',
                        is_delete_src_img=True)


if __name__ == '__main__':
    dataset_path = 'Датасет/'

    convert_all_images_to_jpeg(dataset_path)
