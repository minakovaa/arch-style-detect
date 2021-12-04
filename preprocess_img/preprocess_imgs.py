import os
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from convert_img import convert_img, rename_files_in_folder
from find_similar_imgs import find_similar_images


def process_convert_img(images_dir, target_ext, is_rename,
                                     is_delete_original, ratio_to_compress, max_img_size,
                                     is_delete_duplicates, is_show_duplicates):
    find_similar_images(images_dir, delete_duplicates=is_delete_duplicates, is_show_duplicates=is_show_duplicates)

    folder_name_with_imgs = os.path.basename(os.path.normpath(images_dir))
    if is_rename:
        rename_files_in_folder(folder_path=images_dir,
                               common_name=folder_name_with_imgs)

    for dirpath, dirnames, files in os.walk(images_dir):

        for dirname in dirnames:
            process_convert_img(os.path.join(images_dir, dirname), target_ext, is_rename,
                                is_delete_original, ratio_to_compress, max_img_size,
                                is_delete_duplicates, is_show_duplicates)

        for file in files:
            file_path = os.path.join(dirpath, file)

            target_size = None
            if ratio_to_compress is not None:
                target_size = int(os.stat(file_path).st_size * ratio_to_compress)
            elif max_img_size is not None and max_img_size < os.stat(file_path).st_size:
                target_size = max_img_size

            convert_img(file_path, is_delete_src_img=is_delete_original,
                        to_ext=target_ext, target_size=target_size)


def callback_convert_img(arguments):
    images_dir = arguments.images_dir
    target_ext = arguments.target_ext
    is_rename = arguments.rename
    is_delete_original = arguments.delete_original
    ratio_to_compress = arguments.ratio_to_compress
    max_img_size = arguments.max_img_size

    is_delete_duplicates = arguments.delete_duplicates
    is_show_duplicates = arguments.show_duplicates

    process_convert_img(images_dir, target_ext, is_rename,
                        is_delete_original, ratio_to_compress, max_img_size,
                        is_delete_duplicates, is_show_duplicates)


def setup_parser(parser):
    parser.set_defaults(callback=callback_convert_img)

    parser.add_argument('--images-dir', dest='images_dir', metavar='DIR', required=True,
                        help='Path to folder with images (may contain subfolders)')

    parser.add_argument('--target-ext', dest='target_ext', metavar='EXT', type=str, required=True,
                                    help='Target extension to convert.')

    sp = parser.add_mutually_exclusive_group()
    sp.add_argument('--ratio-to-compress', dest='ratio_to_compress', metavar='RATIO', type=float, default=None, required=False,
                    help='Ratio to compress size of image. Float number should be between 0. and 1.')
    sp.add_argument('--max-img-size', dest='max_img_size', metavar='NUM_BYTES', type=int, default=None, required=False,
                    help='Maximum size of each image in bytes.')

    parser.add_argument('--no-delete-original', dest='delete_original', action='store_false',
                        help='Not delete original images after converting and compressing.')
    parser.set_defaults(delete_original=True)

    parser.add_argument('--no-rename', dest='rename', action='store_false',
                        help='Rename images according folder name and numerate them')
    parser.set_defaults(rename=True)

    parser.add_argument('--no-delete-duplicates', dest='delete_duplicates', action='store_false',
                        help='Optional delete duplicates except one.'
                             'Find for image duplicates and optionally delete all but one.')
    parser.set_defaults(delete_duplicates=True)

    parser.add_argument('--no-show-duplicates', dest='show_duplicates', action='store_false',
                        help='Optional show duplicates on the screen. By default show all duplicates.')
    parser.set_defaults(show_duplicates=True)


def main():
    parser = ArgumentParser(
        description='Convert images with size compressing and deleting duplicates in selected folder. '
                    'If folder have subfolders, then convert images in subfolders.',
        formatter_class=ArgumentDefaultsHelpFormatter
    )

    setup_parser(parser)
    arguments = parser.parse_args()

    arguments.callback(arguments)


if __name__ == '__main__':
    main()
