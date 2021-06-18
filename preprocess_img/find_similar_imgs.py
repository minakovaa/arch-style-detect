import os
from imagehash import phash
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

# Avalible Methods in imagehash:
#
# ahash: Average hash
# phash: Perceptual hash
# dhash: Difference hash
# whash-haar: Haarwavelet hash
# whash-db4: Daubechies wavelet hash
# colorhash: HSV color hash
# crop-resistant: Crop - resistant hash


def find_similar_images(folder_path: str,
                        delete_duplicates=False):
    if not os.path.isdir(folder_path):
        return None

    hashfunc = phash

    def is_image(filename):
        f = filename.lower()
        return f.endswith(".png") or f.endswith(".jpg") or \
               f.endswith(".jpeg") or f.endswith(".bmp") or \
               f.endswith(".gif") or '.jpg' in f or f.endswith(".svg")

    image_filenames = [os.path.join(folder_path, path) for path in os.listdir(folder_path) if is_image(path)]

    images_by_hash = {}

    for img in sorted(image_filenames):
        try:
            hash = hashfunc(Image.open(img))
        except Exception as e:
            print('Problem:', e, 'with', img)
            continue
        if hash in images_by_hash:
            # print(img, '  already exists as', ' '.join(images_by_hash[hash]))
            # 'img' already exists as images_by_hash[hash]
            if delete_duplicates:
                os.remove(img)

        images_by_hash[hash] = images_by_hash.get(hash, []) + [img]

    # in 'duplicates' key: original image, value: list of duplicates without original one
    duplicates = {imgs[0]: imgs[1:] for hash, imgs in images_by_hash.items() if len(imgs) > 1}

    return duplicates


def delete_duplicates_in_subfolders(main_folder, delete_duplicates=False):
    folders = os.listdir(main_folder)
    duplicates = {}

    for folder_path in folders:
        folder_dupl = find_similar_images(folder_path=os.path.join(main_folder, folder_path),
                                          delete_duplicates=delete_duplicates)

        if folder_dupl:
            duplicates.update(folder_dupl)

    return duplicates


if __name__ == '__main__':
    main_folder = os.path.join(os.getcwd(), '../dataset')

    duplicates = delete_duplicates_in_subfolders(main_folder, delete_duplicates=True)

    for orig_img, dupl_imgs in duplicates.items():
        print(orig_img, dupl_imgs)
