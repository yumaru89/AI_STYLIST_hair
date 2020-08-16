import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from PIL import Image

from utils import has_img_ext


def crop_center(pil_img, crop_width, crop_height):
    img_width, img_height = pil_img.size
    return pil_img.crop(((img_width - crop_width) // 2,
                         (img_height - crop_height) // 2,
                         (img_width + crop_width) // 2,
                         (img_height + crop_height) // 2))


def main():
    img_dir = "img/model/face"
    img_paths = [os.path.join(img_dir, k) for k in sorted(os.listdir(img_dir)) if has_img_ext(k)]
    save_dir = "img/model/resize_face"
    os.makedirs(save_dir, exist_ok=True)
    img_height = 600

    for img_path in img_paths:
        img = Image.open(img_path)
        w, h = img.size
        ratio = img_height / h
        re_img = img.resize((int(w*ratio), img_height))
        trim_img = crop_center(re_img, 480, img_height)
        path = os.path.join(save_dir, os.path.splitext(os.path.basename(img_path))[0] + '.png')
        trim_img.save(path)

if __name__ == '__main__':
    main()
