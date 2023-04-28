import os
from io import BytesIO
import cairosvg
from PIL import Image
from torchvision import transforms
import numpy as np

def read_image(svg_file, size=(32, 32)):
    with open(svg_file, 'rb') as f:
        svg_data = f.read()
    png_data = cairosvg.svg2png(bytestring=svg_data)
    img = Image.open(BytesIO(png_data)).convert("RGBA")
    if size is not None:
        img = img.resize(size)
    return img

def covert_png_to_L(img):
    alpha_data = img.getdata(3)
    alpha_img = Image.new('L', img.size)
    alpha_img.putdata(alpha_data)
    return alpha_img


def convert_svg_to_png(root):
    dirs = os.listdir(root)
    for d in dirs:
        for file in os.listdir(root + "/" + d):
            if file.endswith('.svg'):
                svg_path = os.path.join(root + "/" + d + "/", file)
                png_path = os.path.join(svg_path.replace('.svg', '.png'))
                img = read_image(svg_path)
                if img.mode == "RGBA":
                    alphaImage = covert_png_to_L(img)
                    alphaImage.save(png_path)
                else:
                    img.save(png_path)
                os.remove(svg_path)
                print(f"transform {svg_path} to {png_path}")
            # if file.endswith('.png'):
            #     png_path = os.path.join(root + "/" + d + "/", file)
            #     img = Image.open(png_path, 'r')
            #     if img.mode == "RGBA":
            #         alphaImage = covert_png_to_L(img)
            #         alphaImage.save(png_path)
            #         print(f"transform {png_path} to L image")
            #     else:
            #         print("pass")

# Example usage
svg_dir = 'Data'
convert_svg_to_png(svg_dir)

