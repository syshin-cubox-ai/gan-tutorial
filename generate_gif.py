import glob
import os

from PIL import Image

img_paths = glob.glob('images/*.jpg')
img_paths.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
frames = [Image.open(img_path) for img_path in img_paths]
frames[0].save('gen.gif', 'GIF', save_all=True, append_images=frames, duration=100, loop=0)
