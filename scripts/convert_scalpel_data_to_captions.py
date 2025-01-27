import pdb
from PIL import Image
from PIL import ImageDraw
import math
import json
from pathlib import Path

root_dir = Path('../scalpel-angles')
image_dir = root_dir / 'images'
json_dir = root_dir / 'jsons'

images = list(image_dir.rglob('*.png'))
jsons = [f.name for f in json_dir.rglob('*.json')]

draw_images = False

with open(str(root_dir / 'scalpel_dataset.csv'), 'w') as f:
    f.write('image, caption, instruments, bboxes, angle\n')

    for im in images:
        json_file = im.parent.parent / 'jsons' / (im.name.replace('.png', '.json'))

        if json_file.name not in jsons:
            continue

        with open(str(json_file), 'r') as j_file:
            data = json.load(j_file)

        angle = data['image']['platform_rotation']

        if draw_images:
            img = Image.open(str(im))
            draw = ImageDraw.Draw(img)

        def find_order(instrument, angle):
            if angle < 90 or angle > 270:
                return instrument['bbox'][0]
            elif angle == 90:
                return instrument['bbox'][1]
            elif angle == 270:
                return -instrument['bbox'][1]
            else:
                return -instrument['bbox'][0]

        # Work out the order of the instruments
        order = sorted(data['instruments'], key=lambda x: find_order(x, angle))

        f.write(f'{im.name},"This image shows the following instruments: {", ".join([i["name"] for i in order])} at a {angle} degree angle.", ')

        for i in data['instruments']:
            f.write(f'[{",".join([i["name"] for i in data["instruments"]])}], [{",".join([",".join([str(b) for b in i["bbox"]]) for i in data["instruments"]])}],' 
                    f'{angle},')

            if draw_images:
                draw.rectangle([i["bbox"][0], i["bbox"][1], i["bbox"][0] + i["bbox"][2], i["bbox"][1] + i["bbox"][3]], outline="red")
        if draw_images:
            img.save(im.name.replace('.png', '_annotated.png'))

        f.write('\n')
        # Check for straight bboxes    
        #if data['angle'] == 0:
        #    f.write(f'{im.name}, {i["name"]}, {",".join([str(b) for b in i["bbox"]])}\n')
