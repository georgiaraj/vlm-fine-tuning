import pdb
import csv
import argparse
from PIL import Image
from PIL import ImageDraw
import math
import json
from pathlib import Path


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default='scalpel-angles')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    root_dir = Path(args.dataset_path)
    image_dir = root_dir / 'images'
    json_dir = root_dir / 'jsons'

    images = list(image_dir.rglob('*.png'))
    jsons = [f.name for f in json_dir.rglob('*.json')]

    draw_images = False

    all_orders = []
    data_set = 'train'

    with open(str(root_dir / 'scalpel_dataset.csv'), 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['image', 'caption', 'data_set', 'instruments', 'bboxes', 'angle'])

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

            instruments = [i['name'].replace('"', '') for i in order]
            bboxes = [[i['bbox'] for i in order]]
            if len(set(all_orders)) == 28 and instruments not in all_orders:
                data_set = 'test'

            all_orders.append(','.join(instruments))

            writer.writerow([im.name, f'This image shows the following instruments: {", ".join(instruments)} at a {angle} degree angle.', data_set, instruments, bboxes, angle])

            if draw_images:
                img.save(im.name.replace('.png', '_annotated.png'))

    print(f'CSV file created at {root_dir / "scalpel_dataset.csv"}')    

