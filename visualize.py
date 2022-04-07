import os
import numpy as np
import json
from PIL import Image, ImageDraw

data_path = 'data/RedLights2011_Medium'
preds_path = 'data/hw01_preds/preds.json'

with open(preds_path) as f:
    bbs = json.load(f)

for path in bbs:
    with Image.open(os.path.join(data_path, path)) as im:
        draw = ImageDraw.Draw(im)
        for box in bbs[path]:
            w = 1
            # draw.line([(box[0], box[1]), (box[0], box[3])], width=5)
            draw.line([
                (box[1], box[0]),
                (box[3], box[0])
            ], width=w)

            draw.line([
                (box[3], box[0]),
                (box[3], box[2])
            ], width=w)

            draw.line([
                (box[3], box[2]),
                (box[1], box[2])
            ], width=w)

            draw.line([
                (box[1], box[2]),
                (box[1], box[0])
            ], width=w)

        im.save(f'data/hw01_viz/{path}')