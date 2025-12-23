from collections import defaultdict
import numpy as np
import os
import cv2
import json

datadir = "/Users/shreyas/Developer/Research/GroundingDINO/data/satellite"
maskdir = "/Users/shreyas/Developer/Research/GroundingDINO/data/satellite/masks-yolo"
true_maskdir = "/Users/shreyas/Developer/Research/GroundingDINO/data/satellite/masks"
destdir = "/Users/shreyas/Developer/Research/GroundingDINO/data/satellite/boxes"

def readimg(path):
    return cv2.imread(path)

def show(*args):
    img = np.concatenate([
        np.pad(_, ((20, 20), (20, 20), (0, 0))) for _ in args
    ], axis=1)

    cv2.imshow("img", img)
    return cv2.waitKey(-1)

def boxes_watershed(iinfo):
    return [(x, y, x+w, y+h) for x, y, w, h in boxes[iinfo["id"]]]

def boxes_yolo(iinfo):
    apt = os.path.join(maskdir, iinfo["file_name"].replace('tif', 'json'))
    apt = json.load(open(apt))

    boxes = []
    for pred in apt["predictions"]:
        if not pred["confidence"] > 0:
            continue

        x = pred["x"]
        y = pred["y"]
        w = pred["width"]
        h = pred["height"]

        box = [x - w / 2, y - h / 2, x + w / 2, y + h / 2]
        boxes.append(list(map(int, box)))
    return boxes

def drawboxes(simg, boxes):
    rimg = simg.copy()

    for box in boxes:
        cv2.rectangle(rimg, box[:2], box[2:], (0, 255, 255), 1)
    return rimg


coco = json.load(open("/Users/shreyas/Developer/DINOv2/notebooks/annotations.json"))
imgs = {m["id"]: m for m in coco["images"]}

boxes = defaultdict(list)
for ann in coco["annotations"]:
    boxes[ann["image_id"]].append(ann["bbox"])

Q = 113
W = 119
E = 101
ESC = 27

images = iter(imgs.values())
iinfo = next(images)

_i = 0

while images:
    print(_i, iinfo["file_name"])
    _i += 1

    dest = os.path.join(
        destdir, iinfo["file_name"].replace("tif", "json"))

    if os.path.exists(dest):
        print("Skipping", iinfo["file_name"])
        iinfo = next(images)

        continue

    simg = readimg(os.path.join(datadir, "images", iinfo["file_name"]))

    box1 = boxes_watershed(iinfo)
    box2 = boxes_yolo(iinfo)

    mimg = readimg(os.path.join(true_maskdir, iinfo["file_name"].replace("tif", "png")))
    mimg *= 255

    cv2.applyColorMap(mimg, cv2.COLORMAP_VIRIDIS, mimg)

    key = show(
        drawboxes(simg, box1),
        drawboxes(simg, box2),
        mimg )

    if key == Q:
        json.dump({
            "type": "watershed",
            "box": box1
        }, open(dest, "w"))

    elif key == W:
        json.dump({
            "type": "yolo",
            "box": box2
        }, open(dest, "w"))

    elif key == E:
        pass

    elif key == ESC:
        break

    else:
        continue

    iinfo = next(images)
