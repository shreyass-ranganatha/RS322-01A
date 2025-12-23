import numpy as np
import os
import json
import cv2
import glob

def readimg(path):
    return cv2.imread(path)

def show(*args):
    img = np.concatenate([
        np.pad(_, ((20, 20), (20, 20), (0, 0))) for _ in args
    ], axis=1)

    cv2.imshow("img", img)
    cv2.waitKey(-1)

def drawboxes(simg, boxes):
    rimg = simg.copy()
    boxes = boxes["box"]

    for box in boxes:
        cv2.rectangle(rimg, box[:2], box[2:], (0, 255, 255), 1)
    return rimg


imgsdir = "/Users/shreyas/Developer/Research/GroundingDINO/data/satellite/images"
boxsdir = "/Users/shreyas/Developer/Research/GroundingDINO/data/satellite/boxes"
maskdir = "/Users/shreyas/Developer/Research/GroundingDINO/data/satellite/masks"
putsdir = "/Users/shreyas/Developer/Research/GroundingDINO/data/satellite/boxes-vis-2"

for f in glob.glob(os.path.join(boxsdir, "*.json")):
    m = os.path.basename(f).replace(".json", "")

    # if m != "high_density_mango_20":
    #     continue

    boxpath = os.path.join(boxsdir, f"{m}.json")

    if not os.path.exists(boxpath):
        raise FileNotFoundError(f"{m}.json")

    simg = readimg(os.path.join(imgsdir, f"{m}.tif"))
    bimg = drawboxes(simg, json.load(open(boxpath)))

    # cv2.imwrite(os.path.join(putsdir, f"{m}.png"), bimg)
    cv2.imwrite(os.path.join(putsdir, f"{m}.png"), bimg)
