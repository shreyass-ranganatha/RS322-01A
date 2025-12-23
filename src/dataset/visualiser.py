import matplotlib.pyplot as pt
import glob
import os

imgdir = "/Users/shreyas/Developer/Research/GroundingDINO/data/satellite/images"
mskdir = "/Users/shreyas/Developer/Research/GroundingDINO/data/satellite/masks"

def getfs(path):
    for f in glob.glob(os.path.join(path, "*.tif")):
        yield os.path.basename(f).replace(".tif", "")

savedir = "/Users/shreyas/Developer/Research/GroundingDINO/data/satellite/masks-vis"

for f in getfs(imgdir):
    m = os.path.join(mskdir, f"{f}.png")
    m = pt.imread(m)

    pt.imsave(os.path.join(savedir, f"{f}.png"), m)
