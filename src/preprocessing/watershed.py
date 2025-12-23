# %%
import matplotlib.pyplot as pt
import matplotlib.image
import numpy as np
import cv2

# %%
def show(*args, labels=[], show_axis=True):
    l = len(args)

    if len(labels) < l:
        labels += [None] * (l - len(labels))

    fig, plots = pt.subplots(int(l ** .5), l // int(l ** .5))

    try:
        plots = plots.flatten()
    except:
        plots = [plots]

    for m, p, l in zip(args, plots, labels):
        ax: matplotlib.image.AxesImage = p.imshow(m)

        if not show_axis:
            ax.axes.axis('off')

        if l:
            ax.axes.title.set_text(l)
    pt.show()


# %% [markdown]
# ## Read Mask

# %%
# READ MASK

KERN = np.ones((3, 3), dtype=np.uint8)

image_id = "high_density_mango_77"
# image_id = "low_density_coconut_3"

s = cv2.imread(
    f"/Users/shreyas/Developer/Research/GroundingDINO/data/satellite/images/{image_id}.tif",
    cv2.IMREAD_GRAYSCALE)

cv2.normalize(s, s, 0, 255, cv2.NORM_MINMAX)

k = cv2.imread(
    f"/Users/shreyas/Developer/Research/GroundingDINO/data/satellite/masks/{image_id}.png",
    cv2.IMREAD_GRAYSCALE)

_, s_t = cv2.threshold(s, 0, 512, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# mask = cv2.bitwise_and(s_t, k)
# mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, KERN, iterations=2)
mask = k

show(s, mask)

# %%
from cv2 import SimpleBlobDetector

detector = SimpleBlobDetector()
keypoints = detector.detect(mask)

keypoints
