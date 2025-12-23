# %%
from transformers.models.grounding_dino.modeling_grounding_dino import GroundingDinoObjectDetectionOutput
from transformers import GroundingDinoForObjectDetection, GroundingDinoProcessor
import torch
import numpy as np

from PIL import Image, ImageDraw
import matplotlib.pyplot as pt

# %%
checkpoint = "IDEA-Research/grounding-dino-base"

m: GroundingDinoForObjectDetection = GroundingDinoForObjectDetection.from_pretrained(checkpoint)
p: GroundingDinoProcessor = GroundingDinoProcessor.from_pretrained(checkpoint)

# %%
simg = Image.open("000000039769.jpg")
simg

# %%
labels = [
    {
        "class_labels": torch.LongTensor([1, 1]),
        "boxes": torch.FloatTensor([
            [0.11176593, 0.28462592, 0.39940253, 0.8102392 ],
            [0.6327725, 0.19324061, 0.9056199, 0.63215554]
        ]),
    }
]

labels = [
    {
        "class_labels": torch.LongTensor([1, 1]),
        "boxes": torch.FloatTensor([
            [0.0159, 0.1094, 0.4953, 0.9854],
            [0.5418, 0.0469, 0.9966, 0.7785]
        ]),
    }
]

labels

# %%
inps = p(simg, text="elephant . dog . cat .", return_tensors="pt")

with torch.no_grad():
    out: GroundingDinoObjectDetectionOutput = m(**inps, labels=labels)

# print(*out, sep='\n')
print(out.loss_dict)

# %%
pred, = p.post_process_grounded_object_detection(
    outputs=out,
    input_ids=inps.input_ids,
    box_threshold=.3)

# # %%
# rimg = simg.copy()
# draw = ImageDraw.Draw(rimg)

# for box in pred["boxes"]:

#     b2 = box.detach().cpu().numpy().copy()
#     b2[0], b2[2] = b2[0] + (b2[2] - b2[0]) * .2, b2[2] - (b2[2] - b2[0]) * .2
#     b2[1], b2[3] = b2[1] + (b2[3] - b2[1]) * .2, b2[3] - (b2[3] - b2[1]) * .2
#     print(box)
#     b2 = b2 * np.array([simg.width, simg.height, simg.width, simg.height])

#     box = box * torch.tensor([simg.width, simg.height, simg.width, simg.height])

#     draw.rectangle(box.tolist(), outline="yellow", width=2)
#     draw.rectangle(b2.tolist(), outline="red", width=2)

# rimg.show()
