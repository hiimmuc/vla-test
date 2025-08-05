import os

import jax
import matplotlib.pyplot as plt
import numpy as np
from octo.model.octo_model import OctoModel
from PIL import Image

os.environ["TOKENIZERS_PARALLELISM"] = "false"


model = OctoModel.load_pretrained("hf://rail-berkeley/octo-small-1.5")

# download one example BridgeV2 image
IMAGE_URL = "sample.png"
img = np.array(Image.open(IMAGE_URL).resize((256, 256)))
plt.imshow(img)

# create obs & task dict, run inference

# add batch + time horizon 1
img = img[np.newaxis, np.newaxis, ...]
observation = {"image_primary": img, "timestep_pad_mask": np.array([[True]])}
task = model.create_tasks(texts=["cook the egg"])
action = model.sample_actions(
    observation,
    task,
    unnormalization_statistics=model.dataset_statistics["bridge_dataset"]["action"],
    rng=jax.random.PRNGKey(0),
)
print(action)
