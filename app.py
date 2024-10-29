### 1. Imports and class names setup ###
from model import create_vitB16_model

import torch
from typing import Tuple, Dict
from timeit import default_timer as timer
import gradio as gr
import os


class_names = []

# Open Cifar class names file and read each line into a list
with open('class_names.txt', 'r') as f:
    class_names = [obj.strip() for obj in f]

### 2. Model and transforms preparation ###
model, model_transforms = create_vitB16_model(num_classes=len(class_names))

# Load save weights
model.load_state_dict(torch.load(f='09_pretrained_vit_feature_extractor_cifar_60_percent.pth',
                                 map_location='cpu'))

# 3. Predict Function

def predict(img) -> Tuple[Dict, float]:
  # Start a timer
  start_time = timer()

  # Transform the input image for use with vitB16
  img = model_transforms(img).unsqueeze(dim=0)

  # Put model into eval mode, make prediction
  model.eval()
  with torch.inference_mode():
    # Pass transformed image through the model and turn the prediction logits into probabilities
    pred_logit = model(img)
    pred_prob = torch.softmax(pred_logit, dim=1)

  # Create a prediction label and prediction probability dictionary
  pred_labels_and_probs = {class_names[i]: float(pred_prob[0][i]) for i in range(len(class_names))}

  # Calculate pred time
  end_time = timer()
  pred_time = round(end_time - start_time, 4)

  # Return pred dict and pred time
  return pred_labels_and_probs, pred_time

### 4. Gradio app ###


# Create title, description and article
title = "Object Detection 🤖"
description = "A [vision Transformer B16 feature extractor](https://pytorch.org/vision/stable/models/generated/torchvision.models.vit_b_16.html) computer vision model to classify 10 classes of Objects from the Cifar10 dataset. [Cifar10](https://pytorch.org/vision/stable/generated/torchvision.datasets.CIFAR10.html?highlight=cifar)"
article = "Created with 🤎 (and a mixture of mathematics, statistics, and tons of calculations 👩🏽‍🔬) by Arpit Vaghela [GitHub](https://github.com/magnifiques)"

# Create example list
example_list = [["examples/" + example] for example in os.listdir("examples")]

demo = gr.Interface(fn=predict,
                    inputs=gr.Image(type='pil'),
                    outputs=[gr.Label(num_top_classes=3, label='Predictions'),
                             gr.Number(label="Prediction time (s)")],
                    examples=example_list,
                    title=title,
                    description=description,
                    article=article)

# Launch the demo!
demo.launch(debug=False, # print errors locally?
            share=True) # generate a publically shareable URL
