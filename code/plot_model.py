# coding: utf-8
import warnings

import tensorflow as tf
import os.path
from utils.separable_conv2d import SeparableConv2DKeras, BilinearUpSampling2D
from utils import model_tools

from tensorflow.contrib.keras.python.keras.utils.vis_utils import plot_model

weight_file_name = 'model_weights'
output_file_name = os.path.join('../docs/images/', 'model')
extensions = ['.png', '.svg']

# Load model
model = model_tools.load_network(weight_file_name)

# Output model
for ext in extensions:
    # Horizontal iamge
    try:
        filename = output_file_name + '-horizontal' + ext
        plot_model(model, to_file=filename, 
            show_shapes=True,  show_layer_names=True, 
            rankdir='LR')
    except TypeError as e:
        warnings.warn('A horizontal plotting supports tensorflow v1.3 or lator')

    # Vertical iamge
    filename = output_file_name + '-vertical' + ext
    plot_model(model, to_file=filename, 
        show_shapes=True,  show_layer_names=True)

