import numpy as np
import os
from matplotlib import pyplot   
import tensorflow as tf
import shutil
import json
import io

def save_settings(args, param):
    if not os.path.exists(args.result_path):
        os.makedirs(args.result_path)
    del args.param
    args_dict = vars(args)
    with open(os.path.join(args.result_path,'args.json'), "w") as f:
        json.dump(args_dict, f, indent=4, sort_keys=False)
    shutil.copy('param/param.py', args.result_path)
    args.param = param

def plot_to_image(figure):
  """Converts the matplotlib plot specified by 'figure' to a PNG image and
  returns it. The supplied figure is closed and inaccessible after this call."""
  # Save the plot to a PNG in memory.
  buf = io.BytesIO()
  pyplot.savefig(buf, format='png')
  # Closing the figure prevents it from being displayed directly inside
  # the notebook.
  pyplot.close(figure)
  buf.seek(0)
  # Convert PNG buffer to TF image
  image = tf.image.decode_png(buf.getvalue(), channels=4)
  # Add the batch dimension
  image = tf.expand_dims(image, 0)
  return image

