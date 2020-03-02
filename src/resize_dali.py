

from __future__ import division
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops
import nvidia.dali.types as types
import numpy as np
from timeit import default_timer as timer
import numpy as np
import matplotlib.pyplot as plt
import math
import os.path

db_folder = "../data/train_resized/train"

class HybridPipe(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, pipelined = True, exec_async = True):
        super(HybridPipe, self).__init__(batch_size, num_threads, device_id, seed = 12, exec_pipelined=pipelined, exec_async=exec_async)
        # self.input = ops.CaffeReader(path = db_folder, random_shuffle = True)
        self.input = ops.FileReader(file_root = db_folder, random_shuffle = True)
        self.decode = ops.ImageDecoder(device = "mixed", output_type = types.RGB)
        self.augmentations = {}
        self.augmentations["resize"] = ops.Resize(device = "gpu", resize_x = 256, resize_y = 256)
        self.iter = 0

    def define_graph(self):
        self.jpegs, self.labels = self.input()
        images = self.decode(self.jpegs)
        n = len(self.augmentations)
        outputs = [images for _ in range(n+1)]
        aug_list = list(self.augmentations.values())
        # outputs[0] is the original cropped image
        for i in range(n):
            outputs[i+1] = aug_list[i](outputs[i+1])
        return [self.labels] + outputs

    def iter_setup(self):
        pass


batch_size = 32

pipe = HybridPipe(batch_size=batch_size, num_threads=2, device_id = 0)
pipe.build()

pipe_out = pipe.run()