import numpy as np
import brica
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from skimage.transform import resize
from torchvision import models

class VC(object):
    """ Visual Cortex module.

    You can add feature extraction code as like if needed.
    """

    def __init__(self):
        self.timing = brica.Timing(2, 1, 0)
        self.vgg = torch.load("./data/vgg_13.pth")
        self.vgg.classifier = self.vgg.classifier[0]


    def __call__(self, inputs):
        if 'from_retina' not in inputs:
            raise Exception('VC did not recieve from Retina')

        retina_image = inputs['from_retina']
        obs = resize(np.array(retina_image), (224, 224))
        obs = np.expand_dims(obs, axis=0)
        obs = torch.from_numpy(obs.transpose((0, 3, 1, 2))).float() / 255.
        feature = self.vgg(obs)

        # Current implementation just passes through input retina image to FEF and PFC.

        return dict(to_fef=feature.cpu(),
                    to_pfc=feature.cpu())
