import cv2
import numpy as np

import brica
from .utils import load_image

import torch
import torch.nn as nn
import torch.nn.functional as F

"""
This is a sample implemention of PFC (Prefrontal cortex) module.
You can change this as you like.
"""

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(4096, 1024)
        self.dropout1 = nn.Dropout()
        self.fc2 = nn.Linear(1024, 256)
        self.dropout2 = nn.Dropout()
        self.fc3 = nn.Linear(256, 64)
        self.fc4 = nn.Linear(64, 7)
    def forward(self, x):
        x = F.relu(self.dropout1(self.fc1(x)))
        x = F.relu(self.dropout2(self.fc2(x)))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

class Phase(object):
    INIT = -1  # Initial phase
    START = 0  # Start phase while finding red cross cursor
    TARGET = 1 # Target finding phsae

class CursorFindAccumulator(object):
    def __init__(self, decay_rate=0.9):
        # Accumulated likelilood
        self.decay_rate = decay_rate
        self.likelihood = 0.0
        self.cursor_template = load_image("data/debug_cursor_template_w.png")

    def accumulate(self, value):
        self.likelihood += value
        self.likelihood = np.clip(self.likelihood, 0.0, 1.0)

    def reset(self):
        self.likelihood = 0.0

    def process(self, retina_image):
        match = cv2.matchTemplate(retina_image, self.cursor_template,
                                  cv2.TM_CCOEFF_NORMED)
        match_rate = np.max(match)
        self.accumulate(match_rate * 0.1)

    def post_process(self):
        # Decay likelihood
        self.likelihood *= self.decay_rate



class PFC(object):
    def __init__(self, use_cuda=False):
        self.timing = brica.Timing(3, 1, 0)
        self.cursor_find_accmulator = CursorFindAccumulator()
        self.phase = Phase.INIT
        self.cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.cuda else "cpu")
        self.classifier = Classifier()
        param = torch.load('./data/classifier.pth', map_location=lambda storage, loc: storage)
        self.classifier.load_state_dict(param)
        self.classifier = self.classifier.to(self.device)

    def __call__(self, inputs):
        if 'from_vc' not in inputs:
            raise Exception('PFC did not recieve from VC')
        if 'from_fef' not in inputs:
            raise Exception('PFC did not recieve from FEF')
        if 'from_bg' not in inputs:
            raise Exception('PFC did not recieve from BG')

        # Image from Visual cortex module.
        features = torch.from_numpy(inputs['from_vc']).to(self.device)
        # Allocentrix map image from hippocampal formatin module.

        output = self.classifier(features)
        output = output.cpu().detach().numpy()

        # This is a very sample implementation of phase detection.
        # You should change here as you like.
        # self.cursor_find_accmulator.process(features)
        # self.cursor_find_accmulator.post_process()

        # if self.phase == Phase.INIT:
        #     if self.cursor_find_accmulator.likelihood > 0.7:
        #         self.phase = Phase.START
        # elif self.phase == Phase.START:
        #     if self.cursor_find_accmulator.likelihood < 0.4:
        #         self.phase = Phase.TARGET
        # else:
        #     if self.cursor_find_accmulator.likelihood > 0.6:
        #         self.phase = Phase.START

        self.phase = Phase.START

        if self.phase == Phase.INIT or self.phase == Phase.START:
            # TODO: 領野をまたいだ共通phaseをどう定義するか？
            fef_message = 0
        else:
            fef_message = 1

        return dict(to_fef=fef_message,
                    to_bg=output)
