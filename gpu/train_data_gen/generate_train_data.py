import numpy as np
import os
import cv2
from oculoenv import PointToTargetContent, Environment
from oculoenv import RandomDotMotionDiscriminationContent
from oculoenv import OddOneOutContent
from oculoenv import VisualSearchContent
from oculoenv import ChangeDetectionContent
from oculoenv import MultipleObjectTrackingContent

# referenced https://github.com/susumuota/gym-oculoenv
'''
git clone https://github.com/susumuota/oculoenv.git # not wbap's
cd oculoenv
pip install -e .
cd ..

pip install gym

git clone https://github.com/openai/baselines.git
cd baselines
pip install -e .
cd ..

git clone https://github.com/susumuota/gym-oculoenv.git
cd gym-oculoenv
pip install -e .
cd ..
'''
D_RANGE=0.3

def save_image(image, file_path):
    module_dir = os.path.dirname(os.path.abspath(__file__))
    absolute_path = os.path.join(module_dir, file_path)

    # Change RGB to BGR
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(absolute_path, image)


def generate_train_images(env, number_of_image=2000, dir='train_image'):
    for n in range(number_of_image):
        dh = np.random.uniform(low=-D_RANGE, high=D_RANGE)
        dv = np.random.uniform(low=-D_RANGE, high=D_RANGE)
        obs, reward, done, _ = env.step([dh,dv])
        save_image(obs['screen'], dir + ('/%d.png' % n))
        env.reset()

contents = [
    # classid contents, skip_red_cursor, retina, number of images
    ('0', PointToTargetContent(), False, True, 1000),
    ('0', OddOneOutContent(), False, True, 1000),
    ('1', PointToTargetContent(), True, True, 2000),
    ('2', RandomDotMotionDiscriminationContent(), True, True, 2000),
    ('3', OddOneOutContent(), True, True, 2000),
    ('4', VisualSearchContent(), True, True, 2000),
    ('5', ChangeDetectionContent(), True, True, 2000),
    ('6', MultipleObjectTrackingContent(), True, True, 2000)
]

for (id, content, skip, retina, number_of_image) in contents:
    env = Environment(content, skip_red_cursor=skip, retina=retina)
    generate_train_images(env, number_of_image=number_of_image, dir='../application/functions/data/classifier/train/%s' % id)
    generate_train_images(env, number_of_image=int(number_of_image*0.2), dir='../application/functions/data/classifier/test/%s' % id)
