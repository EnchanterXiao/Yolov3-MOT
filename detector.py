from model.darknet import Darknet
import numpy as np
import torch
import cv2 as cv
from torch.autograd import Variable
from utils.util import non_max_suppression
from utils.util import *
import pickle as pkl


def pre_image(img, inp_dim):
    """
    Prepare image for inputting to the neural network.
    Returns a Variable
    """

    img = cv.resize(img, (inp_dim, inp_dim))
    img = img[:, :, ::-1].transpose((2, 0, 1)).copy()
    img = torch.from_numpy(img).float().div(255.0).unsqueeze(0)
    return img


class Detector(object):
    def __init__(self, img_size=608):
        self.cuda = torch.cuda.is_available()
        self.img_size = img_size
        self.model = Darknet('config/yolov3.cfg', img_size=self.img_size)
        self.model.load_weights('weights/yolov3.weights')
        if self.cuda:
            self.model.cuda()
        self.model.eval()

    def detect(self, img):
        '''
        :param img:
        :return: x1,y1,x2,y2
        '''
        Tensor = torch.cuda.FloatTensor if self.cuda else torch.FloatTensor
        input_img = pre_image(img, self.img_size)
        input_img = Variable(input_img.type(Tensor))
        # Get detections
        with torch.no_grad():
            detection = self.model(input_img)
            detection = non_max_suppression(detection, num_classes=80, confidence=0.5, nms_conf=0.4)
            if detection.shape[0]>0:
                detection[:, 1] = detection[:, 1] * img.shape[1] / self.img_size
                detection[:, 2] = detection[:, 2] * img.shape[0] / self.img_size
                detection[:, 3] = detection[:, 3] * img.shape[1] / self.img_size
                detection[:, 4] = detection[:, 4] * img.shape[0] / self.img_size
                return detection[:, 1:5]
            else:
                return None





