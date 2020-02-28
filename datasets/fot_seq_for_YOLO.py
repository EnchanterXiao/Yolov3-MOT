import numpy as np
import os
import torch
import torch.utils.data as data
import cv2 as cv
import xml.etree.cElementTree as ET
from skimage.transform import resize


class FOTSeq_yolov3_train(data.Dataset):
    def __init__(self, root, seq_name, min_height, img_size=608):
        self.root = root
        self.seq_name = seq_name
        self.min_height = min_height

        self.img_shape = (img_size, img_size)
        self.max_objects = 50

        self.im_root = os.path.join(self.root, self.seq_name)
        self.im_names = sorted([name for name in os.listdir(self.im_root) if os.path.splitext(name)[-1] == '.jpg'])
        self.det_names = sorted([name for name in os.listdir(self.im_root) if os.path.splitext(name)[-1] == '.xml'])

    def __len__(self):
        return len(self.im_names)

    def __getitem__(self, i):
        # ---------
        #  Image
        # ---------
        im_name = os.path.join(self.im_root, self.im_names[i])
        im = cv.imread(im_name)
        im = im[..., ::-1] #BGR2RGB

        h, w, _ = im.shape
        dim_diff = np.abs(h-w)
        # Upper (left) and lower (right) padding
        pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
        # Determine padding
        pad = ((pad1, pad2), (0, 0), (0, 0)) if h <= w else ((0, 0), (pad1, pad2), (0, 0))
        # Add padding
        input_img = np.pad(im, pad, 'constant', constant_values=128) / 255.
        padded_h, padded_w, _ = input_img.shape
        # Resize and normalize
        input_img = resize(input_img, (*self.img_shape, 3), mode='reflect')
        # Channels-first
        input_img = np.transpose(input_img, (2, 0, 1))
        # As pytorch tensor
        input_img = torch.from_numpy(input_img).float()

        # ---------
        #  Label
        # ---------

        anno = ET.parse(os.path.join(self.im_root, self.det_names[i]))
        bboxs = []
        for obj in anno.findall('object'):
            bndbox_anno = obj.find('bndbox')
            bbox = [int(bndbox_anno.find(tag).text)
                    for tag in ('xmin', 'ymin', 'xmax', 'ymax')]
            bboxs.append(bbox)
        bboxs = np.stack(bboxs).astype(np.float32)  # from list to array
        tlwhs = bboxs.copy()
        tlwhs[:, 2:] = tlwhs[:, 2:]-tlwhs[:, :2]

        # Extract coordinates for unpadded + unscaled image
        x1 = bboxs[:, 0]*1
        y1 = bboxs[:, 1]*1
        x2 = bboxs[:, 2]*1
        y2 = bboxs[:, 3]*1
        bboxs[:, 2] = (bboxs[:, 2] - bboxs[:, 0])*1.0/w
        bboxs[:, 3] = (bboxs[:, 3] - bboxs[:, 1])*1.0/h
        # print(bboxs[:, 3])
        # Adjust for added padding
        x1 += pad[1][0]
        y1 += pad[0][0]
        x2 += pad[1][0]
        y2 += pad[0][0]
        # Calculate ratios from coordinates
        bboxs[:, 0] = ((x1 + x2) / 2) / padded_w
        bboxs[:, 1] = ((y1 + y2) / 2) / padded_h
        bboxs[:, 2] *= w / padded_w
        bboxs[:, 3] *= h / padded_h
        fill_c = np.zeros([bboxs.shape[0], 1])
        bboxs = np.c_[fill_c, bboxs]

        # Fill matrix
        filled_labels = np.zeros((self.max_objects, 5))
        if bboxs is not None:
            filled_labels[range(len(bboxs))[:self.max_objects]] = bboxs[:self.max_objects]
        filled_labels = torch.from_numpy(filled_labels)

        return input_img, filled_labels


def get_loader(root, name, min_height=0, num_workers=3, batch_size=1):
    dataset = FOTSeq_yolov3_train(root, name, min_height, 608)
    data_loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return data_loader


