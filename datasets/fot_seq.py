import numpy as np
import os
import torch.utils.data as data
import cv2 as cv
import xml.etree.cElementTree as ET


class FOTSeq(data.Dataset):
    def __init__(self, root, seq_name, min_height):
        self.root = root
        self.seq_name = seq_name
        self.min_height = min_height

        self.im_root = os.path.join(self.root, self.seq_name)
        self.im_names = sorted([name for name in os.listdir(self.im_root) if os.path.splitext(name)[-1] == '.jpg'])
        self.det_names = sorted([name for name in os.listdir(self.im_root) if os.path.splitext(name)[-1] == '.xml'])

    def __len__(self):
        return len(self.im_names)

    def __getitem__(self, i):
        im_name = os.path.join(self.im_root, self.im_names[i])
        im = cv.imread(im_name)
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

        return im, bboxs, tlwhs


def collate_fn(data):
    return data[0]


def get_loader(root, name, min_height=0, num_workers=3, batch_size=1):
    dataset = FOTSeq(root, name, min_height)

    data_loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)

    return data_loader

# test = FOTSeq('../data/football', '1', 0)
# for i in range(test.__len__()):

# print(test.__getitem__(0))



