import numpy as np
import os
import torch.utils.data as data
from scipy.misc import imread
from skimage.transform import resize
import torch

from utils.io import read_mot_results, unzip_objs


"""
labels={'ped', ...			% 1
'person_on_vhcl', ...	% 2
'car', ...				% 3
'bicycle', ...			% 4
'mbike', ...			% 5
'non_mot_vhcl', ...		% 6
'static_person', ...	% 7
'distractor', ...		% 8
'occluder', ...			% 9
'occluder_on_grnd', ...		%10
'occluder_full', ...		% 11
'reflection', ...		% 12
'crowd' ...			% 13
};
"""


# def read_mot_results(filename, is_gt=False):
#     labels = {1, -1}
#     targets = dict()
#     if os.path.isfile(filename):
#         with open(filename, 'r') as f:
#             for line in f.readlines():
#                 linelist = line.split(',')
#                 if len(linelist) < 7:
#                     continue
#                 fid = int(linelist[0])
#                 targets.setdefault(fid, list())
#
#                 if is_gt and ('MOT16-' in filename or 'MOT17-' in filename):
#                     label = int(float(linelist[-2])) if len(linelist) > 7 else -1
#                     if label not in labels:
#                         continue
#                 tlwh = tuple(map(float, linelist[2:7]))
#                 target_id = int(linelist[1])
#
#                 targets[fid].append((tlwh, target_id))
#
#     return targets


class MOTSeq(data.Dataset):
    def __init__(self, root, det_root, seq_name, min_height, min_det_score):
        self.root = root
        self.seq_name = seq_name
        self.min_height = min_height
        self.min_det_score = min_det_score

        self.im_root = os.path.join(self.root, self.seq_name, 'img1')
        self.im_names = sorted([name for name in os.listdir(self.im_root) if os.path.splitext(name)[-1] == '.jpg'])

        if det_root is None:
            self.det_file = os.path.join(self.root, self.seq_name, 'det', 'det.txt')
        else:
            self.det_file = os.path.join(det_root, '{}.txt'.format(self.seq_name))
        self.dets = read_mot_results(self.det_file, is_gt=False, is_ignore=False)

        self.gt_file = os.path.join(self.root, self.seq_name, 'gt', 'gt.txt')
        if os.path.isfile(self.gt_file):
            self.gts = read_mot_results(self.gt_file, is_gt=True, is_ignore=False)
        else:
            self.gts = None

    def __len__(self):
        return len(self.im_names)

    def __getitem__(self, i):
        im_name = os.path.join(self.im_root, self.im_names[i])
        # im = cv2.imread(im_name)
        im = imread(im_name)  # rgb
        im = im[:, :, ::-1]  # bgr

        frame = i + 1
        dets = self.dets.get(frame, [])
        tlwhs, _, scores = unzip_objs(dets)
        scores = np.asarray(scores)

        keep = (tlwhs[:, 3] >= self.min_height) & (scores > self.min_det_score)
        tlwhs = tlwhs[keep]
        scores = scores[keep]

        if self.gts is not None:
            gts = self.gts.get(frame, [])
            gt_tlwhs, gt_ids, _ = unzip_objs(gts)
        else:
            gt_tlwhs, gt_ids = None, None

        return im, tlwhs, scores, gt_tlwhs, gt_ids

        # '''
        # comput ap
        # '''
        # im = im[..., ::-1]  # BGR2RGB
        # h, w, _ = im.shape
        # dim_diff = np.abs(h - w)
        # # Upper (left) and lower (right) padding
        # pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
        # # Determine padding
        # pad = ((pad1, pad2), (0, 0), (0, 0)) if h <= w else ((0, 0), (pad1, pad2), (0, 0))
        # # Add padding
        # input_img = np.pad(im, pad, 'constant', constant_values=128) / 255.
        # padded_h, padded_w, _ = input_img.shape
        # # Resize and normalize
        # input_img = resize(input_img, (608, 608, 3), mode='reflect')
        # # Channels-first
        # input_img = np.transpose(input_img, (2, 0, 1))
        # # As pytorch tensor
        # input_img = torch.from_numpy(input_img).float()
        # # ---------
        # #  Label
        # # ---------
        # bboxs = gt_tlwhs
        # bboxs = np.stack(bboxs).astype(np.float32)  # from list to array
        # tlwhs = bboxs.copy()
        # tlwhs[:, 2:] = tlwhs[:, 2:] - tlwhs[:, :2]
        #
        # # Extract coordinates for unpadded + unscaled image
        # # x1 = w * (bboxs[:, 0] - bboxs[:, 2] / 2)
        # # y1 = w * (bboxs[:, 1] - bboxs[:, 3] / 2)
        # # x2 = w * (bboxs[:, 0] + bboxs[:, 2] / 2)
        # # y2 = w * (bboxs[:, 0] + bboxs[:, 3] / 2)
        # x1 = bboxs[:, 0] * 1
        # y1 = bboxs[:, 1] * 1
        # x2 = bboxs[:, 2] * 1
        # y2 = bboxs[:, 3] * 1
        # bboxs[:, 2] = (bboxs[:, 2] - bboxs[:, 0]) * 1.0 / w
        # bboxs[:, 3] = (bboxs[:, 3] - bboxs[:, 1]) * 1.0 / h
        # # print(bboxs[:, 3])
        # # Adjust for added padding
        # x1 += pad[1][0]
        # y1 += pad[0][0]
        # x2 += pad[1][0]
        # y2 += pad[0][0]
        # # Calculate ratios from coordinates
        # bboxs[:, 0] = ((x1 + x2) / 2) / padded_w
        # bboxs[:, 1] = ((y1 + y2) / 2) / padded_h
        # bboxs[:, 2] *= w / padded_w
        # bboxs[:, 3] *= h / padded_h
        # fill_c = np.zeros([bboxs.shape[0], 1])
        # # print(bboxs.shape)
        # # print("fillc:", format(fill_c.shape))
        # bboxs = np.c_[fill_c, bboxs]
        # # Fill matrix
        # filled_labels = np.zeros((50, 5))
        # if bboxs is not None:
        #     filled_labels[range(len(bboxs))[:50]] = bboxs[:50]
        # filled_labels = torch.from_numpy(filled_labels)
        # return input_img, filled_labels


def collate_fn(data):
    return data[0]


def get_loader(root, det_root, name, min_height=0, min_det_score=-np.inf, num_workers=3):
    dataset = MOTSeq(root, det_root, name, min_height, min_det_score)

    # data_loader = data.DataLoader(dataset, 1, False, num_workers=num_workers) # , collate_fn=collate_fn) #track
    data_loader = data.DataLoader(dataset, 1, False, num_workers=num_workers, collate_fn=collate_fn) #compute map

    return data_loader
