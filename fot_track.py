import torch
import os
import logging
from datasets.fot_seq import get_loader
from tracker.mot_tracker import OnlineTracker
from utils.visualization import *
from detector import Detector
from utils.log import logger
from utils.timer import Timer


def mkdirs(path):
    if os.path.exists(path):
        return
    os.makedirs(path)

def write_results(filename, results, data_type):
    if data_type == 'mot':
        save_format = '{frame},{id},{x1},{y1},{w},{h},1,-1,-1,-1\n'
    elif data_type == 'kitti':
        save_format = '{frame} {id} pedestrian 0 0 -10 {x1} {y1} {x2} {y2} -10 -10 -10 -1000 -1000 -1000 -10\n'
    else:
        raise ValueError(data_type)

    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids in results:
            if data_type == 'kitti':
                frame_id -= 1
            for tlwh, track_id in zip(tlwhs, track_ids):
                if track_id < 0:
                    print("yes")
                    continue
                x1, y1, w, h = tlwh
                x2, y2 = x1 + w, y1 + h
                line = save_format.format(frame=frame_id, id=track_id, x1=x1, y1=y1, x2=x2, y2=y2, w=w, h=h)
                f.write(line)
    logger.info('save results to {}'.format(filename))

def eval_seq(dataloader, data_type, result_filename, save_dir=None, show_image=True):
    # print('save_dir:' ,save_dir)
    if save_dir is not None:
        mkdirs(save_dir)

    tracker = OnlineTracker()
    detector = Detector(img_size=608)
    timer = Timer()
    results = []
    wait_time = 1

    for frame_id, batch in enumerate(dataloader):
        if frame_id % 20 == 0:
            logger.info('Processing frame {} ({:.2f} fps)'.format(frame_id, 1./max(1e-5, timer.average_time)))


        frame, bboxs, tlwhs = batch

        #run detection
        timer.tic()
        detections = detector.detect(frame)
        if detections is not None:
            detections[:, 2:] = detections[:, 2:] - detections[:, :2]

        # run tracking
        online_targets = tracker.update(frame, detections)
        online_tlwhs = []
        online_ids = []
        for t in online_targets:
            online_tlwhs.append(t.tlwh)
            online_ids.append(t.track_id)
        timer.toc()

        # save results
        results.append((frame_id + 1, online_tlwhs, online_ids))

        online_im = plot_tracking(frame, online_tlwhs, online_ids, frame_id=frame_id,
                                      fps=1. / timer.average_time)

        if show_image:
            cv2.imshow('online_im', online_im)
            # cv2.waitKey(0)
        if save_dir is not None:
            cv2.imwrite(os.path.join(save_dir, '{:05d}.jpg'.format(frame_id)), online_im)

        key = cv2.waitKey(wait_time)
        key = chr(key % 128).lower()
        if key == 'q':
            exit(0)
        elif key == 'p':
            cv2.waitKey(0)
        elif key == 'a':
            wait_time = int(not wait_time)

    # save results
    write_results(result_filename, results, data_type)

def main(data_root='data/football',
         seqs=('1',), exp_name='demo', save_image=False, show_image=True):
    logger.setLevel(logging.INFO)
    result_root = os.path.join(data_root, '..', 'results', exp_name)
    mkdirs(result_root)
    data_type = 'mot'

    # run tracking
    for seq in seqs:
        output_dir = os.path.join(data_root, 'outputs', seq) if save_image else None

        logger.info('start seq: {}'.format(seq))
        loader = get_loader(root=data_root, name=seq, min_height=0, num_workers=0, batch_size=1)
        result_filename = os.path.join(result_root, '{}.txt'.format(seq))
        eval_seq(loader, data_type, result_filename,
                 save_dir=output_dir, show_image=show_image)

if __name__ == '__main__':

    seqs_str = '''1
                2
                3
                4
                6
                7'''

    seqs = [seq.strip() for seq in seqs_str.split()]

    main(data_root='data/football',
         seqs=seqs,
         exp_name='fot_val_2',
         save_image=False,
         show_image=True)