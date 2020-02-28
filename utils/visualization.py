import numpy as np
import cv2

def tlwhs_to_tlbrs(tlwhs):
    tlbrs = np.copy(tlwhs)
    if len(tlbrs) == 0:
        return tlbrs
    tlbrs[:, 2] += tlwhs[:, 0]
    tlbrs[:, 3] += tlwhs[:, 1]
    return tlbrs


def get_color(idx):
    idx = idx * 3
    color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)

    return color


def resize_image(image, max_size=800):
    if max(image.shape[:2]) > max_size:
        scale = float(max_size) / max(image.shape[:2])
        image = cv2.resize(image, None, fx=scale, fy=scale)
    return image


def plot_tracking(image, tlwhs, obj_ids, scores=None, frame_id=0, fps=0., ids2=None):
    im = np.copy(image)
    im_h, im_w = im.shape[:2]

    top_view = np.zeros([im_w, im_w, 3], dtype=np.uint8) + 255

    text_scale = max(1, image.shape[1] / 800.)
    text_thickness = 2 if text_scale > 1.1 else 1
    line_thickness = max(1, int(image.shape[1] / 250.))

    radius = max(5, int(im_w/140.))
    cv2.putText(im, 'frame: %d fps: %.2f num: %d' % (frame_id, fps, len(tlwhs)),
                (0, int(15 * text_scale)), cv2.FONT_HERSHEY_PLAIN, text_scale, (0, 0, 255), thickness=2)
    # cv2.putText(im, 'frame: %d  num: %d' % (frame_id, len(tlwhs)),
    #             (0, int(15 * text_scale)), cv2.FONT_HERSHEY_PLAIN, text_scale, (0, 0, 255), thickness=2) # 无fps

    for i, tlwh in enumerate(tlwhs):
        x1, y1, w, h = tlwh
        intbox = tuple(map(int, (x1, y1, x1 + w, y1 + h)))
        obj_id = int(obj_ids[i])
        id_text = '{}'.format(int(obj_id))
        if ids2 is not None:
            id_text = id_text + ', {}'.format(int(ids2[i]))
        _line_thickness = 1 if obj_id <= 0 else line_thickness
        color = get_color(abs(obj_id))

        cv2.rectangle(im, intbox[0:2], intbox[2:4], color=color, thickness=1)
        cv2.putText(im, id_text, (intbox[0], intbox[1] + 30), cv2.FONT_HERSHEY_PLAIN, text_scale, (0, 0, 255),
                    thickness=1)

    return im


def plot_trajectory(image, tlwhs, track_ids):
    image = image.copy()
    for one_tlwhs, track_id in zip(tlwhs, track_ids):
        color = get_color(int(track_id))
        for tlwh in one_tlwhs:
            x1, y1, w, h = tuple(map(int, tlwh))
            cv2.circle(image, (int(x1 + 0.5 * w), int(y1 + h)), 2, color, thickness=2)

    return image


def plot_detections(image, tlbrs, scores=None, color=(255, 0, 0), ids=None):
    im = np.copy(image)
    text_scale = max(1, image.shape[1] / 800.)
    thickness = 2 if text_scale > 1.3 else 1
    for i, det in enumerate(tlbrs):
        x1, y1, x2, y2 = np.asarray(det[:4], dtype=np.int)
        if len(det) >= 7:
            label = 'det' if det[5] > 0 else 'trk'
            if ids is not None:
                text = '{}# {:.2f}: {:d}'.format(label, det[6], ids[i])
                cv2.putText(im, text, (x1, y1 + 30), cv2.FONT_HERSHEY_PLAIN, text_scale, (0, 255, 255),
                            thickness=thickness)
            else:
                text = '{}# {:.2f}'.format(label, det[6])

        if scores is not None:
            text = '{:.2f}'.format(scores[i])
            cv2.putText(im, text, (x1, y1 + 30), cv2.FONT_HERSHEY_PLAIN, text_scale, (0, 255, 255),
                        thickness=thickness)

        cv2.rectangle(im, (x1, y1), (x2, y2), color, 1)

    return im

'''
frames: image, frame_id, fps
detections: det_tlwhs, det_color=(0,0,255) red
results: result_tlwhs, result_ids, result_color=(0,255,0) green
gound-truth: gt_tlwhs, gt_ids, gt_color=(255,0,0) blue
'''
def plot_information(image, det_tlwhs, result_tlwhs, result_ids, gt_tlwhs, gt_ids, frame_id=0, fps=0.):
    im = np.copy(image)

    det_switch = 1
    result_switch = 1
    gt_switch = 1

    det_color = (0, 0, 255)
    result_color = (0, 255, 0)
    gt_color = (255, 0, 0)
    text_scale = max(0.5, image.shape[1] / 1200.)
    text_thickness = 1 if text_scale > 0.6 else 1
    line_thickness = max(0.5, int(image.shape[1] / 1000.))

    cv2.putText(im, 'frame: %d fps: %.2f num: %d' % (frame_id, fps, len(result_tlwhs)),
                (0, int(15 * text_scale)), cv2.FONT_HERSHEY_PLAIN, text_scale, (255, 255, 255), thickness=2)

    # plot detections
    if det_switch:
        for det_i, det_tlwh in enumerate(det_tlwhs):
            x1, y1, w, h = det_tlwh
            intbox = tuple(map(int, (x1, y1, x1 + w, y1 + h)))

            cv2.rectangle(im, intbox[0:2], intbox[2:4], det_color, thickness=line_thickness*2)

    # plot results
    if result_switch:
        for i, tlwh in enumerate(result_tlwhs):
            x1, y1, w, h = tlwh
            intbox = tuple(map(int, (x1, y1, x1 + w, y1 + h)))
            obj_id = int(result_ids[i])
            id_text = '{}'.format(int(obj_id))
            _line_thickness = 0.5 if obj_id <= 0 else line_thickness

            cv2.rectangle(im, intbox[0:2], intbox[2:4], result_color, thickness=_line_thickness)
            cv2.putText(im, id_text, (intbox[0], intbox[1] + 30), cv2.FONT_HERSHEY_PLAIN, text_scale, result_color,
                        thickness=text_thickness)

    # plot gt
    if gt_switch:
        for gt_i, gt_tlwh in enumerate(gt_tlwhs):
            x1, y1, w, h = gt_tlwh
            intbox = tuple(map(int, (x1, y1, x1 + w, y1 + h)))
            obj_id = int(gt_ids[gt_i])
            id_text = '{}'.format(int(obj_id))
            _line_thickness = 0.5 if obj_id <= 0 else line_thickness

            cv2.rectangle(im, intbox[0:2], intbox[2:4], gt_color, thickness=_line_thickness*2)
            cv2.putText(im, id_text, (intbox[0], intbox[1] + 30), cv2.FONT_HERSHEY_PLAIN, text_scale, gt_color,
                        thickness=text_thickness)

    return im


