from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from .utils import _gather_feat, _tranpose_and_gather_feat


# NMS based on heatmap scores to extract the peak keypoints
def _nms(heat, kernel=3):
    # _, c, h, w = heat.size()
    #
    pad = (kernel - 1) // 2
    #
    # hmax = nn.functional.max_pool2d(
    #     heat, (kernel, kernel), stride=1, padding=pad)
    # keep_1 = (hmax == heat).float()
    # heat_1 = heat * keep_1
    #
    # heat_max = (heat_1[0] >= 0.6).float()
    #
    # # 去除置信度大于阈值，周边参数影响
    # heat_ones = torch.ones_like(heat[0])
    # y_ind = torch.nonzero(heat_max[0])
    # for j in range(len(y_ind)):
    #     if y_ind[j][0] == 0 or y_ind[j][0] == 152 - 1:
    #         continue
    #     if y_ind[j][1] == 0 or y_ind[j][0] == 272 - 1:
    #         continue
    #     heat_ones[0][y_ind[j][0] - 1:y_ind[j][0] + 2, y_ind[j][1] - 1:y_ind[j][1] + 2] = 0
    #
    # keep_2 = heat_ones.unsqueeze(0)
    # heat_refine = keep_2 * heat
    # hmax_2 = nn.functional.max_pool2d(
    #     heat_refine, (kernel, kernel), stride=1, padding=pad)
    # keep_3 = (hmax_2 == heat).float()
    #
    # keep = keep_1 + keep_3
    # keep = torch.where(keep > 0, 1.0, 0.0)
    #
    # keep_1 = keep_1.cpu().numpy()
    # keep_2 = keep_2.cpu().numpy()
    # keep_3 = keep_3.cpu().numpy()
    # keep = keep.cpu().numpy()
    # XX = keep_1 == keep

    hmax = nn.functional.max_pool2d(
    heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()

    return heat * keep


def _topk_channel(scores, K=40):
    batch, cat, height, width = scores.size()

    topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)

    topk_inds = topk_inds % (height * width)
    topk_ys = torch.true_divide(topk_inds, width).int().float()
    topk_xs = (topk_inds % width).int().float()

    return topk_scores, topk_inds, topk_ys, topk_xs


def _topk(scores, K=40):
    batch, cat, height, width = scores.size()

    topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)

    topk_inds = topk_inds % (height * width)
    topk_ys = torch.true_divide(topk_inds, width).int().float()
    topk_xs = (topk_inds % width).int().float()

    topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
    topk_clses = torch.true_divide(topk_ind, K).int()
    topk_inds = _gather_feat(
        topk_inds.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_ys = _gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_xs = _gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, K)

    return topk_score, topk_inds, topk_clses, topk_ys, topk_xs


def mot_decode(heat, wh, reg=None, ltrb=False, K=100):
    batch, cat, height, width = heat.size()

    # heat = torch.sigmoid(heat)
    # perform nms on heatmaps
    heat = _nms(heat)  # _nms 类

    scores, inds, clses, ys, xs = _topk(heat, K=K)  # _topk 类

    if reg is not None:
        reg = _tranpose_and_gather_feat(reg, inds)
        reg = reg.view(batch, K, 2)
        xs = xs.view(batch, K, 1) + reg[:, :, 0:1]
        ys = ys.view(batch, K, 1) + reg[:, :, 1:2]
    else:
        xs = xs.view(batch, K, 1) + 0.5
        ys = ys.view(batch, K, 1) + 0.5
    wh = _tranpose_and_gather_feat(wh, inds)

    if ltrb:
        wh = wh.view(batch, K, 4)
    else:
        wh = wh.view(batch, K, 2)
    clses = clses.view(batch, K, 1).float()
    scores = scores.view(batch, K, 1)

    if ltrb:
        bboxes = torch.cat([xs - wh[..., 0:1],
                            ys - wh[..., 1:2],
                            xs + wh[..., 2:3],
                            ys + wh[..., 3:4]], dim=2)
    else:
        bboxes = torch.cat([xs - wh[..., 0:1] / 2,
                            ys - wh[..., 1:2] / 2,
                            xs + wh[..., 0:1] / 2,
                            ys + wh[..., 1:2] / 2], dim=2)
    detections = torch.cat([bboxes, scores, clses], dim=2)

    return detections, inds
