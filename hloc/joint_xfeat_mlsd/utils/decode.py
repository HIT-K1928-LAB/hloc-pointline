import torch
import torch.nn as nn
import torch.nn.functional as F


def deccode_lines_TP(tpMap, score_thresh=0.1, len_thresh=2, topk_n=1000, ksize=3):
    '''
    tpMap:
        center: tpMap[1, 0, :, :]
        displacement: tpMap[1, 1:5, :, :]
    '''
    b, c, h, w = tpMap.shape
    assert b == 1, 'only support bsize==1'
    displacement = tpMap[:, 1:5, :, :]
    center = tpMap[:, 0, :, :]
    heat = torch.sigmoid(center)
    hmax = F.max_pool2d(heat, (ksize, ksize), stride=1, padding=(ksize - 1) // 2)
    keep = (hmax == heat).float()
    heat = heat * keep
    heat = heat.reshape(-1, )

    scores, indices = torch.topk(heat, topk_n, dim=-1, largest=True)
    valid_inx = torch.where(scores > score_thresh)
    scores = scores[valid_inx]
    indices = indices[valid_inx]

    yy = torch.floor_divide(indices, w).unsqueeze(-1)
    xx = torch.fmod(indices, w).unsqueeze(-1)
    center_ptss = torch.cat((xx, yy), dim=-1)

    start_point = center_ptss + displacement[0, :2, yy, xx].reshape(2, -1).permute(1, 0)
    end_point = center_ptss + displacement[0, 2:, yy, xx].reshape(2, -1).permute(1, 0)

    lines = torch.cat((start_point, end_point), dim=-1)

    lines_swap = torch.cat((end_point, start_point), dim=-1)

    all_lens = (end_point - start_point) ** 2
    all_lens = all_lens.sum(dim=-1)
    all_lens = torch.sqrt(all_lens)
    valid_inx = torch.where(all_lens > len_thresh)

    center_ptss = center_ptss[valid_inx]
    lines = lines[valid_inx]
    lines_swap = lines_swap[valid_inx]
    scores = scores[valid_inx]

    return center_ptss, lines, lines_swap, scores


def deccode_lines_TP_with_descriptor(tpMap, descriptor_map, descriptor_extractor,
                                     score_thresh=0.1, len_thresh=2, topk_n=1000,
                                     ksize=3, coord_scale=512.0, num_samples=None):
    center_ptss, lines, lines_swap, scores = deccode_lines_TP(
        tpMap,
        score_thresh=score_thresh,
        len_thresh=len_thresh,
        topk_n=topk_n,
        ksize=ksize,
    )

    line_descriptors = None
    if descriptor_map is not None and descriptor_extractor is not None:
        line_descriptors = descriptor_extractor(
            descriptor_map,
            lines,
            coord_scale=coord_scale,
            num_samples=num_samples,
        )

    return center_ptss, lines, lines_swap, scores, line_descriptors
