
import math

import cv2
import numpy as np
import torch

from ..utils.base_model import BaseModel


EPS = 1e-8


def _raise_min(current, target):
    if current is None:
        return target
    return max(current, target)


def _lower_max(current, target):
    if current is None or current <= 0:
        return target
    return min(current, target)


def _apply_precision_preset(conf):
    conf = dict(conf)
    preset = conf.get('precision_preset', 'strict')
    if preset == 'default':
        return conf
    if preset == 'strict':
        conf['post_min_line_score'] = max(float(conf.get('post_min_line_score', 0.0)), 0.10)
        conf['post_min_line_length_px'] = max(float(conf.get('post_min_line_length_px', 0.0)), 30.0)
        conf['max_lines_per_image'] = int(_lower_max(conf.get('max_lines_per_image', 0), 120))
        conf['line_nms_center_thresh'] = max(float(conf.get('line_nms_center_thresh', 0.0)), 20.0)
        conf['line_nms_angle_thresh_deg'] = max(float(conf.get('line_nms_angle_thresh_deg', 0.0)), 8.0)
        conf['line_nms_endpoint_thresh'] = max(float(conf.get('line_nms_endpoint_thresh', 0.0)), 24.0)
        conf['min_similarity'] = max(float(conf.get('min_similarity', 0.75)), 0.75)
        conf['ratio_threshold'] = min(float(conf.get('ratio_threshold', 0.82)), 0.82)
        conf['ransac_reproj_thresh'] = min(float(conf.get('ransac_reproj_thresh', 8.0)), 8.0)
        conf['line_inlier_endpoint_thresh'] = min(float(conf.get('line_inlier_endpoint_thresh', 18.0)), 18.0)
        conf['line_inlier_center_thresh'] = min(float(conf.get('line_inlier_center_thresh', 14.0)), 14.0)
        conf['line_inlier_angle_thresh_deg'] = min(float(conf.get('line_inlier_angle_thresh_deg', 10.0)), 10.0)
        conf['line_inlier_length_ratio_min'] = max(float(conf.get('line_inlier_length_ratio_min', 0.0)), 0.60)
        return conf
    if preset == 'very_strict':
        conf['post_min_line_score'] = max(float(conf.get('post_min_line_score', 0.0)), 0.14)
        conf['post_min_line_length_px'] = max(float(conf.get('post_min_line_length_px', 0.0)), 40.0)
        conf['max_lines_per_image'] = int(_lower_max(conf.get('max_lines_per_image', 0), 80))
        conf['line_nms_center_thresh'] = max(float(conf.get('line_nms_center_thresh', 0.0)), 16.0)
        conf['line_nms_angle_thresh_deg'] = max(float(conf.get('line_nms_angle_thresh_deg', 0.0)), 6.0)
        conf['line_nms_endpoint_thresh'] = max(float(conf.get('line_nms_endpoint_thresh', 0.0)), 18.0)
        conf['min_similarity'] = max(float(conf.get('min_similarity', 0.86)), 0.86)
        conf['ratio_threshold'] = min(float(conf.get('ratio_threshold', 0.78)), 0.78)
        conf['ransac_reproj_thresh'] = min(float(conf.get('ransac_reproj_thresh', 6.0)), 6.0)
        conf['line_inlier_endpoint_thresh'] = min(float(conf.get('line_inlier_endpoint_thresh', 14.0)), 14.0)
        conf['line_inlier_center_thresh'] = min(float(conf.get('line_inlier_center_thresh', 10.0)), 10.0)
        conf['line_inlier_angle_thresh_deg'] = min(float(conf.get('line_inlier_angle_thresh_deg', 8.0)), 8.0)
        conf['line_inlier_length_ratio_min'] = max(float(conf.get('line_inlier_length_ratio_min', 0.0)), 0.70)
        return conf
    return conf


def _to_numpy_2d(tensor):
    arr = tensor.detach().cpu().float().numpy()
    if arr.ndim == 3:
        arr = arr[0]
    return arr


def compute_line_lengths(lines_xyxy):
    if lines_xyxy.size == 0:
        return np.zeros((0,), dtype=np.float32)
    delta = lines_xyxy[:, 2:] - lines_xyxy[:, :2]
    return np.linalg.norm(delta, axis=1).astype(np.float32)


def line_angle_diff_deg(line_a_xyxy, line_b_xyxy):
    vec_a = line_a_xyxy[2:] - line_a_xyxy[:2]
    vec_b = line_b_xyxy[2:] - line_b_xyxy[:2]
    na = np.linalg.norm(vec_a)
    nb = np.linalg.norm(vec_b)
    if na < EPS or nb < EPS:
        return 180.0
    cos_val = np.clip(abs(float(np.dot(vec_a, vec_b))) / max(float(na * nb), EPS), 0.0, 1.0)
    return math.degrees(math.acos(cos_val))


def endpoint_alignment_error(projected_line_xyxy, target_line_xyxy):
    p0, p1 = projected_line_xyxy[:2], projected_line_xyxy[2:]
    t0, t1 = target_line_xyxy[:2], target_line_xyxy[2:]
    direct = 0.5 * (np.linalg.norm(p0 - t0) + np.linalg.norm(p1 - t1))
    swapped = 0.5 * (np.linalg.norm(p0 - t1) + np.linalg.norm(p1 - t0))
    return float(min(direct, swapped))


def center_alignment_error(projected_line_xyxy, target_line_xyxy):
    center_a = (projected_line_xyxy[:2] + projected_line_xyxy[2:]) * 0.5
    center_b = (target_line_xyxy[:2] + target_line_xyxy[2:]) * 0.5
    return float(np.linalg.norm(center_a - center_b))


def line_length_ratio(line_a_xyxy, line_b_xyxy):
    len_a = np.linalg.norm(line_a_xyxy[2:] - line_a_xyxy[:2])
    len_b = np.linalg.norm(line_b_xyxy[2:] - line_b_xyxy[:2])
    if len_a < EPS or len_b < EPS:
        return 0.0
    return float(min(len_a, len_b) / max(len_a, len_b))


def apply_homography(points_xy, homography):
    if points_xy.size == 0:
        return points_xy.astype(np.float32)
    points = points_xy.reshape(-1, 1, 2).astype(np.float32)
    projected = cv2.perspectiveTransform(points, homography).reshape(-1, 2)
    return projected.astype(np.float32)


def pairwise_cosine_similarity(desc1, desc2):
    if desc1.size == 0 or desc2.size == 0:
        return np.zeros((desc1.shape[0], desc2.shape[0]), dtype=np.float32)
    desc1 = desc1 / np.clip(np.linalg.norm(desc1, axis=1, keepdims=True), EPS, None)
    desc2 = desc2 / np.clip(np.linalg.norm(desc2, axis=1, keepdims=True), EPS, None)
    return np.matmul(desc1, desc2.T).astype(np.float32)


def mutual_descriptor_matches(desc1, desc2, min_similarity=0.75, ratio_thresh=0.82):
    if desc1.size == 0 or desc2.size == 0:
        return []
    sim = pairwise_cosine_similarity(desc1, desc2)
    if sim.shape[1] < 2:
        ratio_thresh = None
    best12 = np.argmax(sim, axis=1)
    best21 = np.argmax(sim, axis=0)
    matches = []
    for idx1, idx2 in enumerate(best12.tolist()):
        if best21[idx2] != idx1:
            continue
        similarity = float(sim[idx1, idx2])
        if similarity < min_similarity:
            continue
        if ratio_thresh is not None:
            row = sim[idx1]
            top2 = np.partition(row, -2)[-2:]
            top2 = np.sort(top2)[::-1]
            d0 = 2.0 * (1.0 - top2[0])
            d1 = 2.0 * (1.0 - top2[1])
            if d0 > (ratio_thresh ** 2) * max(d1, EPS):
                continue
        matches.append({'idx1': idx1, 'idx2': idx2, 'similarity': similarity})
    return matches


def estimate_homography_from_matches(matches, lines1_xyxy, lines2_xyxy, reproj_thresh):
    if len(matches) < 4:
        return None, np.zeros((len(matches),), dtype=bool)
    centers1 = np.asarray([(lines1_xyxy[m['idx1'], :2] + lines1_xyxy[m['idx1'], 2:]) * 0.5 for m in matches], dtype=np.float32)
    centers2 = np.asarray([(lines2_xyxy[m['idx2'], :2] + lines2_xyxy[m['idx2'], 2:]) * 0.5 for m in matches], dtype=np.float32)
    homography, mask = cv2.findHomography(centers1, centers2, cv2.RANSAC, reproj_thresh)
    if mask is None:
        return homography, np.zeros((len(matches),), dtype=bool)
    return homography, mask.reshape(-1).astype(bool)


def line_nms(lines, scores, center_thresh, angle_thresh_deg, endpoint_thresh):
    order = np.argsort(-scores)
    keep = []
    centers = 0.5 * (lines[:, :2] + lines[:, 2:])
    for idx in order.tolist():
        suppress = False
        for kept in keep:
            if np.linalg.norm(centers[idx] - centers[kept]) > center_thresh:
                continue
            if line_angle_diff_deg(lines[idx], lines[kept]) > angle_thresh_deg:
                continue
            if endpoint_alignment_error(lines[idx], lines[kept]) > endpoint_thresh:
                continue
            suppress = True
            break
        if not suppress:
            keep.append(idx)
    return np.asarray(keep, dtype=np.int64)


def post_filter_line_result(lines, scores, descriptors, conf):
    count = lines.shape[0]
    keep = np.arange(count, dtype=np.int64)
    if count == 0:
        return lines, scores, descriptors, keep

    mask = np.ones((count,), dtype=bool)
    min_score = float(conf.get('post_min_line_score', 0.0))
    min_len = float(conf.get('post_min_line_length_px', 0.0))
    if min_score > 0.0:
        mask &= scores >= min_score
    if min_len > 0.0:
        mask &= compute_line_lengths(lines) >= min_len
    keep = keep[mask]
    lines = lines[mask]
    scores = scores[mask]
    descriptors = descriptors[mask]
    if lines.shape[0] == 0:
        return lines, scores, descriptors, keep

    center_thresh = float(conf.get('line_nms_center_thresh', 0.0))
    angle_thresh = float(conf.get('line_nms_angle_thresh_deg', 0.0))
    endpoint_thresh = float(conf.get('line_nms_endpoint_thresh', 0.0))
    if center_thresh > 0.0 and angle_thresh > 0.0 and endpoint_thresh > 0.0 and lines.shape[0] > 1:
        nms_keep = line_nms(lines, scores, center_thresh, angle_thresh, endpoint_thresh)
        keep = keep[nms_keep]
        lines = lines[nms_keep]
        scores = scores[nms_keep]
        descriptors = descriptors[nms_keep]

    max_lines = int(conf.get('max_lines_per_image', 0))
    if max_lines > 0 and lines.shape[0] > max_lines:
        order = np.argsort(-scores)[:max_lines]
        keep = keep[order]
        lines = lines[order]
        scores = scores[order]
        descriptors = descriptors[order]

    return lines, scores, descriptors, keep


def evaluate_matches_against_homography(matches, lines1_xyxy, lines2_xyxy, homography, endpoint_thresh, angle_thresh_deg, center_thresh, length_ratio_min=0.0):
    evaluated = []
    for match in matches:
        line1 = lines1_xyxy[match['idx1']]
        line2 = lines2_xyxy[match['idx2']]
        projected = apply_homography(line1.reshape(2, 2), homography).reshape(-1)
        endpoint_error = endpoint_alignment_error(projected, line2)
        center_error = center_alignment_error(projected, line2)
        angle_error = line_angle_diff_deg(projected, line2)
        length_ratio = line_length_ratio(projected, line2)
        record = dict(match)
        record['correct'] = bool(
            endpoint_error <= endpoint_thresh
            and center_error <= center_thresh
            and angle_error <= angle_thresh_deg
            and (length_ratio_min <= 0.0 or length_ratio >= length_ratio_min)
        )
        record['endpoint_error'] = float(endpoint_error)
        record['center_reproj_error'] = float(center_error)
        record['angle_diff_deg'] = float(angle_error)
        record['length_ratio'] = float(length_ratio)
        evaluated.append(record)
    return evaluated


def _mean_or_zero(values):
    if not values:
        return 0.0
    return float(np.mean(np.asarray(values, dtype=np.float32)))


class LineNearestNeighbor(BaseModel):
    default_conf = {
        'ratio_threshold': 0.82,
        'min_similarity': 0.75,
        'precision_preset': 'strict',
        'post_min_line_score': 0.0,
        'post_min_line_length_px': 0.0,
        'max_lines_per_image': 0,
        'line_nms_center_thresh': 0.0,
        'line_nms_angle_thresh_deg': 0.0,
        'line_nms_endpoint_thresh': 0.0,
        'ransac_reproj_thresh': 8.0,
        'line_inlier_endpoint_thresh': 18.0,
        'line_inlier_center_thresh': 14.0,
        'line_inlier_angle_thresh_deg': 10.0,
        'line_inlier_length_ratio_min': 0.60,
    }
    required_inputs = ['line_descriptors0', 'line_descriptors1', 'line_segments0', 'line_segments1']

    def _init(self, conf):
        self.conf = _apply_precision_preset(conf)

    def _forward(self, data):
        device = data['line_descriptors0'].device
        lines0_all = _to_numpy_2d(data['line_segments0'])
        lines1_all = _to_numpy_2d(data['line_segments1'])
        desc0_all = _to_numpy_2d(data['line_descriptors0'])
        desc1_all = _to_numpy_2d(data['line_descriptors1'])
        scores0_all = _to_numpy_2d(data.get('line_scores0', torch.zeros((1, lines0_all.shape[0]), device=device))).reshape(-1)
        scores1_all = _to_numpy_2d(data.get('line_scores1', torch.zeros((1, lines1_all.shape[0]), device=device))).reshape(-1)

        lines0, scores0, desc0, keep0 = post_filter_line_result(lines0_all, scores0_all, desc0_all, self.conf)
        lines1, scores1, desc1, keep1 = post_filter_line_result(lines1_all, scores1_all, desc1_all, self.conf)

        full_matches0 = np.full((lines0_all.shape[0],), -1, dtype=np.int32)
        full_scores0 = np.zeros((lines0_all.shape[0],), dtype=np.float32)
        full_verified0 = np.zeros((lines0_all.shape[0],), dtype=np.uint8)

        matches = mutual_descriptor_matches(
            desc0,
            desc1,
            min_similarity=float(self.conf.get('min_similarity', 0.75)),
            ratio_thresh=self.conf.get('ratio_threshold', 0.82),
        )
        candidate_count = len(matches)
        verified_count = 0
        verified_similarities = []
        verified_endpoint_errors = []
        verified_center_errors = []
        verified_angle_errors = []

        if candidate_count > 0:
            homography, inlier_mask = estimate_homography_from_matches(
                matches, lines0, lines1, reproj_thresh=float(self.conf.get('ransac_reproj_thresh', 8.0))
            )
            if homography is not None:
                evaluated = evaluate_matches_against_homography(
                    matches,
                    lines0,
                    lines1,
                    homography,
                    endpoint_thresh=float(self.conf.get('line_inlier_endpoint_thresh', 18.0)),
                    angle_thresh_deg=float(self.conf.get('line_inlier_angle_thresh_deg', 10.0)),
                    center_thresh=float(self.conf.get('line_inlier_center_thresh', 14.0)),
                    length_ratio_min=float(self.conf.get('line_inlier_length_ratio_min', 0.60)),
                )
                for is_inlier, match in zip(inlier_mask.tolist(), evaluated):
                    orig0 = int(keep0[match['idx1']])
                    orig1 = int(keep1[match['idx2']])
                    full_matches0[orig0] = orig1
                    full_scores0[orig0] = float(match['similarity'])
                    is_verified = bool(is_inlier and match['correct'])
                    full_verified0[orig0] = 1 if is_verified else 0
                    if is_verified:
                        verified_count += 1
                        verified_similarities.append(float(match['similarity']))
                        verified_endpoint_errors.append(float(match['endpoint_error']))
                        verified_center_errors.append(float(match['center_reproj_error']))
                        verified_angle_errors.append(float(match['angle_diff_deg']))
            else:
                for match in matches:
                    orig0 = int(keep0[match['idx1']])
                    orig1 = int(keep1[match['idx2']])
                    full_matches0[orig0] = orig1
                    full_scores0[orig0] = float(match['similarity'])

        verified_ratio = float(verified_count / max(candidate_count, 1)) if candidate_count > 0 else 0.0
        mean_similarity = _mean_or_zero(verified_similarities)
        mean_endpoint_error = _mean_or_zero(verified_endpoint_errors)
        mean_center_error = _mean_or_zero(verified_center_errors)
        mean_angle_error = _mean_or_zero(verified_angle_errors)

        return {
            'line_matches0': torch.from_numpy(full_matches0).to(device=device).unsqueeze(0),
            'line_matching_scores0': torch.from_numpy(full_scores0).to(device=device).unsqueeze(0),
            'line_verified_mask0': torch.from_numpy(full_verified0).to(device=device).unsqueeze(0),
            'line_match_stats': torch.tensor([[candidate_count, verified_count]], dtype=torch.int32, device=device),
            'line_candidate_matches': torch.tensor([[candidate_count]], dtype=torch.int32, device=device),
            'line_verified_matches': torch.tensor([[verified_count]], dtype=torch.int32, device=device),
            'line_verified_ratio': torch.tensor([[verified_ratio]], dtype=torch.float32, device=device),
            'line_mean_verified_similarity': torch.tensor([[mean_similarity]], dtype=torch.float32, device=device),
            'line_mean_verified_endpoint_error': torch.tensor([[mean_endpoint_error]], dtype=torch.float32, device=device),
            'line_mean_verified_center_error': torch.tensor([[mean_center_error]], dtype=torch.float32, device=device),
            'line_mean_verified_angle_error_deg': torch.tensor([[mean_angle_error]], dtype=torch.float32, device=device),
        }
