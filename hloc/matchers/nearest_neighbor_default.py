import torch

from ..utils.base_model import BaseModel


def find_nn(sim, ratio_thresh, distance_thresh):
    sim_nn, ind_nn = sim.topk(2 if ratio_thresh else 1, dim=-1, largest=True)
    dist_nn = 2 * (1 - sim_nn)
    mask = torch.ones(ind_nn.shape[:-1], dtype=torch.bool, device=sim.device)
    if ratio_thresh:
        mask = mask & (dist_nn[..., 0] <= (ratio_thresh**2) * dist_nn[..., 1])
    if distance_thresh:
        mask = mask & (dist_nn[..., 0] <= distance_thresh**2)
    matches = torch.where(mask, ind_nn[..., 0], ind_nn.new_tensor(-1))
    scores = torch.where(mask, (sim_nn[..., 0] + 1) / 2, sim_nn.new_tensor(0))
    return matches, scores


def mutual_check(m0, m1):
    inds0 = torch.arange(m0.shape[-1], device=m0.device)
    loop = torch.gather(m1, -1, torch.where(m0 > -1, m0, m0.new_tensor(0)))
    ok = (m0 > -1) & (inds0 == loop)
    m0_new = torch.where(ok, m0, m0.new_tensor(-1))
    return m0_new


class NearestNeighbor(BaseModel):
    default_conf = {
        "ratio_threshold": None,
        "distance_threshold": None,
        "do_mutual_check": True,
    }
    required_inputs = ["descriptors0", "descriptors1"]

    def _init(self, conf):
        pass

    def _forward(self, data):
        if data["descriptors0"].size(-1) == 0 or data["descriptors1"].size(-1) == 0:
            matches0 = torch.full(
                data["descriptors0"].shape[:2], -1, device=data["descriptors0"].device
            )
            return {
                "matches0": matches0,
                "matching_scores0": torch.zeros_like(matches0),
            }
        ratio_threshold = self.conf["ratio_threshold"]
        if data["descriptors0"].size(-1) == 1 or data["descriptors1"].size(-1) == 1:
            ratio_threshold = None

        # Handle both 2D (N, D) and 3D (B, N, D) descriptors
        desc0 = data["descriptors0"]
        desc1 = data["descriptors1"]

        # Ensure descriptors are in the right shape
        if desc0.ndim == 2:
            desc0 = desc0.unsqueeze(0)  # (1, N, D)
        if desc1.ndim == 2:
            desc1 = desc1.unsqueeze(0)  # (1, M, D)

        # Check if descriptor dimensions match
        if desc0.shape[2] != desc1.shape[2]:
            # If dimensions don't match, use smaller dimension
            min_dim = min(desc0.shape[2], desc1.shape[2])
            desc0 = desc0[:, :, :min_dim]
            desc1 = desc1[:, :, :min_dim]

        # Transpose to (B, D, N) and (B, D, M) for einsum
        desc0 = desc0.transpose(1, 2)  # (B, D, N)
        desc1 = desc1.transpose(1, 2)  # (B, D, M)

        sim = torch.einsum("bdn,bdm->bnm", desc0, desc1)
        matches0, scores0 = find_nn(
            sim, ratio_threshold, self.conf["distance_threshold"]
        )
        if self.conf["do_mutual_check"]:
            matches1, scores1 = find_nn(
                sim.transpose(1, 2), ratio_threshold, self.conf["distance_threshold"]
            )
            matches0 = mutual_check(matches0, matches1)
        return {
            "matches0": matches0,
            "matching_scores0": scores0,
        }
