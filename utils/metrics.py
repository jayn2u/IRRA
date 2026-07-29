from prettytable import PrettyTable
import torch
import numpy as np
import os
import torch.nn.functional as F
import logging


def rank(similarity, q_pids, g_pids, max_rank=10, get_mAP=True, chunk_size=2048):
    if not get_mAP:
        # acclerate sort with topk
        _, indices = torch.topk(
            similarity, k=max_rank, dim=1, largest=True, sorted=True
        )  # q * topk
        pred_labels = g_pids[indices.cpu()]  # q * k
        matches = pred_labels.eq(q_pids.view(-1, 1))  # q * k

        all_cmc = matches[:, :max_rank].cumsum(1) # cumulative sum
        all_cmc[all_cmc > 1] = 1
        all_cmc = all_cmc.float().mean(0) * 100
        return all_cmc, indices

    # get_mAP needs the full ranking (not just top-k), so `indices` is q * g.
    # On large galleries (e.g. ICFG-PEDES: ~19.8k x 19.8k) that tensor alone is
    # several GiB, and materializing it for every query at once can OOM on top
    # of whatever training memory is still resident. Each query's ranking is
    # independent of every other query's, so we can process queries in chunks
    # and accumulate the running sums that feed the final means -- this is
    # bit-for-bit the same aggregate formula as computing everything in one
    # shot, just with a bounded memory footprint per step.
    num_q = similarity.size(0)
    cmc_sum = torch.zeros(max_rank)
    ap_sum = 0.0
    inp_sum = 0.0
    indices_chunks = []

    for start in range(0, num_q, chunk_size):
        end = min(start + chunk_size, num_q)
        sim_chunk = similarity[start:end]
        q_pids_chunk = q_pids[start:end]

        indices_chunk = torch.argsort(sim_chunk, dim=1, descending=True)
        pred_labels = g_pids[indices_chunk.cpu()]  # chunk * g
        matches = pred_labels.eq(q_pids_chunk.view(-1, 1))  # chunk * g

        chunk_cmc = matches[:, :max_rank].cumsum(1)
        chunk_cmc[chunk_cmc > 1] = 1
        cmc_sum += chunk_cmc.float().sum(0)

        num_rel = matches.sum(1)  # chunk
        tmp_cmc = matches.cumsum(1)  # chunk * g

        inp = [tmp_cmc[i][match_row.nonzero()[-1]] / (match_row.nonzero()[-1] + 1.) for i, match_row in enumerate(matches)]
        inp_sum += torch.cat(inp).sum().item()

        tmp_cmc = [tmp_cmc[:, i] / (i + 1.0) for i in range(tmp_cmc.shape[1])]
        tmp_cmc = torch.stack(tmp_cmc, 1) * matches
        AP = tmp_cmc.sum(1) / num_rel  # chunk
        ap_sum += AP.sum().item()

        indices_chunks.append(indices_chunk.cpu())

    all_cmc = (cmc_sum / num_q) * 100
    mAP = torch.tensor(ap_sum / num_q * 100)
    mINP = torch.tensor(inp_sum / num_q * 100)
    indices = torch.cat(indices_chunks, dim=0)

    return all_cmc, mAP, mINP, indices


class Evaluator():
    def __init__(self, img_loader, txt_loader):
        self.img_loader = img_loader # gallery
        self.txt_loader = txt_loader # query
        self.logger = logging.getLogger("IRRA.eval")

    def _compute_embedding(self, model):
        model = model.eval()
        device = next(model.parameters()).device

        qids, gids, qfeats, gfeats = [], [], [], []
        # text
        for pid, caption in self.txt_loader:
            caption = caption.to(device)
            with torch.no_grad():
                text_feat = model.encode_text(caption)
            qids.append(pid.view(-1)) # flatten 
            qfeats.append(text_feat)
        qids = torch.cat(qids, 0)
        qfeats = torch.cat(qfeats, 0)

        # image
        for pid, img in self.img_loader:
            img = img.to(device)
            with torch.no_grad():
                img_feat = model.encode_image(img)
            gids.append(pid.view(-1)) # flatten 
            gfeats.append(img_feat)
        gids = torch.cat(gids, 0)
        gfeats = torch.cat(gfeats, 0)

        return qfeats, gfeats, qids, gids
    
    def eval(self, model, i2t_metric=False, return_metrics=False):
        """Evaluate retrieval.

        Returns t2i R1 by default; with `return_metrics=True` returns a flat
        dict of all computed metrics (used for per-epoch wandb logging).
        """

        qfeats, gfeats, qids, gids = self._compute_embedding(model)

        qfeats = F.normalize(qfeats, p=2, dim=1) # text features
        gfeats = F.normalize(gfeats, p=2, dim=1) # image features

        similarity = qfeats @ gfeats.t()

        t2i_cmc, t2i_mAP, t2i_mINP, _ = rank(similarity=similarity, q_pids=qids, g_pids=gids, max_rank=10, get_mAP=True)
        t2i_cmc, t2i_mAP, t2i_mINP = t2i_cmc.numpy(), t2i_mAP.numpy(), t2i_mINP.numpy()
        table = PrettyTable(["task", "R1", "R5", "R10", "mAP", "mINP"])
        table.add_row(['t2i', t2i_cmc[0], t2i_cmc[4], t2i_cmc[9], t2i_mAP, t2i_mINP])

        metrics = {
            't2i_R1': float(t2i_cmc[0]),
            't2i_R5': float(t2i_cmc[4]),
            't2i_R10': float(t2i_cmc[9]),
            't2i_mAP': float(t2i_mAP),
            't2i_mINP': float(t2i_mINP),
        }

        if i2t_metric:
            i2t_cmc, i2t_mAP, i2t_mINP, _ = rank(similarity=similarity.t(), q_pids=gids, g_pids=qids, max_rank=10, get_mAP=True)
            i2t_cmc, i2t_mAP, i2t_mINP = i2t_cmc.numpy(), i2t_mAP.numpy(), i2t_mINP.numpy()
            table.add_row(['i2t', i2t_cmc[0], i2t_cmc[4], i2t_cmc[9], i2t_mAP, i2t_mINP])
            metrics.update({
                'i2t_R1': float(i2t_cmc[0]),
                'i2t_R5': float(i2t_cmc[4]),
                'i2t_R10': float(i2t_cmc[9]),
                'i2t_mAP': float(i2t_mAP),
                'i2t_mINP': float(i2t_mINP),
            })
        # table.float_format = '.4'
        table.custom_format["R1"] = lambda f, v: f"{v:.3f}"
        table.custom_format["R5"] = lambda f, v: f"{v:.3f}"
        table.custom_format["R10"] = lambda f, v: f"{v:.3f}"
        table.custom_format["mAP"] = lambda f, v: f"{v:.3f}"
        table.custom_format["mINP"] = lambda f, v: f"{v:.3f}"
        self.logger.info('\n' + str(table))

        if return_metrics:
            return metrics
        return t2i_cmc[0]
