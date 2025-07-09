import os
import random
import subprocess
import threading
from copy import deepcopy
from functools import reduce

import math
import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR, StepLR, ReduceLROnPlateau
from tqdm import tqdm


def calc_accuracy(y_pred, y_true):
    """
    Compute multi-label accuracy based on top-k predictions per sample.

    Args:
        y_pred (torch.Tensor): Model predictions (logits or probabilities) of shape (batch_size, num_classes).
        y_true (torch.Tensor): Ground truth binary labels of shape (batch_size, num_classes).

    Returns:
        tensor: Multi-label accuracy.
    """
    num_labels_per_sample = y_true.sum(dim=1)
    sorted_indices = torch.argsort(y_pred, dim=1, descending=True)

    y_pred_bin = torch.zeros_like(y_pred)

    for i in range(y_pred.size(0)):
        top_k = num_labels_per_sample[i].int()
        y_pred_bin[i, sorted_indices[i, :top_k]] = 1

    acc = (((y_pred_bin + y_true) == 2).sum(1) / num_labels_per_sample).mean()

    return acc


def get_centroids(labels, proxies):
    # centroids = (labels.unsqueeze(2) * proxies.unsqueeze(0)).sum(1) / labels.sum(1).unsqueeze(1)
    centroids = (labels @ proxies) / labels.sum(1).unsqueeze(1)
    return centroids


def rename_output(args):
    if os.path.exists(f"./output/{args.backbone}"):
        m_vals = [int(x.split("-m")[1]) for x in os.listdir("./output") if len(x.split("-m")) == 2]
        m_next = (max(m_vals) if len(m_vals) > 0 else 0) + 1
        print(f"rename previous output {args.backbone} -> {args.backbone}-m{m_next}")
        os.rename(f"./output/{args.backbone}", f"./output/{args.backbone}-m{m_next}")


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_checkpoint(args, best_checkpoint):
    if best_checkpoint is None:
        logger.warning(f"no further improvement")
        return
    if "memo" in best_checkpoint:
        # support utils_cm!
        memo = best_checkpoint.pop("memo")
        if "epoch" in best_checkpoint.keys():
            torch.save(best_checkpoint, f"{args.save_dir}/e{best_checkpoint['epoch']}_{memo}.pth")
        else:
            torch.save(best_checkpoint, f"{args.save_dir}/{memo}.pth")
    else:
        torch.save(best_checkpoint, f"{args.save_dir}/e{best_checkpoint['epoch']}_{best_checkpoint['map']:.3f}.pth")


def load_checkpoint(args):
    pkl_list = [x for x in os.listdir(args.save_dir) if x.endswith(".pth")]
    if len(pkl_list) == 0:
        logger.warning(f"no checkpoint found")
        return None
    pkl_list.sort(key=lambda x: int(x.split("_")[0].replace("e", "")), reverse=True)
    check_point = torch.load(args.save_dir + "/" + pkl_list[0])
    return check_point


def load_selected_states(model, state_dict, startswith_filter=None, verbose=True):
    if startswith_filter is None:
        new_checkpoint = state_dict
    else:
        new_checkpoint = {}
        for k, v in state_dict.items():
            if any(k.startswith(prefix) for prefix in startswith_filter):
                new_checkpoint[k] = v

    missing, unexpected = model.load_state_dict(new_checkpoint, strict=False)
    if unexpected:
        raise RuntimeError("Unexpected state dict keys: {}.".format(unexpected))
    if verbose:
        print(missing if missing else "<All keys matched successfully>")

    return model


def validate_clear():
    global validation_thread
    if "validation_thread" in globals():
        del validation_thread


def validate_smart(args, query_loader, dbase_loader, early_stopping, epoch, **kwargs):
    global validation_thread

    if "validation_thread" in globals():
        if validation_thread.is_alive():
            print("thread is still running, waiting for it to finish...")
            validation_thread.join()
        if early_stopping.early_stop:
            validate_clear()
            # no need for further validation
            return True

    val_fnc = kwargs.pop("val_fnc", validate)
    parallel_val = kwargs.pop("parallel_val", True)

    if parallel_val and (epoch + 1) != args.n_epochs and early_stopping.counter < early_stopping.patience - 1:
        available_gpu = next((x for x in get_gpu_info() if x["index"] != args.device.split(":")[1]), None)
        if available_gpu:
            print(f'using gpu:{available_gpu["index"]} ({available_gpu["gpu_util"]}%) for evaluation with threading...')

            for k, v in kwargs.items():
                if "model" in k:
                    # validation_model.load_state_dict(kwargs["model"].state_dict())
                    kwargs[k] = deepcopy(v).to(torch.device(f"cuda:{available_gpu['index']}"))
            kwargs["verbose"] = False

            # start validation in a separate thread
            validation_thread = threading.Thread(
                target=val_fnc, args=(args, query_loader, dbase_loader, early_stopping, epoch), kwargs=kwargs
            )
            validation_thread.start()
            return False

    # perform validation without threading
    early_stop = val_fnc(args, query_loader, dbase_loader, early_stopping, epoch, **kwargs)
    if early_stop or (epoch + 1) == args.n_epochs:
        validate_clear()

    return early_stop


def validate(args, query_loader, dbase_loader, early_stopping, epoch, **kwargs):
    out_idx = kwargs.pop("out_idx", None)
    verbose = kwargs.pop("verbose", True)
    map_fnc = kwargs.pop("map_fnc", mean_average_precision)

    qB, qL = predict(kwargs["model"], query_loader, out_idx=out_idx, verbose=verbose)
    rB, rL = predict(kwargs["model"], dbase_loader, out_idx=out_idx, verbose=verbose)
    map_v = map_fnc(qB, rB, qL, rL, args.topk)
    map_k = "" if args.topk is None else f"@{args.topk}"

    del qB, rB, qL, rL
    torch.cuda.empty_cache()

    map_o = early_stopping.best_map
    early_stopping(epoch, map_v.item(), **kwargs)
    logger.info(
        f"[Evaluating][dataset:{args.dataset}][bits:{args.n_bits}][epoch:{epoch}/{args.n_epochs - 1}][best-mAP{map_k}:{map_o:.4f}][mAP{map_k}:{map_v:.4f}][count:{early_stopping.counter}]"
    )
    return early_stopping.early_stop


class EarlyStopping:
    def __init__(self, patience=10, best_epoch=0, best_map=0.0):
        self.patience = patience
        self.best_epoch = best_epoch
        self.best_map = best_map
        self.best_checkpoint = None
        self.early_stop = False
        self.counter = 0

    def reset(self):
        self.patience = 10
        self.best_epoch = 0
        self.best_map = 0.0
        self.best_checkpoint = None
        self.early_stop = False
        self.counter = 0

    def __call__(self, current_epoch, current_map, **kwargs):
        if current_map > self.best_map:
            self.best_epoch, self.best_map = current_epoch, current_map
            self.counter = 0
            self.best_checkpoint = {
                "epoch": self.best_epoch,
                "map": self.best_map,
            }
            memo = kwargs.pop("memo", None)
            if memo:
                self.best_checkpoint["memo"] = memo
            # for k, v in kwargs.items():
            #     self.best_checkpoint[k] = deepcopy(v.state_dict())
            self.best_checkpoint.update({k: deepcopy(v.state_dict()) for k, v in kwargs.items()})
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


def print_in_md(rst):
    """
    print training result in Markdown format, like below:
    | K\D |  cifar   |  flickr  | nuswide  |   coco   |
    |----:|:--------:|:--------:|:--------:|:--------:|
    |  16 |    -     | 0.222@09 | 0.333@99 | 0.444@99 |
    |  32 | 0.111@99 | 0.222@99 | 0.333@09 |    -     |
    |  64 | 0.111@99 | 0.222@99 | 0.333@99 | 0.444@99 |
    | 128 |    -     |    -     | 0.333@99 | 0.444@99 |
    """
    if len(rst) == 0:
        return
    # datasets = sorted(set([x["dataset"] for x in rst]), key=lambda x: get_class_num(x))
    datasets = sorted(
        set([x["dataset"] for x in rst]),
        key=lambda x: {"cifar": 1, "flickr": 2, "nuswide": 3, "coco": 4, "veri": 5, "vehicleid": 6}.get(x, 9),
    )
    hash_bits = sorted(set([x["hash_bit"] for x in rst]))
    best_epochs = sorted(set([x["best_epoch"] for x in rst]))

    max_len_e = len(str(best_epochs[-1]))
    max_len_k = max(len(str(hash_bits[-1])), 3)  # len("K\D") -> 3

    head = f"| {'K'+chr(92)+'D':^{max_len_k}} |"
    sept = f"|{':':->{max_len_k+2}}|"

    for dataset in datasets:
        head += f"{dataset:^{max_len_e+8}}|"  # len(" 0.xxx@ ") -> 8
        sept += f":{'-':-^{max_len_e+6}}:|"

    rows = []
    for hash_bit in hash_bits:
        temp_str = f"| {hash_bit:>{max_len_k}} |"
        for dataset in datasets:
            ret = [x for x in rst if x["dataset"] == dataset and x["hash_bit"] == hash_bit]
            if len(ret) == 1:
                temp_str += f" {ret[0]['best_map']:.3f}@{ret[0]['best_epoch']:0>{max_len_e}} |"
            else:
                temp_str += f"{'-':^{max_len_e+8}}|"
        rows.append(temp_str)

    print(head)
    print(sept)
    for row in rows:
        print(row)


def gen_test_data(B, C, K, is_multi_hot=False, normalize_embeddings=True, device="cpu"):
    """
    Args:
        B: batch size
        C: number of classes
        K: dim of embeddings
        is_multi_hot: is multi-label dataset or not
        normalize_embeddings: normalize embeddings or not
        device: cpu or gpu
    Returns:
        embeddings: [B, K]
        singles: [B, ],  categorical ids, None if is_multi_hot
        onehots: [B, C], onehot encoded categorical ids
    """
    embeddings = torch.randn(B, K, requires_grad=True).to(device)
    if normalize_embeddings:
        embeddings = F.normalize(embeddings, p=2, dim=-1)
    if is_multi_hot:
        singles = None
        onehots = (torch.randn(B, C) > 0.8).float().to(device)
    else:
        singles = torch.randint(low=0, high=C, size=[B]).to(device)  # categorical id
        onehots = F.one_hot(singles, C).float().to(device)
    return embeddings, singles, onehots


def build_optimizer(optim_type, parameters, **kwargs):
    # optimizer_names = ["Adam", "RMSprop", "SGD"]
    # optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)
    if optim_type == "sgd":
        optimizer = optim.SGD(parameters, **kwargs)
    elif optim_type == "rmsprop":
        optimizer = optim.RMSprop(parameters, **kwargs)
    elif optim_type == "adam":
        optimizer = optim.Adam(parameters, **kwargs)
    elif optim_type == "amsgrad":
        optimizer = optim.Adam(parameters, amsgrad=True, **kwargs)
    elif optim_type == "adamw":
        optimizer = optim.AdamW(parameters, **kwargs)
    else:
        raise NotImplementedError(f"not support optimizer: {optim_type}")
    return optimizer


def build_scheduler(scheduler_type, optimizer, **kwargs):
    if scheduler_type == "none":
        scheduler = None
    elif scheduler_type == "step":
        if "milestones" in kwargs:
            scheduler = MultiStepLR(optimizer, **kwargs)
        else:
            scheduler = StepLR(optimizer, **kwargs)
    elif scheduler_type == "cosine":
        scheduler = CosineAnnealingLR(optimizer, **kwargs)
    elif scheduler_type == "reduce":
        scheduler = ReduceLROnPlateau(optimizer, **kwargs)
    else:
        raise NotImplementedError(f"not support scheduler: {scheduler_type}")
    return scheduler


def get_gpu_info():
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,utilization.gpu,utilization.memory,memory.total,memory.used,memory.free",
        "--format=csv,noheader,nounits",
    ]

    try:
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"{e.stderr.strip() or e.stdout.strip()}")
        return None
    except FileNotFoundError:
        print("Error: nvidia-smi not found. Ensure NVIDIA drivers and CUDA are installed correctly.")
        return None

    gpu_info = result.stdout.strip().split("\n")

    info_keys = ["index", "name", "gpu_util", "mem_util", "mem_total", "mem_used", "mem_free"]

    return [dict(zip(info_keys, info.split(", "))) for info in gpu_info]


def init():
    assert torch.cuda.is_available(), "CUDA is not available"
    # batch_size is too small will cause "Too many open files..."
    torch.multiprocessing.set_sharing_strategy("file_system")


def find_diff_same(t1, t2, dim):
    """
    Args:
        t1 = torch.tensor([1, 9, 12, 5, 24])
        t2 = torch.tensor([1, 24])
    Returns:
        diff: torch.tensor([5, 9, 12])
        same: torch.tensor([1, 24])
    From:
        https://stackoverflow.com/questions/55110047
    """
    t1 = torch.unique(t1, dim=dim)
    t2 = torch.unique(t2, dim=dim)
    combined = torch.cat((t1, t2))
    uniques, counts = combined.unique(dim=dim, return_counts=True)
    difference = uniques[counts == 1]
    intersection = uniques[counts > 1]
    return difference, intersection


def calc_learnable_params(*args):
    n_parameters = 0
    for x in args:
        if x:
            n_parameters += sum(p.numel() for p in x.parameters() if p.requires_grad)
    return n_parameters


def sizeof_fmt(num, suffix="B"):
    for unit in ("", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"):
        if abs(num) < 1024.0:
            return f"{num:3.1f}{unit}{suffix}"
        num /= 1024.0
    return f"{num:.1f}Yi{suffix}"


def hash_center_type(n_classes, n_bits):
    """
    used in CenterHashing, CSQ, ...
    """
    lg2 = 0 if n_bits < 1 else int(math.log(n_bits, 2))
    if 2**lg2 != n_bits:
        return "random"

    if n_classes <= n_bits:
        return "ha_d"
    elif n_classes > n_bits and n_classes <= 2 * n_bits:
        return "ha_2d"
    else:
        return "random"


def predict(net, dataloader, out_idx=None, use_sign=True, verbose=True):
    device = next(net.parameters()).device

    codes, clses = [], []
    net.eval()

    # logger.info(f"predicting({len(dataloader.dataset)})...")
    if verbose:
        try:
            _iter = tqdm(dataloader, desc=f"extracting {dataloader.dataset.usage} features")
        except AttributeError:
            _iter = tqdm(dataloader, desc=f"extracting features")
    else:
        _iter = dataloader

    # for img, cls, _ in tqdm(dataloader):
    # for img, tag, cls, _ in tqdm(dataloader): # CrossModal
    for x in _iter:
        with torch.no_grad():
            out = net(x[0].to(device))
        if out_idx is None:
            rst = out
        elif isinstance(out_idx, list):
            rst = reduce(lambda d, key: d[key], out_idx, out)
        else:
            rst = out[out_idx]
        codes.append(rst)
        clses.append(x[-2])
    return torch.cat(codes).sign() if use_sign else torch.cat(codes), torch.cat(clses).to(device)


def calc_map(y_pred, y_true):
    """
    Compute mean average precision (mAP) for multi-label classification.
    Adapted from ASL.
    Args:
        y_pred (torch.Tensor): Model predictions (logits or probabilities) of shape (batch_size, num_classes).
        y_true (torch.Tensor): Ground truth binary labels of shape (batch_size, num_classes).
    Returns:
        mAP (torch.Tensor): Mean average precision.
    """
    N = y_true.shape[1]

    ap_sum = 0.0

    # Compute average precision for each class
    for i in range(N):

        # Arrange position according to probabilities
        topk_indices = torch.argsort(y_pred[:, i], descending=True)
        retrieval = y_true[topk_indices, i]

        # Get rank (1-based index) in all samples
        rank_in_all = retrieval.nonzero(as_tuple=True)[0] + 1
        if rank_in_all.numel() == 0:
            # Can not retrieve images
            continue

        # Generate rank (1-based index) in positive samples
        rank_in_pos = torch.arange(1, rank_in_all.numel() + 1, device=rank_in_all.device)

        ap_sum += (rank_in_pos / rank_in_all).mean()

    return ap_sum / N


def calc_map_eval(codes, labels):
    N, K = codes.shape

    # Compute hamming distance
    hamming_dists = 0.5 * (K - codes @ codes.T)
    hamming_dists.fill_diagonal_(torch.finfo(hamming_dists.dtype).max)

    # Get top-k nearest neighbors by hamming distance
    _, topk_indices = torch.topk(hamming_dists, N, largest=False)

    # Compute retrieval relevance matrix
    if labels.ndim == 1:
        masks = labels.unsqueeze(1) == labels.unsqueeze(0)
    elif labels.ndim == 2:
        masks = labels @ labels.T > 0
    else:
        raise NotImplementedError(f"not support: {labels.shape}")
    masks.fill_diagonal_(False)
    masks = masks.gather(1, topk_indices)

    ranks_in_pos_wo_mask = torch.cumsum(masks, dim=1)
    ranks_in_all_wo_mask = torch.arange(1, masks.shape[1] + 1, device=masks.device)

    ap = (ranks_in_pos_wo_mask / ranks_in_all_wo_mask) * masks
    nonzero = masks.sum(1) > 0

    return (ap.sum(1)[nonzero] / masks.sum(1)[nonzero]).sum() / N


def mean_average_precision(qB, rB, qL, rL, topk=None):
    """
    Calculate mean average precision (mAP).

    Args:
        qB (torch.Tensor): Query data hash code.
        rB (torch.Tensor): Database data hash code.
        qL (torch.Tensor): Query data labels, single or one-hot.
        rL (torch.Tensor): Database data labels, single or one-hot.
        topk (Any): Calculate top k data mAP.

    Returns:
        mAP (torch.Tensor): Mean Average Precision.
    """
    N, K = qB.shape

    if topk is None:
        topk = rL.shape[0]

    # Retrieve images from database
    if qL.ndim == 1:
        masks = qL.unsqueeze(1) == rL.unsqueeze(0)
    elif qL.ndim == 2:
        masks = qL @ rL.T > 0
    else:
        raise NotImplementedError(f"not support: {qL.shape}")

    # Compute hamming distance
    hamming_dists = 0.5 * (K - qB @ rB.T)

    ap_sum = 0.0
    for i in range(N):
        # Calculate hamming distance
        hamming_dist = hamming_dists[i]

        # Arrange position according to hamming distance
        topk_indices = torch.topk(hamming_dist, topk, dim=0, largest=False).indices
        retrieval = masks[i, topk_indices]

        # Get rank (1-based index) in all samples
        rank_in_all = retrieval.nonzero(as_tuple=True)[0] + 1
        if rank_in_all.numel() == 0:
            # Can not retrieve images
            continue

        # Generate rank (1-based index) in positive samples
        rank_in_pos = torch.arange(1, rank_in_all.numel() + 1, device=rank_in_all.device)  # way2

        ap_sum += (rank_in_pos / rank_in_all).mean()

    return ap_sum / N


def calc_hamming_dist(B1, B2):
    """
    calc Hamming distance
    Args:
        B1 (torch.Tensor): each bit of B1 ∈ {-1, 1}^k
        B2 (torch.Tensor): each bit of B2 ∈ {-1, 1}^k
    Returns:
        Hamming distance ∈ {0, k}
    """
    k = B2.shape[1]
    if len(B1.shape) < 2:
        B1 = B1.unsqueeze(0)
    distH = 0.5 * (k - B1.mm(B2.t()))
    return distH


def pr_curve(qB, rB, qL, rL):
    num_query = qB.shape[0]
    num_bit = qB.shape[1]

    hamming_dists = calc_hamming_dist(qB, rB)

    P = torch.zeros(num_query, num_bit + 1)
    R = torch.zeros(num_query, num_bit + 1)
    for i in range(num_query):
        gnd = (qL[i].unsqueeze(0).mm(rL.t()) > 0).float().squeeze()
        tsum = torch.sum(gnd)
        if tsum == 0:
            continue
        hamm = hamming_dists[i]
        tmp = (hamm <= torch.arange(0, num_bit + 1).reshape(-1, 1).float().to(hamm.device)).float()
        total = tmp.sum(dim=-1)
        total = total + (total == 0).float() * 0.1
        t = gnd * tmp
        count = t.sum(dim=-1)
        p = count / total
        r = count / tsum
        P[i] = p
        R[i] = r
    mask = (P > 0).float().sum(dim=0)
    mask = mask + (mask == 0).float() * 0.1
    P = P.sum(dim=0) / mask
    R = R.sum(dim=0) / mask
    return P, R


def p_topK(qB, rB, qL, rL, K=None):
    if K is None:
        K = [1, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    num_query = qL.shape[0]
    p = [0] * len(K)
    for iter in range(num_query):
        gnd = (qL[iter].unsqueeze(0).mm(rL.t()) > 0).float().squeeze()
        tsum = torch.sum(gnd)
        if tsum == 0:
            continue
        hamm = calc_hamming_dist(qB[iter, :], rB).squeeze()
        for i in range(len(K)):
            total = min(K[i], rL.shape[0])
            ind = torch.sort(hamm)[1][:total]
            gnd_ = gnd[ind]
            p[i] += gnd_.sum() / total
    p = torch.Tensor(p) / num_query
    return p


def cos(A, B=None):
    """cosine"""
    # An = normalize(A, norm='l2', axis=1)
    An = A / np.linalg.norm(A, ord=2, axis=1)[:, np.newaxis]
    if (B is None) or (B is A):
        return np.dot(An, An.T)
    # Bn = normalize(B, norm='l2', axis=1)
    Bn = B / np.linalg.norm(B, ord=2, axis=1)[:, np.newaxis]
    return np.dot(An, Bn.T)


def hamming(A, B=None):
    """A, B: [None, bit]
    elements in {-1, 1}
    """
    if B is None:
        B = A
    bit = A.shape[1]
    return (bit - A.dot(B.T)) // 2


def euclidean(A, B=None, sqrt=False):
    aTb = np.dot(A, B.T)
    if (B is None) or (B is A):
        aTa = np.diag(aTb)
        bTb = aTa
    else:
        aTa = np.diag(np.dot(A, A.T))
        bTb = np.diag(np.dot(B, B.T))
    D = aTa[:, np.newaxis] - 2.0 * aTb + bTb[np.newaxis, :]
    if sqrt:
        D = np.sqrt(D)
    return D


def NDCG(qF, rF, qL, rL, what=0, k=-1):
    """Normalized Discounted Cumulative Gain
    ref: https://github.com/kunhe/TALR/blob/master/%2Beval/NDCG.m
    """
    n_query = qF.shape[0]
    if (k < 0) or (k > rF.shape[0]):
        k = rF.shape[0]
    Rel = np.dot(qL, rL.T).astype(int)
    G = 2**Rel - 1
    D = np.log2(2 + np.arange(k))
    if what == 0:
        Rank = np.argsort(1 - cos(qF, rF))
    elif what == 1:
        Rank = np.argsort(hamming(qF, rF))
    elif what == 2:
        Rank = np.argsort(euclidean(qF, rF))

    _NDCG = 0
    for g, rnk in zip(G, Rank):
        dcg_best = (np.sort(g)[::-1][:k] / D).sum()
        if dcg_best > 0:
            dcg = (g[rnk[:k]] / D).sum()
            _NDCG += dcg / dcg_best
    return _NDCG / n_query


def get_precision_recall_by_Hamming_Radius(database_output, database_labels, query_output, query_labels, radius=2):
    bit_n = query_output.shape[1]
    ips = np.dot(query_output, database_output.T)
    ips = (bit_n - ips) / 2

    precX = []
    for i in range(ips.shape[0]):
        label = query_labels[i, :]
        label[label == 0] = -1
        idx = np.reshape(np.argwhere(ips[i, :] <= radius), (-1))
        all_num = len(idx)
        if all_num != 0:
            imatch = np.sum(database_labels[idx[:], :] == label, 1) > 0
            match_num = np.sum(imatch)
            precX.append(match_num / all_num)
        else:
            precX.append(0.0)
    return np.mean(np.array(precX))


if __name__ == "__main__":
    pass
