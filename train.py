import json
import os
import pickle
import time

import numpy as np
import torch
from loguru import logger
from timm.utils import AverageMeter

from _data import build_loader, build_loaders, get_topk, get_class_num
from _network import build_model
from _utils import (
    build_optimizer,
    build_scheduler,
    calc_learnable_params,
    EarlyStopping,
    init,
    mean_average_precision,
    print_in_md,
    save_checkpoint,
    seed_everything,
    validate_smart,
    predict,
)
from config import get_config
from loss import TripletLoss
from xbm import XBM


def train_epoch(args, dataloader, net, criterion, optimizer, scheduler, xbm, epoch, drift):
    tic = time.time()

    stat_meters = {}
    for x in [
        "n_triplets(batch)",
        "n_triplets(xbm)",
        "hardness(batch)",
        "hardness(xbm)",
        "batch-loss",
        "xbm-loss",
        "loss",
        "mAP",
    ]:
        stat_meters[x] = AverageMeter()

    net.train()
    for images, labels, _ in dataloader:
        images, labels = images.to(args.device), labels.to(args.device)
        embeddings = net(images)

        if epoch + 1 > args.xbm_warmup:
            xbm.set(embeddings.detach(), labels)

        loss, n_triplets, hardness = criterion(embeddings, labels, embeddings, labels)
        stat_meters["n_triplets(batch)"].update(n_triplets)
        stat_meters["hardness(batch)"].update(hardness)
        if n_triplets > 0:
            stat_meters["batch-loss"].update(loss)

        if epoch + 1 > args.xbm_warmup:
            xbm_feats, xbm_targets = xbm.get()
            # get current batch begin pointer
            # ptr = xbm.ptr - labels.shape[0]
            xbm_loss, n_triplets, hardness = criterion(embeddings, labels, xbm_feats, xbm_targets)
            stat_meters["n_triplets(xbm)"].update(n_triplets)
            stat_meters["hardness(xbm)"].update(hardness)
            if n_triplets > 0:
                stat_meters["xbm-loss"].update(xbm_loss)
            loss = loss + args.xbm_weight * xbm_loss

        if not isinstance(loss, torch.Tensor):
            print("not enough triplets!")
            continue

        stat_meters["loss"].update(loss)

        # print("iter", drift["iter"])
        for x in [10, 100, 1000]:
            if drift["iter"] % x == 0:
                qB, _ = predict(net, drift["dataloader"], use_sign=False, verbose=False)
                new = qB.cpu().numpy()
                if drift[f"emb-{x}"] is None:
                    drift[f"emb-{x}"] = new
                    print(f"[iter {drift['iter']}] just save emb-{x}")
                else:
                    drift[f"rst-{x}"].append(np.mean(np.sum((drift[f"emb-{x}"] - new) ** 2, axis=1)))
                    drift[f"emb-{x}"] = new
                    print(f"[iter {drift['iter']}] calc avg dist & save rst-{x}")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        drift["iter"] += 1

        # to check overfitting
        q_cnt = labels.shape[0] // 10
        map_v = mean_average_precision(
            embeddings[:q_cnt].sign(), embeddings[q_cnt:].sign(), labels[:q_cnt], labels[q_cnt:]
        )
        stat_meters["mAP"].update(map_v)

        torch.cuda.empty_cache()

    if scheduler is None:
        last_lr = None
    else:
        last_lr = scheduler.get_last_lr()
        scheduler.step()

    toc = time.time()
    lr_str = "[lr:{}]".format("/".join("{:.8f}".format(x) for x in last_lr)) if last_lr else ""
    sm_str = ""
    for x in stat_meters.keys():
        sm_str += f"[{x}:{stat_meters[x].avg:.1f}]" if "n_" in x else f"[{x}:{stat_meters[x].avg:.4f}]"
    logger.info(
        f"[Training][dataset:{args.dataset}][bits:{args.n_bits}][epoch:{epoch}/{args.n_epochs - 1}][time:{(toc - tic):.3f}]{lr_str}{sm_str}"
    )


def train_init(args):
    # setup net
    net = build_model(args, True)

    # setup criterion
    criterion = TripletLoss(args.margin, args.hardness, "normalize" not in args.backbone)

    logger.info(f"number of learnable params: {calc_learnable_params(net)}")

    # setup optimizer
    # {"params": list(set(net.parameters()).difference(set(net.fc.parameters()))), "lr": args.lr},
    # {"params": net.fc.parameters(), "lr": args.lr_fc},
    optimizer = build_optimizer(args.optimizer, net.parameters(), lr=args.lr, weight_decay=args.wd)

    if args.scheduler == "cosine":
        scheduler = build_scheduler(args.scheduler, optimizer, T_max=64, eta_min=1e-5)
    elif args.scheduler == "step":
        scheduler = build_scheduler(args.scheduler, optimizer, milestones=[10], gamma=0.2)
    else:
        scheduler = build_scheduler(args.scheduler, optimizer)

    return net, criterion, optimizer, scheduler


def train(args, train_loader, query_loader, dbase_loader):
    net, criterion, optimizer, scheduler = train_init(args)

    xbm = XBM(args)

    early_stopping = EarlyStopping(args.patience)

    # paper Fig. 3
    drift_loader = build_loader(
        args.data_dir,
        args.dataset,
        "query",
        None,
        batch_size=args.batch_size,
        num_workers=args.n_workers,
    )
    drift_loader.dataset.data = train_loader.dataset.data[:1000]
    drift = {
        "iter": 0,
        "emb-10": None,
        "emb-100": None,
        "emb-1000": None,
        "rst-10": [],
        "rst-100": [],
        "rst-1000": [],
        "dataloader": drift_loader,
    }

    for epoch in range(args.n_epochs):
        train_epoch(args, train_loader, net, criterion, optimizer, scheduler, xbm, epoch, drift)

        # we monitor mAP@topk validation accuracy every 5 epochs
        if (epoch + 1) % 5 == 0 or (epoch + 1) == args.n_epochs:
            early_stop = validate_smart(
                args,
                query_loader,
                dbase_loader,
                early_stopping,
                epoch,
                model=net,
                parallel_val=args.parallel_val,
            )
            if early_stop:
                break

    if early_stopping.counter == early_stopping.patience:
        logger.info(
            f"without improvement, will save & exit, best mAP: {early_stopping.best_map:.3f}, best epoch: {early_stopping.best_epoch}"
        )
    else:
        logger.info(
            f"reach epoch limit, will save & exit, best mAP: {early_stopping.best_map:.3f}, best epoch: {early_stopping.best_epoch}"
        )

    save_checkpoint(args, early_stopping.best_checkpoint)

    for x in ["iter", "emb-10", "emb-100", "emb-1000", "dataloader"]:
        del drift[x]
    with open(f"./drift_{args.dataset}_{args.n_bits}.pkl", "wb") as f:
        pickle.dump(drift, f)

    return early_stopping.best_epoch, early_stopping.best_map


def main():
    init()
    args = get_config()

    dummy_logger_id = None
    rst = []
    # for dataset in ["cifar", "nuswide", "flickr", "coco"]:
    for dataset in ["flickr"]:
        print(f"processing dataset: {dataset}")
        args.dataset = dataset
        args.n_classes = get_class_num(dataset)
        args.topk = get_topk(dataset)

        train_loader, query_loader, dbase_loader = build_loaders(
            dataset, args.data_dir, batch_size=args.batch_size, num_workers=args.n_workers
        )
        args.n_samples = train_loader.dataset.__len__()

        # for hash_bit in [16, 32, 64, 128]:
        for hash_bit in [32]:
            print(f"processing hash-bit: {hash_bit}")
            seed_everything()
            args.n_bits = hash_bit

            args.save_dir = f"./output/{args.backbone}/{dataset}/{hash_bit}"
            os.makedirs(args.save_dir, exist_ok=True)
            if any(x.endswith(".pth") for x in os.listdir(args.save_dir)):
                print(f"*.pth exists in {args.save_dir}, will pass")
                continue

            if dummy_logger_id is not None:
                logger.remove(dummy_logger_id)
            dummy_logger_id = logger.add(f"{args.save_dir}/train.log", mode="w", level="INFO")

            with open(f"{args.save_dir}/config.json", "w") as f:
                json.dump(
                    vars(args),
                    f,
                    indent=4,
                    sort_keys=True,
                    default=lambda o: o if type(o) in [bool, int, float, str] else str(type(o)),
                )

            best_epoch, best_map = train(args, train_loader, query_loader, dbase_loader)
            rst.append({"dataset": dataset, "hash_bit": hash_bit, "best_epoch": best_epoch, "best_map": best_map})

    print_in_md(rst)


if __name__ == "__main__":
    main()
