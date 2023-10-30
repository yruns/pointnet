import torch
from datasets.shapenet import ShapeNetDataset
from models.pointnet import PointNet
from torch.utils.data import DataLoader
from torch.optim import Adam, SGD
from transformers import get_linear_schedule_with_warmup
from args_shapenet import Args4Shapenet
from loguru import logger
from utils.tools import seed_everything

import os

def train():

    # define hyperparameters
    args = Args4Shapenet()
    seed_everything(args.seed)
    logger.add('%s/{time}.log' % args.log_path)

    args.print_args(logger)

    if torch.cuda.is_available():
        device = "cuda:0"
    elif torch.backends.mps.is_built():
        device = "mps"
    else:
        device = "cpu"

    # define dataloader
    batch_size = int(args.batch_size / args.gradient_accumulation_steps)

    train_dataset = ShapeNetDataset(root=args.data_dir, mode="train", points_num=args.points_num, normal_channel=args.normal)
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=train_dataset.collate_fn
    )
    test_dataset = ShapeNetDataset(root=args.data_dir, mode="test", points_num=args.points_num, normal_channel=args.normal)
    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=test_dataset.collate_fn
    )

    # define model
    model = PointNet(k=args.k, mode="seg", device=device).to(device)

    # define criterion
    criterion = torch.nn.CrossEntropyLoss()

    # define optimizer
    if args.optimizer == "Adam":
        optimizer = Adam(
            params=model.parameters(),
            lr=args.lr,
            betas=(0.9, 0.999),
            weight_decay=args.weight_decay,
            eps=args.adam_epsilon
        )
    elif args.optimizer == "SGD":
        optimizer = SGD(
            params=model.parameters(),
            lr=args.lr,
            momentum=0.9,
            weight_decay=args.weight_decay
        )
    else:
        raise ValueError("Optimizer not supported!")
    total_steps = int(len(train_dataloader) * args.epochs / args.gradient_accumulation_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps * args.warmup_rate),
                                                num_training_steps=total_steps)

    # Begin training
    logger.info('\n')
    logger.info("***** Running training *****")
    logger.info("  Device = {}".format(device))
    logger.info("  Num Epochs = {}".format(args.epochs))
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = {}".format(
        args.batch_size))
    logger.info("  Type of optimizer = {}".format(optimizer.__class__.__name__))
    logger.info("  Total optimization steps = {}".format(total_steps))
    logger.info("  Learning rate = {}".format(args.lr))
    logger.info('\n')

    global_step = 0
    best_acc = 0.0
    best_class_avg_iou = 0.0
    best_instance_avg_iou = 0.0

    for epoch in range(args.epochs):
        mean_correct = []
        model.train()
        for step, batch in enumerate(train_dataloader):
            seg_type, inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)

            # forward
            logits = model(inputs)  # [-1, n, k]



if __name__ == '__main__':
    train()

