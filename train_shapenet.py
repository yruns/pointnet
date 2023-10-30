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
            logits = logits.view(-1, args.k)  # [-1, k]
            targets = targets.view(-1)  # [-1]

            loss = criterion(logits, targets)

            # backward
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()
            global_step += 1

            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            if step % args.log_interval == 0 and step > 0:
                logger.info("Epoch: {:2d} | Step: {:4d} | Loss: {:.4f}".format(epoch, step, loss.item()))

            if global_step % args.eval_interval == 0 and global_step > 0:
                logger.info("***** Running evaluation *****")
                acc = evaluate(model, args, test_dataloader, device)
                model.train()

                if acc > best_acc:
                    output_dir = os.path.join(args.checkpoint_path, 'pointnet-{}-{}'.format(epoch, step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    logger.info("Saving model checkpoint to %s" % output_dir)
                    torch.save(model.state_dict(), os.path.join(output_dir, 'model.pt'))
                    args.save_settings(output_dir)
                    best_acc = acc

                logger.info("Epoch: {:2d} | Step: {:4d} | Test Acc: {:.4f}\n".format(epoch, step, acc))
                logger.info("Current best acc: {:.4f}".format(best_acc))


def evaluate(model, args, test_data_loader, device):
    model.eval()
    total_correct = 0
    total_seen = 0
    total_class_iou = 0
    total_instance_iou = 0

    with torch.no_grad():
        for batch in test_data_loader:
            seg_type, inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)

            logits = model(inputs)
            logits = logits.view(-1, args.k)
            targets = targets.view(-1)

            pred = logits.max(dim=1)[1]
            correct = pred.eq(targets).sum().item()
            total_correct += correct
            total_seen += len(targets)

    return total_correct / total_seen


if __name__ == '__main__':
    train()

