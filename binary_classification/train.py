from torchvision.transforms import transforms
from torch.utils.data import DataLoader
import torch.optim as optim
from efficientnet_pytorch import EfficientNet

import argparse
import logging
import shutil
from pytz import timezone
from datetime import datetime
from tensorboardX import SummaryWriter
from tqdm import tqdm

from binary_classification.autoaugment import ImageNetPolicy
from binary_classification.loader import Food_5K_dataset
from binary_classification.model import *
from binary_classification.Measure import AverageMeter


def init_logger(save_dir):
    tz = timezone('Asia/Seoul')
    c_time = datetime.now(tz=tz).strftime("%Y%m%d/%H%M%S")
    log_dir = f'/{save_dir}/{c_time}'
    log_txt = f'/{save_dir}/{c_time}/log.txt'

    global writer
    writer= SummaryWriter(log_dir)

    logging.Formatter.converter = lambda *args: datetime.now(tz=tz).timetuple()
    logging.basicConfig(level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S', format="[%(asctime)s] %(message)s", handlers=[logging.FileHandler(log_txt), ])

    global logger
    logger = logging.getLogger()
    logger.info(f'Log directory ... {log_txt}')

    return log_dir


def train(model, loader, criterion, optimizer, epoch):
    losses = AverageMeter()
    acc = AverageMeter()

    model.train()
    pbar = tqdm(loader, ncols = 150)
    for i, (label, image, _) in enumerate(loader, start=1):
        optimizer.zero_grad()
        outputs = model(image.cuda())

        one_hot_label = torch.nn.functional.one_hot(label.to(torch.int64), 2)
        one_hot_label = one_hot_label.float()

        loss = criterion(outputs, one_hot_label.cuda())
        # loss = criterion(outputs, label.cuda())

        loss.backward()
        optimizer.step()

        prob, indice = torch.topk(outputs.cpu(), k=1)
        acc.update(torch.sum(indice[:, 0:1] == label.reshape(-1, 1)).item())

        losses.update(loss)

        pbar.set_description(f'[Epoch {epoch}] Train '
                             f'loss: {losses.val:.4f}({losses.avg:.4f}) '
                             f'accuracy: {acc.val / loader.batch_size:.4f}({acc.sum / (i * loader.batch_size):.4f}) ')
        pbar.update()

    log = f'[EPOCH {epoch}] Train ' \
          f'loss: {losses.avg:.4f} ' \
          f'accuracy: {acc.sum / loader.dataset.__len__():.4f} '

    pbar.set_description(log)
    pbar.close()
    logger.info(log)

    writer.add_scalar('Train/loss', losses.avg, epoch)
    writer.add_scalar('Train/accuracy', acc.sum / loader.dataset.__len__(), epoch)

    return losses, acc.sum / loader.dataset.__len__()


@torch.no_grad()
def valid(model, loader, criterion, epoch):
    losses = AverageMeter()
    acc = AverageMeter()

    model.eval()
    pbar = tqdm(loader, ncols=150)
    for i, (label, image, _) in enumerate(loader, start=1):
        outputs = model(image.cuda())

        one_hot_label = torch.nn.functional.one_hot(label.to(torch.int64), 2)
        one_hot_label = one_hot_label.float()

        loss = criterion(outputs, one_hot_label.cuda())
        # loss = criterion(outputs, label.cuda())

        prob, indice = torch.topk(outputs.cpu(), k=1)
        acc.update(torch.sum(indice[:, 0:1] == label.reshape(-1, 1)).item())

        losses.update(loss)

        pbar.set_description(f'[Epoch {epoch}] Valid '
                             f'loss: {losses.val:.4f}({losses.avg:.4f}) '
                             f'accuracy: {acc.val / loader.batch_size:.4f}({acc.sum / (i * loader.batch_size):.4f}) ')
        pbar.update()

    log = f'[EPOCH {epoch}] Valid ' \
          f'loss: {losses.avg:.4f} ' \
          f'accuracy: {acc.sum / loader.dataset.__len__():.4f} '

    pbar.set_description(log)
    pbar.close()
    logger.info(log)

    writer.add_scalar('Valid/loss', losses.avg, epoch)
    writer.add_scalar('Valid/accuracy', acc.sum / loader.dataset.__len__(), epoch)

    return losses, acc.sum / loader.dataset.__len__()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train for Food/Non-food Binary Classification")
    parser.add_argument('-s', '--save_dir', type=str, default=f'/hdd/food_classification_output')
    parser.add_argument('-batch', '--batch_size', type=int, default=128)
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-4)
    parser.add_argument('-wd', '--weight_decay', type=float, default=0.)
    parser.add_argument('-e', '--epoch', type=int, default=100)

    args = parser.parse_args()

    log_dir = init_logger(args.save_dir)

    transform = {
        'train':
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                ImageNetPolicy(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
            ]),
        'valid':
            transforms.Compose([
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
            ])
    }

    train_loader = DataLoader(
        Food_5K_dataset('/hdd/Food-5K/training/', transform=transform['train'], phase='train'), batch_size=args.batch_size,
        num_workers=4, shuffle=True)

    valid_loader = DataLoader(
        Food_5K_dataset('/hdd/Food-5K/validation/', transform=transform['valid'], phase='valid'), batch_size=args.batch_size,
        num_workers=4, shuffle=True)

    # TODO
    # model = Resnet50(n_class=2).cuda()
    model = MobileNet(n_class=2).cuda()

    # model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=2)
    # for n, p in model.named_parameters():
    #     if '_fc' not in n:
    #         p.requires_grad = False
    # model = model.cuda()

    writer.add_graph(model, torch.rand((2, 3, 224, 224)).cuda()) # TODO
    model = nn.DataParallel(model)

    # criterion = nn.CrossEntropyLoss()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                           lr=args.learning_rate,
                           weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 30, 50], gamma=0.1)

    best = 0.0
    for ep in range(1, args.epoch, 1):
        train(model, train_loader, criterion, optimizer, ep)
        loss, acc = valid(model, valid_loader, criterion, ep)

        torch.save({'model_state_dict': model.module.state_dict(),
                    'optim_state_dict': optimizer.state_dict(),
                    'epoch': ep, 'acc': acc,
                    'loss': loss.avg}, f'{log_dir}/last.pt')

        if acc > best:
            best = acc
            shutil.copyfile(f'{log_dir}/last.pt', f'{log_dir}/best.pt')