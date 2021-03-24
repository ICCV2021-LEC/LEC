from torchvision import transforms
from torch.utils.data import DataLoader

from config import settings
from data.CIFAR100 import NC_CIFAR100
from data.CIFAR100_val import CIFAR100_val
from data.CUB200 import NC_CUB200
from data.CUB200_val import NC_CUB200_val
from data.miniImageNet import NC_miniImageNet
from data.miniImageNet_val import NC_miniImageNet_val


def data_loader(args):

    batch = args.batch_size
    mean_vals = settings.mean_vals
    std_vals = settings.std_vals
    if args.dataset == 'CIFAR100':
        tsfm_train = transforms.Compose([transforms.ToPILImage(),
                             transforms.Resize((32, 32)),
                             transforms.RandomCrop(32, padding=8),
                             transforms.RandomHorizontalFlip(),
                             transforms.ToTensor(),
                             transforms.Normalize(mean_vals, std_vals)
                             ])
    if args.dataset == 'CUB200':
        tsfm_train = transforms.Compose([#transforms.ToPILImage(),
                             transforms.Resize((256, 256)),
                             transforms.RandomCrop(224, padding=0),
                             transforms.RandomHorizontalFlip(),
                             transforms.ToTensor(),
                             transforms.Normalize(mean_vals, std_vals)
                             ])
    if args.dataset == 'miniImageNet':
        tsfm_train = transforms.Compose([#transforms.ToPILImage(),
                                     transforms.Resize((84, 84)),
                                     transforms.RandomCrop(84, padding=8),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ColorJitter(brightness=0.4, contrast=0.4,saturation=0.4),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean_vals, std_vals)
                                     ])
    
    if args.dataset == 'CIFAR100':
        img_train = NC_CIFAR100(args, transform=tsfm_train)
    if args.dataset == 'CUB200':
        img_train = NC_CUB200(args, transform=tsfm_train)
    if args.dataset == 'miniImageNet':
        img_train = NC_miniImageNet(args, transform=tsfm_train)

    train_loader = DataLoader(img_train, batch_size=batch, shuffle=True, num_workers=8)

    return train_loader


def val_loader(args):

    batch = args.batch_size
    mean_vals = settings.mean_vals
    std_vals = settings.std_vals
    if args.dataset == 'CIFAR100':
        size=settings.CIFAR_size
    if args.dataset == 'CUB200':
        size=settings.CUB_size
    if args.dataset == 'miniImageNet':
        size=settings.miniImage_size
    
    if args.dataset == 'CIFAR100':
        tsfm_train = transforms.Compose([transforms.ToPILImage(),
                                         transforms.Resize((size, size)), #224
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean_vals, std_vals)
                                         ])
    else:
        tsfm_train = transforms.Compose([#transforms.ToPILImage(),
                                         transforms.Resize((size, size)), #224
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean_vals, std_vals)
                                         ])
    
    if args.dataset == 'CIFAR100':
        img_val = CIFAR100_val(args, transform=tsfm_train)
    if args.dataset == 'CUB200':
        img_val = NC_CUB200_val(args, transform=tsfm_train)
    if args.dataset == 'miniImageNet':
        img_val = NC_miniImageNet_val(args, transform=tsfm_train)
    
    val_loader = DataLoader(img_val, batch_size=batch, shuffle=False, num_workers=8)

    return val_loader
