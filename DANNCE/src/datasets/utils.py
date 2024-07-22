from torchvision import transforms

from .datasets import LazyLoader, ImageSet, Mixed
from DataLoader import OpenBHBDataset
import os


PACS = ['art_painting', 'sketch', 'cartoon', 'photo']
VLCS = ['CALTECH', 'LABELME', 'PASCAL', 'SUN']

OfficeHome = ['art', 'clipart', 'product', 'real_world']
OfficeHome_larger = ['Art', 'Clipart', 'Product', 'Real_World']
DATASETS = {
    'PACS': PACS,
    'VLCS': VLCS,
    'OfficeHome': OfficeHome,
    'OfficeHome_larger': OfficeHome_larger
}
CLASSES = {'PACS': 7, 'VLCS': 5, 'OfficeHome': 65, 'OfficeHome_larger': 65, 'OpenBHB': 1}
DOMAINS = {'PACS': 3, 'VLCS': 3, 'OfficeHome': 3, 'OfficeHome_larger': 3, 'OpenBHB': 70}


def random_color_jitter(magnitude=1):
    assert magnitude > 0
    return transforms.ColorJitter(magnitude * .4, magnitude * .4,
                                  magnitude * .4, min(magnitude * .4, 0.5))


def matsuura_augmentation():
    return transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(.4, .4, .4, .4)
    ])


def normed_tensors():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


def inv_norm():
    return transforms.Normalize(
        mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
        std=[1 / 0.229, 1 / 0.224, 1 / 0.255])


def get_splits(name, leave_out=None, original=False):
    if(name=='OpenBHB'):
        current = os.path.dirname(os.path.realpath('DataLoader.py'))
        root = os.path.dirname(os.path.dirname(current))
        print('current: ', root)
        dataset  = OpenBHBDataset(root_dir=root+'/data')
        train_dataset = dataset.get_subset('train')
        val_dataset = dataset.get_subset('val')
        test_dataset = dataset.get_subset('test')

        return {
            'OpenBHB': {
                'train': train_dataset,
                'val': val_dataset,
                'test': test_dataset
            }
        }, CLASSES[name], DOMAINS[name]
    dataset_names = [d for d in DATASETS[name] if d != leave_out]
    # print('************************   get splits   **************')
    # tmp = {heldout: {tuple(LazyLoader(ImageSet, f'../paths/{name}/{dset}/train.txt' \
    #                 if original else \
    #             f'../paths/{name}/{dset}/test.txt',
    #             parent_dir=f'../data/{name}')
    #         for dset in dataset_names if dset != heldout)} for heldout in dataset_names}
    # print("tmp : ", dataset_names, tmp['art_painting'])
    return {
        heldout: {
            'train':
            LazyLoader(
                Mixed,
                *tuple(
                    LazyLoader(
                        ImageSet,
                        f'../paths/{name}/{dset}/train.txt' \
                            if original else \
                        f'../paths/{name}/{dset}/test.txt',
                        parent_dir=f'../data/{name}')
                    for dset in dataset_names if dset != heldout)),
            'val':
            LazyLoader(
                Mixed,
                *tuple(
                    LazyLoader(ImageSet,
                               f'../paths/{name}/{dset}/val.txt',
                               parent_dir=f'../data/{name}')
                    for dset in dataset_names if dset != heldout)),
            'test':
            LazyLoader(ImageSet,
                       f'../paths/{name}/{heldout}/test.txt',
                       parent_dir=f'../data/{name}')
        }
        for heldout in dataset_names
    }, CLASSES[name], DOMAINS[name]
