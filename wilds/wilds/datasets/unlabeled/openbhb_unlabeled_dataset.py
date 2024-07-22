from pathlib import Path
import shutil
import pandas as pd
import torch
from torch.utils.data import Dataset
import pickle
import numpy as np
import torchvision.transforms.functional as F
from torchvision import transforms
import tarfile
import datetime
import pytz
from PIL import Image
from tqdm import tqdm
from wilds.datasets.unlabeled.wilds_unlabeled_dataset import WILDSUnlabeledDataset
import os

Image.MAX_IMAGE_PIXELS = 10000000000

from wilds.datasets.openbhb_dataset import (
        DATASET,
        _SPLIT_DATA_40,
        _SPLIT_DATA_60,
        _SPLIT_DATA_40_TOTAL,
        _SPLIT_DATA_60_TOTAL,
        _SPLIT_DATA_40_UNLABELED,
        DHS_SITES,
        _MEANS_2009_17,
        _STD_DEVS_2009_17,
        split_by_site
        )


_SPLIT_DATA = _SPLIT_DATA_40_UNLABELED
class OpenBHBUnlabeledDataset(WILDSUnlabeledDataset):
    """
    The unlabeled PovertyMap-WILDS poverty measure prediction dataset.
    This is a processed version of LandSat 5/7/8 satellite imagery originally from Google Earth Engine under the names `LANDSAT/LC08/C01/T1_SR`,`LANDSAT/LE07/C01/T1_SR`,`LANDSAT/LT05/C01/T1_SR`,
    nighttime light imagery from the DMSP and VIIRS satellites (Google Earth Engine names `NOAA/DMSP-OLS/CALIBRATED_LIGHTS_V4` and `NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG`)
    and processed DHS survey metadata obtained from https://github.com/sustainlab-group/africa_poverty and originally from `https://dhsprogram.com/data/available-datasets.cfm`.
    Unlabeled data are sampled from around DHS survey locations.

    Supported `split_scheme`:
        'official' and `countries`, which are equivalent

    Input (x):
        224 x 224 x 8 satellite image, with 7 channels from LandSat and 1 nighttime light channel from DMSP/VIIRS. Already mean/std normalized.

    Output (y):
        y is a real-valued asset wealth index. Higher index corresponds to more asset wealth.

    Metadata:
        each image is annotated with location coordinates (noised for anonymity), survey year, urban/rural classification, country, nighttime light mean, nighttime light median.

    Website: https://github.com/sustainlab-group/africa_poverty

    Original publication:
    @article{yeh2020using,
        author = {Yeh, Christopher and Perez, Anthony and Driscoll, Anne and Azzari, George and Tang, Zhongyi and Lobell, David and Ermon, Stefano and Burke, Marshall},
        day = {22},
        doi = {10.1038/s41467-020-16185-w},
        issn = {2041-1723},
        journal = {Nature Communications},
        month = {5},
        number = {1},
        title = {{Using publicly available satellite imagery and deep learning to understand economic well-being in Africa}},
        url = {https://www.nature.com/articles/s41467-020-16185-w},
        volume = {11},
        year = {2020}
    }

    License:
        LandSat/DMSP/VIIRS data is U.S. Public Domain.

    """
    _dataset_name = 'openBHB'
    _versions_dict = {
        '1.1': {
            'download_url': 'https://worksheets.codalab.org/rest/bundles/0xdfcf71b4f6164cc1a7edb0cbb7444c8c/contents/blob/',
            'compressed_size': 13_091_823_616,
        }
    }

    def __init__(self, version=None, root_dir='data', download=False,
                 split_scheme='official',
                 no_nl=False, fold='A',
                 use_ood_val=True,
                 cache_size=100):
        self._version = version
        self._data_dir = self.initialize_data_dir(root_dir, download, None)
        # self._original_resolution = (224, 224)
        print(self._data_dir)

        self._split_dict = {'train_unlabeled': 10, 'val_unlabeled': 11, 'test_unlabeled': 12}
        self._split_names = {'train_unlabeled': 'Unlabeled Train', 'val_unlabeled': 'Unlabeled Val', 'test_unlabeled': 'Unlabeled Test'}

        if split_scheme=='official':
            split_scheme = 'sites'

        self._split_scheme = split_scheme
        # if self._split_scheme == 'countries':
        #     self._split_dict = {
        #             "train_unlabeled": 10,
        #             "val_unlabeled": 11,
        #             "test_unlabeled": 12,
        #     }
        #     self._split_names = {
        #         "train_unlabeled": "Unlabeled Train",
        #         "val_unlabeled": "Unlabeled Validation",
        #         "test_unlabeled": "Unlabeled Test",
        #     }
        # else:
        #     raise ValueError("Split scheme not recognized")

        # self.no_nl = no_nl
        # if fold not in {'A', 'B', 'C', 'D', 'E'}:
        #     raise ValueError("Fold must be A, B, C, D, or E")

        fold = 'A'
        self.root = Path(self._data_dir)
        self.metadata = pd.read_csv(self.root / 'images/metadata.tsv', sep='\t')
        # country_folds = SURVEY_NAMES[f'2009-17{fold}']
        self.metadata['site'] = self.metadata['site'] -1
        self.metadata['study'] = self.metadata['study'] -1
        
        site_folds = _SPLIT_DATA
        # print("total images is: ", len(self.metadata))
        self._split_array = -1 * np.ones(len(self.metadata))
        
        insite_folds_split = np.arange(len(self.metadata))
        # take the test countries to be ood
        idxs_id, idxs_ood_test = split_by_site(insite_folds_split, site_folds['test'], self.metadata)
        # also create a validation OOD set
        idxs_id, idxs_ood_val = split_by_site(idxs_id, site_folds['val'], self.metadata)

        print("unlabeled train: ", len(idxs_id), file=open('small_log.txt', 'a')) # 387

        self._split_array[idxs_id] = self._split_dict['train_unlabeled']
        self._split_array[idxs_ood_val] = self._split_dict['val_unlabeled']
        self._split_array[idxs_ood_test] = self._split_dict['test_unlabeled']

        # no labels
        self.metadata['age'] = (-100 * np.ones(len(self.metadata)))
        # no urban/rural classification
        self.metadata['study'] = (-100 * np.ones(len(self.metadata)))

        # add country group field
        site_to_idx = {site: i for i, site in enumerate(DHS_SITES)}
        self.metadata['site'] = [site_to_idx[site] for site in self.metadata['site'].tolist()]
        self._metadata_map = {'site': DHS_SITES}
        # rename wealthpooled to y
        self._metadata_fields = ['study', 'y', 'site', 'participant_id']
        self._metadata_array = torch.from_numpy(self.metadata[['study', 'age', 'site', 'participant_id']].astype(float).to_numpy())
        super().__init__(root_dir, download, split_scheme)

    def get_input(self, idx):
        """
        Returns x for a given idx.
        """
        #print(idx)
        participant_id = self.metadata['participant_id'][idx]
        # print("participant_id", participant_id)
        # _preproc-quasiraw_T1w
        # _preproc-cat12vbm_desc-gm_T1w
        print(self.root)
        try:
            img = np.load(self.root / 'images' / f'sub-{participant_id}_preproc-cat12vbm_desc-gm_T1w.npy')
        except FileNotFoundError:
            print("File not found!, using different root directory  ", participant_id)
            new_dir = os.path.join('/data/ssafa013/wildsTest/data', f'{self.dataset_name}_v{self.version}')
            img = np.load(Path(new_dir) / 'images' / f'sub-{participant_id}_preproc-cat12vbm_desc-gm_T1w.npy')
            pass
        img = torch.from_numpy(img).float()

        return img
