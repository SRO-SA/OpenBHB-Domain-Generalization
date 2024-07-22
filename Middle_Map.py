from neuroCombat import neuroCombat
import pandas as pd
import numpy as np
import torch as t
from DataLoader import OpenBHBDataset
from wilds.common.data_loaders import get_train_loader, get_eval_loader
import gc



# Getting example data
# 200 rows (features) and 10 columns (scans)
# data = np.genfromtxt('testdata/testdata.csv', delimiter=",", skip_header=1)
dataset = OpenBHBDataset()

gc.collect()
train_dataset = dataset.get_subset('train')
val_dataset = dataset.get_subset('val')
id_val_dataset = dataset.get_subset('id_val')
test_dataset = dataset.get_subset('test')
id_test_dataset = dataset.get_subset('id_test')


train_loader = get_train_loader("standard", train_dataset, batch_size=1)
validation_loader = get_eval_loader("standard", val_dataset, batch_size=1)
id_validation_loader = get_eval_loader('standard', id_val_dataset, batch_size=1)
test_loader = get_eval_loader('standard', test_dataset, batch_size=1)

# print((train_dataset.metadata_array[:, 3]).type(t.int64))
# print((val_dataset.metadata_array[:, 3]).type(t.int64))

flat_data = []
sites = []
ids = []
ages = []


i = 0
for l in train_loader:
    image = l[0]
    site = l[2]
    age = l[1]

    id = (train_dataset.metadata_array[:, 3])[i].type(t.int64).item()
    # print(image.shape) #torch.Size([1, 1, 1, 121, 145, 121])
    # print(site.numpy()[0][2])
    # print(tensor.numpy())
    flatted = t.flatten(image)
    # print(flatted)
    numpy_flatted = flatted.numpy()
    # numpy_flatted = np.transpose(numpy_flatted)
    # print(numpy_flatted.shape)
    flat_data.append(np.ndarray.tolist(numpy_flatted))
    sites.append(site.numpy()[0][2])
    ids.append(id)
    ages.append(age.numpy()[0][0])
    i+=1
    
    # else:
    #     flat_data['part2'].append(np.ndarray.tolist(numpy_flatted))

i=0
for l in validation_loader:
    image = l[0]
    site = l[2]
    age = l[1]

    id = (val_dataset.metadata_array[:, 3])[i].type(t.int64).item()
    # print(image.shape) #torch.Size([1, 1, 1, 121, 145, 121])
    # print(site.numpy()[0][2])
    # print(tensor.numpy())
    flatted = t.flatten(image)
    # print(flatted)
    numpy_flatted = flatted.numpy()
    # numpy_flatted = np.transpose(numpy_flatted)
    # print(numpy_flatted.shape)
    flat_data.append(np.ndarray.tolist(numpy_flatted))
    sites.append(site.numpy()[0][2])
    ids.append(id)
    ages.append(age.numpy()[0][0])
    i+=1
    
    # else:
    #     flat_data['part2'].append(np.ndarray.tolist(numpy_flatted))
    

i=0
for l in test_loader:
    image = l[0]
    site = l[2]
    age = l[1]

    id = (test_dataset.metadata_array[:, 3])[i].type(t.int64).item()
    # print(image.shape) #torch.Size([1, 1, 1, 121, 145, 121])
    # print(site.numpy()[0][2])
    # print(tensor.numpy())
    flatted = t.flatten(image)
    # print(flatted)
    numpy_flatted = flatted.numpy()
    # numpy_flatted = np.transpose(numpy_flatted)
    # print(numpy_flatted.shape)
    flat_data.append(np.ndarray.tolist(numpy_flatted))
    sites.append(site.numpy()[0][2])
    ids.append(id)
    ages.append(age.numpy()[0][0])
    i+=1
    
    # else:
    #     flat_data['part2'].append(np.ndarray.tolist(numpy_flatted))
    
    
    
# Specifying the batch (scanner variable) as well as a biological covariate to preserve:
covars = {'site':sites}
print(len(sites))
# print(ids)
data = np.array(flat_data, dtype=np.float64)
data = np.transpose(data)

idxs =   pd.DataFrame({'id': ids})
covars = pd.DataFrame(covars)

idxs.to_csv('./TmpData/id.csv', index=False)
# covars.to_csv('./TmpData/covars.csv', index=False)

# np.save('./TmpData/data.npy', data)


