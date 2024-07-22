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
# train_dataset = dataset.get_subset('train')
# val_dataset = dataset.get_subset('val')
# id_val_dataset = dataset.get_subset('id_val')
# test_dataset = dataset.get_subset('test')
# id_test_dataset = dataset.get_subset('id_test')


# train_loader = get_train_loader("standard", train_dataset, batch_size=1)
# validation_loader = get_eval_loader("standard", val_dataset, batch_size=1)
# id_validation_loader = get_eval_loader('standard', id_val_dataset, batch_size=1)
# test_loader = get_eval_loader('standard', test_dataset, batch_size=1)

# print((train_dataset.metadata_array[:, 3]).type(t.int64))
# print((val_dataset.metadata_array[:, 3]).type(t.int64))

flat_data = []
sites = []
ids = []
ages = []

# i = 0
# for l in train_loader:
#     image = l[0]
#     site = l[2]
#     age = l[1]

#     id = (train_dataset.metadata_array[:, 3])[i].type(t.int64).item()
#     # print(image.shape) #torch.Size([1, 1, 1, 121, 145, 121])
#     # print(site.numpy()[0][2])
#     # print(tensor.numpy())
#     flatted = t.flatten(image)
#     # print(flatted)
#     numpy_flatted = flatted.numpy()
#     # numpy_flatted = np.transpose(numpy_flatted)
#     # print(numpy_flatted.shape)
#     flat_data.append(np.ndarray.tolist(numpy_flatted))
#     sites.append(site.numpy()[0][2])
#     ids.append(id)
#     ages.append(age.numpy()[0][0])
#     i+=1
    
#     # else:
#     #     flat_data['part2'].append(np.ndarray.tolist(numpy_flatted))

    
# # Specifying the batch (scanner variable) as well as a biological covariate to preserve:
# covars = {'site':sites}
# print(len(sites))
# # print(ids)
# data = np.array(flat_data, dtype=np.float64)
# data = np.transpose(data)
# data_part1 = data[:2122945//2, :]
# print(data_part1.shape)
# # print(sites)
# covars = pd.DataFrame(covars)  

# # To specify names of the variables that are categorical:
# # continuous_cols = ['age']

# # To specify the name of the variable that encodes for the scanner/batch covariate:
# batch_col = 'site'
# #Harmonization step:
# # print(data)
# # print(sites)
# data_combat_part1 = neuroCombat(dat=data_part1,
#     covars=covars,
#     batch_col=batch_col)["data"]

# print(data_combat_part1.shape)
# gc.collect()

# covars = {'site':sites}
# print(len(sites))
# # print(ids)
# data_part2 = data[2122945//2:, :]
# print(data_part2.shape)
# # print(sites)
# covars = pd.DataFrame(covars)  

# # To specify names of the variables that are categorical:
# # continuous_cols = ['age']

# # To specify the name of the variable that encodes for the scanner/batch covariate:
# batch_col = 'site'
# #Harmonization step:
# # print(data)
# # print(sites)
# data_combat_part2 = neuroCombat(dat=data_part2,
#     covars=covars,
#     batch_col=batch_col)["data"]


# data_combat = np.concatenate((data_combat_part1, data_combat_part2), axis=0)
# harmonized_data = np.transpose(data_combat)
# test_harmonized = []
# print(harmonized_data.shape)
# i = 0
# for image in harmonized_data:
#     # print(image.shape)
#     # if np.isnan(image).any():
#     #     print(image)
#     id = ids[i]
#     harmonized_tensor = t.from_numpy(image)
#     unflattened_data = t.unflatten(harmonized_tensor, 0, (1, 1, 121, 145, 121))
#     res = np.array(unflattened_data)
#     file_name = f'sub-{id}_preproc-cat12vbm_desc-gm_T1w.npy'
#     np.save('./data/openBHB_v1.1_harmonized/'+file_name, res)
#     i+=1









# i = 0
# for l in train_loader:
#     image = l[0]
#     site = l[2]
#     age = l[1]

#     id = (train_dataset.metadata_array[:, 3])[i].type(t.int64).item()
#     # print(image.shape) #torch.Size([1, 1, 1, 121, 145, 121])
#     # print(site.numpy()[0][2])
#     # print(tensor.numpy())
#     flatted = t.flatten(image)
#     # print(flatted)
#     numpy_flatted = flatted.numpy()
#     # numpy_flatted = np.transpose(numpy_flatted)
#     # print(numpy_flatted.shape)
#     flat_data.append(np.ndarray.tolist(numpy_flatted))
#     sites.append(site.numpy()[0][2])
#     ids.append(id)
#     ages.append(age.numpy()[0][0])
#     i+=1
    
#     # else:
#     #     flat_data['part2'].append(np.ndarray.tolist(numpy_flatted))

# i=0
# for l in validation_loader:
#     image = l[0]
#     site = l[2]
#     age = l[1]

#     id = (val_dataset.metadata_array[:, 3])[i].type(t.int64).item()
#     # print(image.shape) #torch.Size([1, 1, 1, 121, 145, 121])
#     # print(site.numpy()[0][2])
#     # print(tensor.numpy())
#     flatted = t.flatten(image)
#     # print(flatted)
#     numpy_flatted = flatted.numpy()
#     # numpy_flatted = np.transpose(numpy_flatted)
#     # print(numpy_flatted.shape)
#     flat_data.append(np.ndarray.tolist(numpy_flatted))
#     sites.append(site.numpy()[0][2])
#     ids.append(id)
#     ages.append(age.numpy()[0][0])
#     i+=1
    
#     # else:
#     #     flat_data['part2'].append(np.ndarray.tolist(numpy_flatted))
    

# i=0
# for l in test_loader:
#     image = l[0]
#     site = l[2]
#     age = l[1]

#     id = (test_dataset.metadata_array[:, 3])[i].type(t.int64).item()
#     # print(image.shape) #torch.Size([1, 1, 1, 121, 145, 121])
#     # print(site.numpy()[0][2])
#     # print(tensor.numpy())
#     flatted = t.flatten(image)
#     # print(flatted)
#     numpy_flatted = flatted.numpy()
#     # numpy_flatted = np.transpose(numpy_flatted)
#     # print(numpy_flatted.shape)
#     flat_data.append(np.ndarray.tolist(numpy_flatted))
#     sites.append(site.numpy()[0][2])
#     ids.append(id)
#     ages.append(age.numpy()[0][0])
#     i+=1
    
#     # else:
#     #     flat_data['part2'].append(np.ndarray.tolist(numpy_flatted))
    
    
    
# Specifying the batch (scanner variable) as well as a biological covariate to preserve:
sites = pd.read_csv('TmpData/covars.csv')['site'].tolist()
ids = pd.read_csv('TmpData/id.csv')['id'].tolist()
print(len(sites))
print(len(ids))


data = np.load('TmpData/data.npy')
print(data.shape)

covars = {'site':sites}
print(len(sites))

# print(ids)
# data = np.array(flat_data, dtype=np.float64)
# data = np.transpose(data)
data_part1 = data[:2122945//5, :]
print(data_part1.shape)
# print(sites)
covars = pd.DataFrame(covars)

# To specify names of the variables that are categorical:
# continuous_cols = ['age']

# To specify the name of the variable that encodes for the scanner/batch covariate:
batch_col = 'site'
#Harmonization step:
# print(data)
# print(sites)
data_combat_part1 = neuroCombat(dat=data_part1,
    covars=covars,
    batch_col=batch_col)["data"]

print(data_combat_part1.shape)
gc.collect()

covars = {'site':sites}
print(len(sites))
# print(ids)
data_part2 = data[(2122945//5):(2122945//5)*2, :]
print(data_part2.shape)
# print(sites)
covars = pd.DataFrame(covars)  

# To specify names of the variables that are categorical:
# continuous_cols = ['age']

# To specify the name of the variable that encodes for the scanner/batch covariate:
batch_col = 'site'
#Harmonization step:
# print(data)
# print(sites)
data_combat_part2 = neuroCombat(dat=data_part2,
    covars=covars,
    batch_col=batch_col)["data"]
gc.collect()

covars = {'site':sites}
print(len(sites))
# print(ids)
data_part3 = data[(2122945//5)*2:(2122945//5)*3, :]
print(data_part3.shape)
# print(sites)
covars = pd.DataFrame(covars)  

# To specify names of the variables that are categorical:
# continuous_cols = ['age']

# To specify the name of the variable that encodes for the scanner/batch covariate:
batch_col = 'site'
#Harmonization step:
# print(data)
# print(sites)
data_combat_part3 = neuroCombat(dat=data_part3,
    covars=covars,
    batch_col=batch_col)["data"]
gc.collect()

covars = {'site':sites}
print(len(sites))
# print(ids)
data_part4 = data[(2122945//5)*3:(2122945//5)*4, :]
print(data_part4.shape)
# print(sites)
covars = pd.DataFrame(covars)  

# To specify names of the variables that are categorical:
# continuous_cols = ['age']

# To specify the name of the variable that encodes for the scanner/batch covariate:
batch_col = 'site'
#Harmonization step:
# print(data)
# print(sites)
data_combat_part4 = neuroCombat(dat=data_part4,
    covars=covars,
    batch_col=batch_col)["data"]
gc.collect()


covars = {'site':sites}
print(len(sites))
# print(ids)
data_part5 = data[(2122945//5)*4:2122945, :]
print(data_part5.shape)
# print(sites)
covars = pd.DataFrame(covars)  

# To specify names of the variables that are categorical:
# continuous_cols = ['age']

# To specify the name of the variable that encodes for the scanner/batch covariate:
batch_col = 'site'
#Harmonization step:
# print(data)
# print(sites)
data_combat_part5 = neuroCombat(dat=data_part5,
    covars=covars,
    batch_col=batch_col)["data"]
gc.collect()

data_combat = np.concatenate((data_combat_part1, data_combat_part2), axis=0)
data_combat = np.concatenate((data_combat, data_combat_part3), axis=0)
data_combat = np.concatenate((data_combat, data_combat_part4), axis=0)
data_combat = np.concatenate((data_combat, data_combat_part5), axis=0)


harmonized_data = np.transpose(data_combat)
test_harmonized = []
print(harmonized_data.shape)
i = 0
for image in harmonized_data:
    # print(image.shape)
    # if np.isnan(image).any():
    #     print(image)
    id = ids[i]
    harmonized_tensor = t.from_numpy(image)
    unflattened_data = t.unflatten(harmonized_tensor, 0, (1, 1, 121, 145, 121))
    res = np.array(unflattened_data)
    file_name = f'sub-{id}_preproc-cat12vbm_desc-gm_T1w.npy'
    np.save('./data/openBHB_v1.1_harmonized/'+file_name, res)
    i+=1

