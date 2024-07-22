from neuroCombat import neuroCombat
import pandas as pd
import numpy as np
import torch as t
from DataLoader import OpenBHBDataset
from wilds.common.data_loaders import get_train_loader, get_eval_loader
import matplotlib.pyplot as plt
import seaborn as sn
import gc

def np_pearson_cor(x, y):
    xv = x - x.mean(axis=0)
    yv = y - y.mean(axis=0)
    xvss = (xv * xv).sum(axis=0)
    yvss = (yv * yv).sum(axis=0)
    result = np.matmul(xv.transpose(), yv) / np.sqrt(np.outer(xvss, yvss))
    # bound the values to -1 to 1 in the event of precision issues
    return np.maximum(np.minimum(result, 1.0), -1.0)

def group_cols_with_same_mask(x):
    "returns a sequence of tuples (mask, columns) where columns are the column indices in x which all have the mask"
    per_mask = {}
    for i in range(x.shape[1]):
        o_mask = np.isfinite(x[:,i])
        # take the binary vector o_mask and convert it to a compact
        # sequence of bytes which we can use as a dict key
        o_mask_b = np.packbits(o_mask).tobytes()
        if o_mask_b not in per_mask:
            per_mask[o_mask_b] = [o_mask, []]
        per_mask[o_mask_b][1].append(i)
    return per_mask.values()

def fast_cor_with_missing(x,y):
    # preallocate storage for the result
    result = np.zeros(shape=(x.shape[1], y.shape[1]))

    x_groups = group_cols_with_same_mask(x)
    y_groups = group_cols_with_same_mask(y)
    for x_mask, x_columns in x_groups:
        for y_mask, y_columns in y_groups:
            # print(x_mask, x_columns, y_mask, y_columns)
            combined_mask = x_mask & y_mask

            # not sure if this is the fastest way to slice out the relevant subset
            x_without_holes = x[:, x_columns][combined_mask, :]
            y_without_holes = y[:, y_columns][combined_mask, :]

            c = np_pearson_cor(x_without_holes, y_without_holes)

            # update result with these correlations
            result[np.ix_(x_columns, y_columns)] = c

    return result


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

flat_data = {"part1": [], "part2": []}
sites = {"part1": [], "part2": []}
ids = {"part1": [], "part2": []}
ages = {'part1': [], 'part2': []}

i = 0
for l in train_loader:
    image = l[0]
    site = l[2]
    age = l[1]

    id = (train_dataset.metadata_array[:, 3])[i].type(t.int64).item()
    # print(image.shape) #torch.Size([1, 1, 1, 121, 145, 121])
    # print(site.numpy()[0][2])
    # print(tensor.numpy())
    squeezed = t.squeeze(image)
    # print(flatted)
    numpy_squeezed = squeezed.numpy()
    data_shape = numpy_squeezed.shape
    # numpy_flatted = np.transpose(numpy_flatted)
    # print(numpy_flatted.shape)
    block_shape = (11, 29, 11)  # Define the shape of the blocks
    result_shape = (11, 5, 11)  # Define the shape of the result

    # Calculate the dimensions for splitting
    x_splits = data_shape[0] // block_shape[0]
    y_splits = data_shape[1] // block_shape[1]
    z_splits = data_shape[2] // block_shape[2]

    # Reshape the original data to allow for mean calculation
    data_reshaped = numpy_squeezed[:x_splits * block_shape[0], :y_splits * block_shape[1], :z_splits * block_shape[2]].reshape(
        x_splits, block_shape[0], y_splits, block_shape[1], z_splits, block_shape[2])

    # Calculate the mean along the specified axes
    result = np.mean(data_reshaped, axis=(1, 3, 5))
    unflattened_data = np.ndarray.flatten(result)
    print(result.shape)

    if i<3000 :  
        flat_data['part1'].append(unflattened_data)
        sites['part1'].append(site.numpy()[0][2])
        ids['part1'].append(id)
        ages['part1'].append(age.numpy()[0][0])


    else:
        flat_data['part2'].append(unflattened_data)
        sites['part2'].append(site.numpy()[0][2])
        ids['part2'].append(id)
        ages['part2'].append(age.numpy()[0][0])
    i+=1
    
    # else:
    #     flat_data['part2'].append(np.ndarray.tolist(numpy_flatted))

    
# Specifying the batch (scanner variable) as well as a biological covariate to preserve:
# covars = {'site':sites['part1'], 'age':ages['part1']} 
print(len(sites['part1']))
# print(ids)
data = np.array(flat_data['part1'], dtype=np.float64)

# Assuming your original 3D array is 'data'


# The 'result' array now contains the mean values for each (11,5,11) block


print(data.shape)
# pcorr = np_pearson_cor(data, data)
pcorr = fast_cor_with_missing(data.T,data.T)
# # correlation_matrix = np.corrcoef(data)
# # Get number of rows in either A or B
# N = data.shape[0]
plot_data = np.ma.masked_equal(pcorr[:,:], 0)

# plt.subplots_adjust(left=0.1, bottom=0.15, right=0.99, top=0.95)
# plt.imshow(plot_data, cmap=plt.colormaps.get_cmap("Reds"), interpolation="nearest", aspect = "auto")
# plt.xticks(range(605), 605, rotation=90, va="top", ha="center")
# plt.colorbar()
# # Store columnw-wise in A and B, as they would be used at few places
# sA = data.sum(0)
# sB = data.sum(0)

# # Basically there are four parts in the formula. We would compute them one-by-one
# p1 = p1 = N*np.dot(data.T,data)
# p2 = sA*sB[:,None]
# p3 = N*((data**2).sum(0)) - (sB**2)
# p4 = N*((data**2).sum(0)) - (sA**2)

# # Finally compute Pearson Correlation Coefficient as 2D array 
# pcorr = ((p1 - p2)/np.sqrt(p4*p3[:,None]))
print(pcorr.shape)

# Get the element corresponding to absolute argmax along the columns 
# out = pcorr[np.nanargmax(np.abs(pcorr),axis=0),np.arange(pcorr.shape[1])]




sn.heatmap(pcorr, annot=False)
# plt.show()

plt.savefig('foo.pdf')
plt.savefig('foo.png')

# data = np.transpose(data)
# data_part1 = data[:2122945//2, :]
# print(data_part1.shape)
# print(sites)
# covars = pd.DataFrame(covars)  

# To specify names of the variables that are categorical:
# continuous_cols = ['age']

# To specify the name of the variable that encodes for the scanner/batch covariate:
# batch_col = 'site'
#Harmonization step:
# print(data)
# print(sites)
# data_combat_part1 = neuroCombat(dat=data_part1,
#     covars=covars,
#     batch_col=batch_col,
#     continuous_cols=continuous_cols)["data"]

# print(data_combat_part1.shape)
# gc.collect()

# covars = {'site':sites['part1'], 'age':ages['part1']} 
# print(len(sites['part1']))
# print(ids)
# data_part2 = data[2122945//2:, :]
# print(data_part2.shape)
# print(sites)
# covars = pd.DataFrame(covars)  

# To specify names of the variables that are categorical:
# continuous_cols = ['age']

# To specify the name of the variable that encodes for the scanner/batch covariate:
# batch_col = 'site'
#Harmonization step:
# print(data)
# print(sites)
# data_combat_part2 = neuroCombat(dat=data_part2,
#     covars=covars,
#     batch_col=batch_col,
#     continuous_cols=continuous_cols)["data"]


# data_combat = np.concatenate((data_combat_part1, data_combat_part2), axis=0)
# harmonized_data = np.transpose(data_combat)
test_harmonized = []
# print(harmonized_data.shape)
# i = 0
# for image in harmonized_data:
#     # print(image.shape)
#     # if np.isnan(image).any():
#     #     print(image)
#     id = ids['part1'][i]
#     harmonized_tensor = t.from_numpy(image)
#     unflattened_data = t.unflatten(harmonized_tensor, 0, (1, 1, 121, 145, 121))
#     res = np.array(unflattened_data)
#     file_name = f'sub-{id}_preproc-cat12vbm_desc-gm_T1w.npy'
#     np.save('./data/openBHB_v1.1_harmonized/'+file_name, res)
#     i+=1
    
    

    
