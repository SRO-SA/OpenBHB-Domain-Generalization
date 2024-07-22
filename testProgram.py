# from wilds import get_dataset
# from wilds.common.data_loaders import get_train_loader
# from wilds.common.grouper import CombinatorialGrouper
# import torchvision.transforms as transforms
# import torch as t


# # Load the full dataset, and download it if necessary
# dataset = get_dataset(dataset="camelyon17", download=False)


# # Get the training set
# train_data = dataset.get_subset(    
#     "train",
#     transform=transforms.Compose(
#         [transforms.Resize((96, 96)), transforms.ToTensor()]
#     ),
# )

# # Prepare the standard data loader
# train_loader = get_train_loader("standard", train_data, batch_size=16)

# # (Optional) Load unlabeled data
# dataset = get_dataset(dataset="camelyon17", download=False, unlabeled=True)
# unlabeled_data = dataset.get_subset(
#     "test_unlabeled",
#     transform=transforms.Compose(
#         [transforms.Resize((96, 96)), transforms.ToTensor()]
#     ),
# )
# unlabeled_loader = get_train_loader("standard", unlabeled_data, batch_size=16)

# # Train loop
# for labeled_batch, unlabeled_batch in zip(train_loader, unlabeled_loader):
#     x, y, metadata = labeled_batch
#     unlabeled_x, unlabeled_metadata = unlabeled_batch
#     print(metadata)






# # Initialize grouper, which extracts domain information
# # In this example, we form domains based on location
# # grouper = CombinatorialGrouper(dataset, [''])

# # # Train loop
# # for x, y_true, metadata in train_loader:
# #     z = grouper.metadata_to_group(metadata)
# #     print(z)

import pandas as pd

# Load train and validation data
val_df = pd.read_csv('./data/openBHB_v1.1/validation/participants.tsv', sep='\t')
train_df = pd.read_csv('./data/openBHB_v1.1/images/participants.tsv', sep='\t')

print(val_df.shape)
# Merge dataframes
merged_df = pd.concat([train_df, val_df], ignore_index=True)

sorted_metadata = merged_df.sort_values(by='participant_id')

# Save merged data
sorted_metadata.to_csv('./data/openBHB_v1.1/metadata.tsv', sep='\t', index=False)
