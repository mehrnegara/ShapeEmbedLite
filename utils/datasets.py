import torch
import torchvision
import numpy as np

def get_dataloaders( dataset_path
                   , batch_size=4
                   , split_train_test=(0.8, 0.2)
                   , split_train_valid=(0.8, 0.2)
                   ):
  # matrix to tensor loader and raw dataset
  def loader(x):
    x = np.load(x)
    x = np.expand_dims(x, axis=0)
    return torch.tensor(x, dtype=torch.float32)
  if not isinstance(dataset_path, list):
    dataset = torchvision.datasets.DatasetFolder( dataset_path
                                                , loader
                                                , extensions='npy' )
    train_set, test_set = torch.utils.data.random_split(dataset, split_train_test)
    train_set, val_set = torch.utils.data.random_split(train_set, split_train_valid)
  else:
    assert len(dataset_path) == 2
    dataset = torchvision.datasets.DatasetFolder( dataset_path[0]
                                                , loader
                                                , extensions='npy' )
    train_set, val_set = torch.utils.data.random_split(dataset, split_train_valid)
    test_set = torchvision.datasets.DatasetFolder( dataset_path[1]
                                                 , loader
                                                 , extensions='npy' )
    test_set = torch.utils.data.Subset(test_set, range(len(test_set)))
  # dataloader for training/validation and testing
  train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
  val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False, drop_last=True)
  test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False)
  return train_loader, val_loader, test_loader
