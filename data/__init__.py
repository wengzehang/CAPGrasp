import torch.utils.data
from data.base_dataset import collate_fn
import numpy as np
import random
import torch

def CreateDataset(opt, caching=True):
    """loads dataset class"""

    if opt.arch == 'vae' or opt.arch == 'gan':
        if opt.orientation_constrained or opt.vanilla:
            from data.grasp_sampler_data import GraspSamplerData
            dataset = GraspSamplerData(opt, caching=caching)
        else:
            raise NotImplementedError("Only support CAPGrasp and GraspNet")
    else:
        from data.grasp_evaluator_data import GraspEvaluatorData
        dataset = GraspEvaluatorData(opt)

    return dataset

class DataLoader:
    """multi-threaded data loading"""

    def __init__(self, opt, caching=True):
        self.opt = opt
        self.caching = caching
        self.create_dataset(caching=self.caching)
        np.random.seed(10)
        torch.manual_seed(0)
        random.seed(0)

    def create_dataset(self, caching=True):
        self.dataset = CreateDataset(self.opt, caching=caching)

    def adapt_old_model(self):
        self.dataset.orien_dict[3] = [-1,-1] # for the old training model

    def split_dataset(self, split_size_percentage=[0.8, 0.15, 0.05]):
        dataset_size = len(self.dataset)
        number_of_training_samples = round(split_size_percentage[0]*dataset_size)
        number_of_test_samples = round(split_size_percentage[1]*dataset_size)
        number_of_validation_samples = dataset_size - number_of_training_samples - number_of_test_samples
        return torch.utils.data.random_split(
            self.dataset, [number_of_training_samples, number_of_test_samples, number_of_validation_samples])

    def create_dataloader(self, data_loader, shuffle_batches):
        self.dataloader = torch.utils.data.DataLoader(
            data_loader,
            batch_size=self.opt.num_objects_per_batch,
            shuffle=shuffle_batches,
            num_workers=int(self.opt.num_threads),
            collate_fn=collate_fn)
        return self.dataloader

    def load_data(self):
        return self.dataloader

    def __len__(self):
        return min(len(self.dataset), self.opt.max_dataset_size)

    def __iter__(self):
        for i, data in enumerate(self.dataloader):
            if i * self.opt.batch_size >= self.opt.max_dataset_size:
                break
            yield data
