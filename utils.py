import os

import numpy as np
import torch
import torch.nn.functional as F
from scipy import io as scio
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from DataProcessing_OneHot import get_one_subject_dataset, get_one_subject_dataset_IV, get_one_subject_dataset_V, get_one_subject_dataset_all




def data_pipeline(rawdata, labels, domain):

    data_num = labels.shape[0]
    all_data_list = []
    for i in range(0, data_num):
        new_datalist = [rawdata[i], labels[i]]
        domain = torch.tensor(domain)
        domain_one_hot = F.one_hot(domain, num_classes=14)
        new_datalist.append(domain_one_hot.numpy())
        all_data_list.append(new_datalist)

    return all_data_list


def data_pipeline_v(rawdata, labels, domain):

    data_num = labels.shape[0]
    all_data_list = []
    for i in range(0, data_num):

        new_datalist = [rawdata[i], labels[i]]
        domain = torch.tensor(domain)
        domain_one_hot = F.one_hot(domain, num_classes=15)
        new_datalist.append(domain_one_hot.numpy())
        all_data_list.append(new_datalist)

    return all_data_list


def random_split_domains(source_num):
    """
    Separating the source domain data and target domain data
    Args:
        source_num:

    Returns:
        source_dataset： list(num)
        target_dataset： list(num)
    """
    import random
    numberlist = list(range(1, 16))
    random_elements = random.sample(numberlist, source_num)
    source_dataset = []
    target_dataset = []
    domain_num = len(random_elements)
    for i in range(1, 16):
        if i in random_elements:
            features, labels = get_one_subject_dataset(i, 1)
            result = data_pipeline_v(features, labels, i)
            source_dataset = source_dataset + result
        else:
            features, labels = get_one_subject_dataset(i, 1)
            result = data_pipeline_v(features, labels, 0)
            target_dataset = target_dataset + result

    return source_dataset, target_dataset


def split_domain(test_num):

    source_dataset = []
    target_dataset = []

    domain_idx = 0
    for i in range(1, 16):  # 1~15

        if i != test_num:
            features, labels = get_one_subject_dataset(i, 1)
            result = data_pipeline(features, labels, domain_idx)
            source_dataset = source_dataset + result
            domain_idx = domain_idx + 1
        else:
            features, labels = get_one_subject_dataset(i, 1)
            result = data_pipeline(features, labels, 0)
            target_dataset = target_dataset + result

    return source_dataset, target_dataset


def split_domain_IV(test_num):

    source_dataset = []
    target_dataset = []
    domain_idx = 0
    for i in range(1, 16):  # 1~15

        if i != test_num:
            features, labels = get_one_subject_dataset_IV(i, 1)
            result = data_pipeline(features, labels, domain_idx)
            source_dataset = source_dataset + result
            domain_idx = domain_idx + 1
        else:
            features, labels = get_one_subject_dataset_IV(i, 1)
            result = data_pipeline(features, labels, 0)
            target_dataset = target_dataset + result

    return source_dataset, target_dataset


def split_domain_V(test_num):

    source_dataset = []
    target_dataset = []
    domain_idx = 0
    for i in range(1, 17):  # 1~16

        if i != test_num:
            features, labels = get_one_subject_dataset_V(i, 1)
            result = data_pipeline_v(features, labels, domain_idx)
            source_dataset = source_dataset + result
            domain_idx = domain_idx + 1
        else:
            features, labels = get_one_subject_dataset_V(i, 1)
            result = data_pipeline_v(features, labels, 0)
            target_dataset = target_dataset + result

    return source_dataset, target_dataset


def partition_into_three(all_list):
    """
    The data of the original dataset is a total list, and each list element in the list represents a trial.
    The list element has three elements representing source data, class label and domain label respectively
    For example, all_list[0] represents the 0th trial
        |---- all_list[0][0]: This represents the source data for the 0th trial
        |---- all_list[0][1]: Class label for the 0th trial
        |---- all_list[0][2]: This represents the field label for the 0th trial
    For the convenience of subsequent operation, it is converted into three tables,
    representing source data, class label and domain label respectively
    Args:
        all_list:

    Returns:
        (rawdata, c_labels, d_labels)

    """
    list_1 = [sublist[0] for sublist in all_list]
    list_2 = [sublist[1] for sublist in all_list]
    list_3 = [sublist[2] for sublist in all_list]
    new_data_list = [list_1, list_2, list_3]

    rawdata = np.array(new_data_list[0])
    rawdata = torch.tensor(rawdata, dtype=torch.float32)
    c_labels = np.array(new_data_list[1])
    c_labels = torch.tensor(c_labels, dtype=torch.float32)
    d_labels = np.array(new_data_list[2])
    d_labels = torch.tensor(d_labels, dtype=torch.float32)

    return rawdata.cuda(), c_labels.cuda(), d_labels.cuda()


class PackedDataset(Dataset):
    """
    Wrap the processed datalist into a Dataset object

    """

    def __init__(self, all_datalist):
        self.datalist = all_datalist

    def __getitem__(self, index):
        return self.datalist[index]

    def __len__(self):
        return len(self.datalist)


def split_domain_all_session(test_num, class_num):
    """
    all sesssion data
    Args:
        test_num:
        class_num: 3：seed  4:seed-iv 5:seed-v

    Returns:
        source_dataset, target_dataset
    """
    source_dataset = []
    target_dataset = []
    domain_idx = 0
    if class_num == 5:
        subject_num = 17
    else:
        subject_num = 16
    for i in range(1, subject_num):  # 1~15

        if i != test_num:
            features, labels = get_one_subject_dataset_all(i, class_num)
            if subject_num == 17:
                result = data_pipeline_v(features, labels, domain_idx)
            else:
                result = data_pipeline(features, labels, domain_idx)
            source_dataset = source_dataset + result
            domain_idx = domain_idx + 1
        else:
            features, labels = get_one_subject_dataset_all(i, class_num)
            if subject_num == 17:
                result = data_pipeline_v(features, labels, 0)
            else:
                result = data_pipeline(features, labels, 0)
            target_dataset = target_dataset + result

    return source_dataset, target_dataset


if __name__ == "__main__":
    # test the code
    target, source = get_one_subject_dataset_all(1, 4)
    # train_dataset, test_dataset = split_domain(1)
    print(target.shape, source.shape)

