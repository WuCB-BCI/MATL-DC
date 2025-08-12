import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from typing import List, Dict
from sklearn import metrics
from sklearn.cluster import KMeans
from torch.autograd import Function
from torch.utils.data import DataLoader
import time
from utils import split_domain, partition_into_three, PackedDataset


class FeatureExtractor(nn.Module):
    """
    FeatureExtractor

    """

    def __init__(self, fea_len, hidden_1, hidden_2):
        super(FeatureExtractor, self).__init__()
        self.fc1 = nn.Linear(fea_len, hidden_1)
        self.fc2 = nn.Linear(hidden_1, hidden_2)

    def forward(self, x):
        x = self.fc1(x)
        # x = F.relu(x)
        x=F.leaky_relu(x)
        x = self.fc2(x)
        # x = F.relu(x)
        x=F.leaky_relu(x)
        return x

    def get_parameters(self) -> List[Dict]:
        params = [
            {"params": self.fc1.parameters(), "lr_mult": 1},
            {"params": self.fc2.parameters(), "lr_mult": 1},
        ]
        return params

class DomainDisentangler(nn.Module):
    """
    domian feature decoupler
    """

    def __init__(self, shallow_fea, hidden_1, hidden_2):
        super().__init__()
        self.fc1 = nn.Linear(shallow_fea, hidden_1)
        self.fc2 = nn.Linear(hidden_1, hidden_2)

    def forward(self, x):
        x = self.fc1(x)
        # x = F.relu(x)
        x = F.leaky_relu(x)
        x = self.fc2(x)
        # x = F.relu(x)
        x = F.leaky_relu(x)
        return x

    def get_parameters(self) -> List[Dict]:
        params = [
            {"params": self.fc1.parameters(), "lr_mult": 1},
            {"params": self.fc2.parameters(), "lr_mult": 1},
        ]
        return params

class ClassDisentangler(nn.Module):
    """
    class feature decoupler
    """

    def __init__(self, shallow_fea, hidden_1, hidden_2):
        super().__init__()
        self.fc1 = nn.Linear(shallow_fea, hidden_1)
        self.fc2 = nn.Linear(hidden_1, hidden_2)

    def forward(self, x):
        x = self.fc1(x)
        # x = F.relu(x)
        x = F.leaky_relu(x)
        x = self.fc2(x)
        # x = F.relu(x)
        x = F.leaky_relu(x)
        return x

    def get_parameters(self) -> List[Dict]:
        params = [
            {"params": self.fc1.parameters(), "lr_mult": 1},
            {"params": self.fc2.parameters(), "lr_mult": 1},
        ]
        return params

class CommonDiscriminator(nn.Module):
    """
    discriminator
    """

    def __init__(self, hidden_1, hidden_2, class_num):
        super(CommonDiscriminator, self).__init__()
        self.fc1 = nn.Linear(hidden_1, hidden_2)
        self.fc2 = nn.Linear(hidden_2, class_num)
        self.dropout1 = nn.Dropout(p=0.25)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        # x = F.relu(x)
        x=F.leaky_relu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

    def get_parameters(self) -> List[Dict]:
        params = [
            {"params": self.fc1.parameters(), "lr_mult": 1},
            {"params": self.fc2.parameters(), "lr_mult": 1},
        ]
        return params

class ReverseLayerF(Function):
    """
    Gradient reversal layer, for dann
    such as：
        disc_in1 = Adver_network.ReverseLayerF.apply(z1, self.args.alpha1)    # alpha1=0.1
        disc_out1 = self.ddiscriminator(disc_in1)
        disc_loss = F.cross_entropy(disc_out1, all_d1, reduction='mean')

    """

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

class DemoModel(nn.Module):
    def __init__(self, parameter):
        super().__init__()
        self.fea_extractor = FeatureExtractor(parameter["fea_len"], parameter["fea_h1"], parameter["fea_h2"])

        self.domain_fea = DomainDisentangler(parameter["fea_h2"], parameter["domain_fea_h1"],
                                             parameter["domain_fea_h2"])
        self.class_fea = ClassDisentangler(parameter["fea_h2"], parameter["class_fea_h1"], parameter["class_fea_h2"])

        self.domain_disc = CommonDiscriminator(parameter["domain_fea_h2"], parameter["domain_dis_h2"],
                                               parameter["domain_num"])
        self.class_disc = CommonDiscriminator(parameter["class_fea_h2"], parameter["class_dis_h2"],
                                              parameter["class_num"])

        self.domain_p = torch.rand((4,64))
        self.class_p = torch.rand((4,3,64))
        self.mapping = None
        self.alpha = parameter['alpha']
        self.max_iter = parameter['max_iter']
        self.upper_threshold = parameter['upper_threshold']
        self.lower_threshold = parameter['lower_threshold']

    def forward(self, s_raw):
        """

        Args:

        Returns:
        dict:{
        "class_fea": Class characteristics
        "domain_fea": Domain characteristics
        "s_c": The class of s according to its class characteristics
        "s_d": The domain of s according to its domain characteristics
        "s_cd": The domain of s judged according to the class features of s, which is used in the subsequent dann
        "s_dc": The class of s judged according to the domain characteristics of s, which is used in the subsequent dann
        }
        """
        self.train()

        s_data = s_raw
        s_data = s_data.to(dtype=torch.float32)
        s_fea = self.fea_extractor(s_data)
        s_class_fea = self.class_fea(s_fea)
        s_domain_fea = self.domain_fea(s_fea)

        s_c = self.class_disc(s_class_fea)
        # GRL
        s_cd = ReverseLayerF.apply(s_class_fea, 0.1)
        s_cd = self.domain_disc(s_cd)

        s_d = self.domain_disc(s_domain_fea)
        # GRL
        s_dc = ReverseLayerF.apply(s_domain_fea, 0.1)
        s_dc = self.class_disc(s_dc)

        result = {"class_fea": s_class_fea, "domain_fea": s_domain_fea, "s_c": s_c, "s_d": s_d,
                  "s_cd": s_cd, "s_dc": s_dc}

        return result

    def get_features(self, data):
        """
        The domain features and class features of the data are obtained
        Args:
            data:A list with three elements: source data, class label, and domain label

        Returns:
            class_fea:tensor(num,fea_len)
            domain_fea:tensor(num,fea_len)

        """
        fea = self.fea_extractor(data)
        class_fea = self.class_fea(fea)
        domain_fea = self.domain_fea(fea)
        return class_fea, domain_fea

    def get_prototype(self, source_zip):
        """
        Compute class prototypes and domain prototypes
        Args:
            source_zip:

        Returns:
            domain_prototype_matrix：tensor(14,64)
            class_prototype_matrix：tensor(14,3,64)

        """

        samples_num = source_zip[0].size()[0]
        subject_num = int(source_zip[2].size()[1])
        one_trail_samples = int(samples_num / subject_num)
        all_selecet_list = []
        for i in range(0, samples_num, one_trail_samples):
            list_1 = source_zip[0][i:i + one_trail_samples]
            list_2 = source_zip[1][i:i + one_trail_samples]
            datalist = [list_1, list_2]
            all_selecet_list.append(datalist)

        domain_prototype_list = []
        class_prototype_list = []

        for x in all_selecet_list:
            data = x[0]

            c_fea, d_fea = self.get_features(data)
            d_prototype = torch.mean(d_fea, dim=0)
            labels = x[1]  # (100,3)
            num_of_class = labels.size()[1]
            c_prototype = torch.matmul(torch.inverse(torch.diag(labels.sum(axis=0)) + torch.eye(num_of_class).cuda()),
                             torch.matmul(labels.T, c_fea))

            domain_prototype_list.append(d_prototype)
            class_prototype_list.append(c_prototype)

            domain_prototype_matrix = torch.stack(domain_prototype_list)
            class_prototype_matrix = torch.stack(class_prototype_list)

        return domain_prototype_matrix, class_prototype_matrix

    def replace_labels(self,source_dlabels,mapping):
        '''
        Map old labels to new labels
        Args:
            source_dlabels:
            mapping:

        Returns:(n,)nparray

        '''

        source_dlabels = torch.argmax(source_dlabels, dim=1).cpu().detach().numpy()
        new_labels = np.copy(source_dlabels)

        for old_label, new_label in mapping.items():
            new_labels[source_dlabels == old_label] = new_label  # 替换标签
        # F.one_hot(torch.tensor(new_labels), n_clusters)

        return new_labels



    def prototype_init_v2(self,source_zip,n_clusters):
        '''
        初Initialize the domain prototype and partition the domain
        Args:
            source_zip:
            n_clusters:

        Returns:

        '''

        def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
            '''
            The source domain data and target domain data are transformed into a kernel matrix, which is K in the above
            Params:
            source: Source domain data, row denotes the number of samples, column denotes the dimension of sample data
            target: The target domain data is the same as the source
            kernel_mul: Multicore MMD with bandwidth as the center and expanded cardinality on both sides, such as bandwidth/kernel_mul, bandwidth, bandwidth*kernel_mul
            kernel_num: This takes the number of distinct Gaussian kernels
            fix_sigma: Whether fixed, if fixed, single-core MMD
            Return:
            sum(kernel_val): The sum of multiple kernel matrices
            '''
            n_samples = int(source.size()[0]) + int(target.size()[0])
            total = torch.cat([source, target], dim=0)  # 将source,target按列方向合并
            total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
            total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
            L2_distance_square = ((total0 - total1) ** 2).sum(2)
            if fix_sigma:
                bandwidth = fix_sigma
            else:
                bandwidth = torch.sum(L2_distance_square) / (n_samples ** 2 - n_samples)
            bandwidth /= kernel_mul ** (kernel_num // 2)
            bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]

            kernel_val = [torch.exp(-L2_distance_square / bandwidth_temp) for bandwidth_temp in bandwidth_list]
            del L2_distance_square,bandwidth_list,bandwidth,total,total0,total1
            return sum(kernel_val)  # /len(kernel_val)

        def compute_mmd(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
            batch_size = int(source.size()[0])
            kernels = guassian_kernel(source, target,
                                      kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
            XX = kernels[:batch_size, :batch_size]
            YY = kernels[batch_size:, batch_size:]
            XY = kernels[:batch_size, batch_size:]
            YX = kernels[batch_size:, :batch_size]
            mmd = torch.mean(XX + YY - XY - YX)
            del XX,YY,XY,YX
            return mmd.item()

        self.eval()
        with torch.no_grad():
            source_rawdata = source_zip[0]
            source_clabels = source_zip[1]
            source_dlabels = source_zip[2]

            c_fea, d_fea = self.get_features(source_rawdata)
            n_people = 14
            n_samples_per_person = 3394
            n_features = 64

            mmd_matrix = torch.zeros((n_people,n_people))
            for i in range(n_people):
                for j in range(i+1,n_people):
                    source_batch = d_fea[i * n_samples_per_person: (i + 1) * n_samples_per_person]
                    target_batch = d_fea[j * n_samples_per_person: (j + 1) * n_samples_per_person]
                    mmd_value = 0.0
                    batch_size = 1697
                    for batch_start in range(0, source_batch.size(0), batch_size):
                        batch_end = min(batch_start + batch_size, source_batch.size(0))
                        mmd_value += compute_mmd(source_batch[batch_start:batch_end],
                                                 target_batch[batch_start:batch_end])

                    # Calculate the mean MMD
                    mmd_matrix[i, j] = mmd_value / (source_batch.size(0) // batch_size)
                    mmd_matrix[j, i] = mmd_matrix[i, j]
            # Clear the CUDA cache to free the video memory
            torch.cuda.empty_cache()
            kmeans = KMeans(n_clusters,init='k-means++',n_init=10)
            mmd_matrix = mmd_matrix.cpu().detach().numpy()
            labels = kmeans.fit_predict(mmd_matrix)
            mapping = {i: label for i, label in enumerate(labels)}
            new_d_labels = self.replace_labels(source_dlabels, mapping)
            new_d_labels = torch.tensor(new_d_labels).cuda()

            one_hot_d_label = F.one_hot(new_d_labels, n_clusters).to(torch.float32)
            temp_cpu = torch.inverse(torch.diag(one_hot_d_label.sum(axis=0)).to('cpu') + torch.eye(one_hot_d_label.size()[1]).cuda().to('cpu'))
            init_d_p = torch.matmul(
                temp_cpu.to('cuda'),
                torch.matmul(one_hot_d_label.T, d_fea))

            class_prototype_matrix = []
            for i in range(n_clusters):
                mask = (new_d_labels == i)
                group_fea = c_fea[mask]
                group_label = source_clabels[mask]

                num_of_class = source_clabels.size()[1]
                tmp_cpu =torch.inverse(torch.diag(group_label.sum(axis=0)).cpu() + torch.eye(num_of_class).cpu())
                c_prototype = torch.matmul(
                    tmp_cpu.cuda(),
                    torch.matmul(group_label.T, group_fea))
                class_prototype_matrix.append(c_prototype)

            class_prototype_matrix = torch.stack(class_prototype_matrix, dim=0)

            init_d_p = torch.as_tensor(init_d_p).cuda()
        return init_d_p.detach(), class_prototype_matrix.detach(), mapping




    def prototype_update(self,source_zip,n_clusters,epoch):
        self.eval()
        source_rawdata = source_zip[0]
        source_clabels = source_zip[1]
        source_dlabels = source_zip[2]
        s_c_fea, s_d_fea = self.get_features(source_rawdata)

        new_d_labels = self.replace_labels(source_dlabels,self.mapping)
        new_d_labels = torch.tensor(new_d_labels).cuda()

        one_hot_d_label = F.one_hot(new_d_labels,n_clusters).to(torch.float32)
        temp_cpu = torch.inverse(torch.diag(one_hot_d_label.sum(axis=0)).cpu() + torch.eye(one_hot_d_label.size()[1]).cpu())
        d_p = torch.matmul(
            temp_cpu.cuda(),
            torch.matmul(one_hot_d_label.T, s_d_fea))

        c_p=[]
        for i in range(n_clusters):#15
            mask = (new_d_labels == i)
            group_fea= s_c_fea[mask]#
            group_label = source_clabels[mask]#
            temp_cpu = torch.inverse(torch.diag(group_label.sum(axis=0)).cpu() + torch.eye(group_label.size()[1]).cpu())
            c_prototype = torch.matmul(
                temp_cpu.cuda(),
                torch.matmul(group_label.T, group_fea))
            c_p.append(c_prototype)

        c_p = torch.stack(c_p,dim=0)
        self.update_alpha(epoch)
        print("α:",self.alpha)
        new_d_p = (1-self.alpha) * self.domain_p + self.alpha * d_p
        new_c_p = (1-self.alpha) * self.class_p + self.alpha * c_p


        return new_d_p.detach(), new_c_p.detach()


    def update_alpha(self, epoch: int,p=2):
        """
        Update the alpha of the prototype update
        """
        max_iter = self.max_iter
        alpha_target = self.lower_threshold
        alpha_0 = self.upper_threshold
        if epoch != 0:
            # 自适应阈值更新
            self.alpha = alpha_target + (alpha_0 - alpha_target) * (1 - epoch / max_iter) ** p


    def domain_label_update(self,source_zip):
        self.eval()
        d_p, c_p = self.get_prototype(source_zip)
        new_dp=self.domain_p

        d_p = d_p / torch.norm(d_p,dim=1,keepdim=True)
        new_dp = new_dp / torch.norm(new_dp,dim=1,keepdim=True)
        domain_predict = torch.matmul(d_p,new_dp.T)
        domain_predict = F.softmax(domain_predict)
        domain_predict = torch.max(domain_predict,dim=1)[1].cpu().detach().numpy()


        mapping = {i: label for i, label in enumerate(domain_predict)}
        return mapping
    def get_parameters(self) -> List[Dict]:
        params = [
            {"params": self.fea_extractor.parameters(), "lr_mult": 1},
            {"params": self.class_fea.parameters(), "lr_mult": 1},
            {"params": self.class_disc.parameters(), "lr_mult": 1},
            {"params": self.domain_fea.parameters(), "lr_mult": 1},
            {"params": self.domain_disc.parameters(), "lr_mult": 1},
        ]
        return params




