from datetime import time
from typing import Optional

from torch.nn import init
from torch.optim import Optimizer
from MATL_DC_framework import *
from Pairwise_Learning_Module import *
from utils import *
import random
import time
import warnings
from StepwiseLR_GRL import StepwiseLR_GRL
warnings.filterwarnings("ignore")
print(torch.__version__)
print(torch.version.cuda)
print(torch.backends.cudnn.version())
print("use or no：", torch.cuda.is_available())  #
print("number of GPU：", torch.cuda.device_count())  #
print("Version of CUDA：", torch.version.cuda)  #
print("index of GPU：", torch.cuda.current_device())  #
print("Name of GPU：", torch.cuda.get_device_name(0))  #

def setup_seed(seed):  # setup the random seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def weigth_init(m):  ## model parameter intialization
    if isinstance(m, nn.Conv2d):
        init.xavier_uniform_(m.weight.data)
        init.constant_(m.bias.data, 0.3)
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm1d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        m.weight.data.normal_(0, 0.03)
        #        torch.nn.init.kaiming_normal_(m.weight.data,a=0,mode='fan_in',nonlinearity='relu')
        m.bias.data.zero_()


def get_generated_targets(disentangler_model, pl_model, labels_s):  # Get generated labels by threshold
    """
    The similarity relation matrix of the source domain labels and the pairwise matrix of the target domain are obtained
    Args:
        disentangler_model:
        pl_model:
        labels_s:

    Returns:
        sim_matrix：
        sim_matrix_target：

    """
    with torch.no_grad():
        disentangler_model.eval()
        pl_model.eval()
        sim_matrix = pl_model.get_cos_similarity_distance(labels_s)
        return sim_matrix


def train(model, pl_model, source_data_loader, epoch, max_epoch,
          lr_scheduler, source_zip, batch_size,n_cluster, optimizer):
    model.train()
    pl_model.train()
    t = 3394 * 14 // batch_size  # (3394/batch_size)
    one_epoch_loss = 0
    one_epoch_dann_loss = 0
    bce_loss = nn.BCELoss()
    ce_loss = nn.CrossEntropyLoss()
    source_data_iter = enumerate(source_data_loader)
    tt = time.time()


    for i in range(t):
        model.train()
        pl_model.train()
        _, s_data_list = next(source_data_iter)
        # load data
        s_data = s_data_list[0].to(device="cuda", dtype=torch.float32)

        s_c_labels = s_data_list[1].to(device="cuda", dtype=torch.float32) #one-hot
        s_d_labels = s_data_list[2].to(device="cuda", dtype=torch.float32) #one-hot

        s_result = model(s_data)
        rij_s = get_generated_targets(model, pl_model, s_c_labels)
        s_c_features = s_result["class_fea"]
        s_d_features = s_result["domain_fea"]

        s_c_predict = pl_model.predict_class(s_c_features, s_d_features, model.domain_p, model.class_p)
        features_pairwise_s = pl_model.get_cos_similarity_distance(s_c_predict)

        #------------------------ point-wise

        eta = 0.00001
        bce_losss = -(torch.log(s_c_predict + eta) * s_c_labels) - (1 - s_c_labels) * torch.log(1 - s_c_predict + eta)
        supervised_pointwise_loss = torch.mean(bce_losss)

        #------------------------ pairwise
        eta = 0.00001
        source_bce = -(torch.log(features_pairwise_s + eta) * rij_s) - (1 - rij_s) * torch.log(1 - features_pairwise_s + eta)
        supervised_pairwise_loss = torch.mean(source_bce)
        #
        # Modify the domain label according to the mapping
        s_d_labels = model.replace_labels(s_d_labels,model.mapping)
        # one-hot
        s_d_labels = F.one_hot(torch.tensor(s_d_labels), n_cluster).to(torch.float32).cuda()
        # Dann Loss

        dann_loss = (bce_loss(s_result["s_c"], s_c_labels) + bce_loss(s_result["s_dc"], s_c_labels)
                     + bce_loss(s_result["s_d"], s_d_labels) + bce_loss(s_result["s_cd"], s_d_labels))

        delta = parameter_pl["delta"]
        gamma = delta * epoch / max_epoch
        loss = supervised_pairwise_loss + dann_loss
        optimizer.zero_grad()
        loss.backward()

        one_epoch_loss = one_epoch_loss + loss.item()
        one_epoch_dann_loss = one_epoch_dann_loss + dann_loss.item()
        optimizer.step()
    print(f'coast:{time.time() - tt:.4f}s')
    pl_model.update_threshold(epoch)
    lr_scheduler.step()
    avg_epoch_loss = one_epoch_loss / t
    avg_epoch_dann_loss = one_epoch_dann_loss / t
    return avg_epoch_loss, avg_epoch_dann_loss


def main(parameter, parameter_pl, subject):
    best_acc = 0.0
    best_nmi = 0.0
    n_cluster = parameter["domain_num"]
    max_iter = parameter["max_iter"]
    learning_rata = 1e-3
    train_dataset, test_dataset = split_domain(subject)  #
    train_dataset = PackedDataset(train_dataset)
    test_dataset = PackedDataset(test_dataset)
    batch_size = 3394 * 2

    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size, shuffle=True)

    source_zip = partition_into_three(train_dataset.datalist)
    source_rawdata = source_zip[0]  # data
    source_clabels = source_zip[1]  # class label
    source_dlabels = source_zip[2]  # domain label

    target_rawdata, target_clabels, _ = partition_into_three(test_dataset.datalist)
    setup_seed(parameter['seed'])

    # MATL-DC
    model = DemoModel(parameter).to("cuda")
    # pairwise learning
    pl_model = Pairwise_learning(parameter_pl).to("cuda")
    #init
    model.apply(weigth_init)
    pl_model.apply(weigth_init)

    optimizer = torch.optim.RMSprop(model.get_parameters() + pl_model.get_parameters(), lr=1e-3,
                                        weight_decay=1e-5)
     # Update the learning rate dynamically
    lr_scheduler = StepwiseLR_GRL(optimizer, init_lr=0.001, gamma=10, decay_rate=0.75, max_iter=max_iter)

    loss_list = []
    dann_loss_list = []
    target_acc_list = np.zeros(max_iter)
    target_nmi_list = np.zeros(max_iter)
    source_acc_list = np.zeros(max_iter)
    source_nmi_list = np.zeros(max_iter)


    model.domain_p, model.class_p, model.mapping = model.prototype_init_v2(source_zip, n_cluster)
    best_epoch = 0
    for epoch in range(0, max_iter):
        #Train model
        one_epoch_loss, one_epoch_dann_loss = train(model, pl_model, train_dataloader,
                                                        epoch, max_iter, lr_scheduler, source_zip, batch_size,
                                                        n_cluster,optimizer)

        if (epoch + 1) == 20:
            _, _, model.mapping = model.prototype_init_v2(source_zip, n_cluster)

        # It's just getting the results
        source_acc, source_nmi = pl_model.cluster_label_update(model, source_rawdata, source_clabels,
                                                                   source_zip)
        print('epoch:', epoch, ' loss: ', one_epoch_loss, ' dann loss:', one_epoch_dann_loss, ' source_acc:',
                  source_acc)

        target_acc, target_nmi = pl_model.target_domain_evaluation(model, target_rawdata,
                                                                       target_clabels, source_zip)

        target_acc_list[epoch] = target_acc
        source_acc_list[epoch] = source_acc
        target_nmi_list[epoch] = target_nmi
        source_nmi_list[epoch] = source_nmi
        print('target_acc:', target_acc, '  traget_NMI:', target_nmi)
        #save loss
        loss_list.append(one_epoch_loss)
        dann_loss_list.append(one_epoch_dann_loss)
        #save the best model
        if target_acc > best_acc:
            weight_name1 = savepath + '\\' + '_model_weight_init_' + str(subject) + '.pth'
            weight_name2 = savepath + '\\' + '_pl_model_weight_init_' + str(subject) + '.pth'
            torch.save(model, weight_name1)
            torch.save(pl_model, weight_name2)
            #  save Prototype representation
            torch.save(model.domain_p, savepath + '\\' + '_domain_p_' + str(subject) + '.pth')
            torch.save(model.class_p, savepath + '\\' + '_class_p_' + str(subject) + '.pth')
            best_nmi = target_nmi
            best_acc = target_acc
            best_epoch = epoch

            # updata Prototype
        model.domain_p, model.class_p = model.prototype_update(source_zip, n_cluster, epoch)
            # model.mapping = model.domain_label_update(source_zip)

    return best_acc, best_nmi,dann_loss_list, loss_list, best_epoch


if __name__ == '__main__':

    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    parameter = {
        "fea_len": 310,
        "fea_h1": 64,
        "fea_h2": 64,
        "domain_fea_h1": 64,
        "domain_fea_h2": 64,
        "class_fea_h1": 64,
        "class_fea_h2": 64,
        "domain_dis_h2": 64,
        "domain_num": 4,
        "class_dis_h2": 64,
        "class_num": 3,
        "alpha": 0.8,
        "upper_threshold": 0.8,
        "lower_threshold": 0.2,
        "max_iter": 150,
        "seed":114514,
        "savepath": 'H:\\Feature_class_P_Research\\MATL-DC_github\\' + \
                                    time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime()) + '_' \
                                    + 'SEED'
    }

    parameter_pl = {
        "max_iter": 100,
        "upper_threshold": 0.9,
        "lower_threshold": 0.5,
        "num_of_class": 3,
        "delta": 2
    }

    all_sub_best_acc = []
    all_sub_best_nmi = []
    for subject in range(1, 16):
        savepath = parameter['savepath']+ '_seed' + str(parameter['seed'])
        os.makedirs(savepath)
        os.chdir(savepath)
        sub_best_acc, sub_best_nmi, dann_loss_list, loss_list, best_epoch  = main(parameter,parameter_pl ,subject)

        all_sub_best_acc.append(sub_best_acc)
        all_sub_best_nmi.append(sub_best_nmi)

        loss_list = np.array(loss_list)
        dann_loss_list = np.array(dann_loss_list)
        np.save(savepath + '\\' + '_train_loss_curve_' + str(subject) + '.npy',
                loss_list)
        np.save(savepath + '\\' + '_dannloss_curve_' + str(subject) + '.npy',
                dann_loss_list)
        np.save(savepath + '\\' + '_best_acc_list_' + str(subject) + '.npy',
                all_sub_best_acc)

        # save beat acc and std
        result_path = savepath + '\\Acc' + str(sub_best_acc) + '_nmi'+ str(sub_best_nmi)+ '_spoch'+str(best_epoch)+'_Sub' + str(subject) + '.txt'
        with open(result_path, 'w') as f:
            f.write('Best target Acc: \n' + str(all_sub_best_acc) +
                    '\nBest target Acc_mean: \n' + str(np.mean(all_sub_best_acc)) +
                    "\nStd: \n" + str(all_sub_best_nmi) +
                    '\n\ninfo:' + '\n' + str(parameter) +
                    '\n\nchange:' +
                    """
                    """)

        print('-------------------subject ', subject, '   over------------------------')
        print(all_sub_best_acc)
