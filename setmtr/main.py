import random as rand
import warnings
import numpy as np
import torch
from torch.optim import Adam
from torchility import Trainer
from torchility.utils import load_yaml
from criteria import *
from models import Model, ModelAux
from dataset import SetMTrDataset, SetMTrDataLoader
from utils import load_dataset, merge_dict
warnings.simplefilter('ignore')

gpu = 0
if torch.cuda.is_available():
    gpu = 1


config = load_yaml("./config.yaml")
istune = config.istune
config = config[config.task]


if istune:
    seed = 50
    rand.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if gpu:
        torch.cuda.manual_seed_all(seed)


loaded_dataset_lst = load_dataset(config.dataset, config.val_ratio)

# Used to save test results for each data
test_results = []

# Train on a datasets
for i, loaded_dataset in enumerate(loaded_dataset_lst):
    print('\n\n','*'*30, 'ds ', i + 1, '*'*30, '\n')

    _, _, train_sets, val_sets, test_sets, ele_num, _ = loaded_dataset
    padd_idx = ele_num         # padding符号
    mask_idx = ele_num + 1     # mask符号


    # datasets
    train_ds = SetMTrDataset(train_sets, ele_num, mask_idx, config.max_set_size,
                            config.mask_p, config.mask_rand_p, config.mask_keep_p, config.repeat, config.has_element, config.has_feat, aux=True)
    val_ds = SetMTrDataset(val_sets, ele_num, mask_idx, config.max_set_size,
                        config.mask_p, config.mask_rand_p, config.mask_keep_p, config.repeat, config.has_element, config.has_feat, aux=True)


    train_dl = SetMTrDataLoader(train_ds, config.batch_size, padd_idx, config.max_set_size, config.has_element, config.has_feat, shuffle=True, aux=True)
    val_dl = SetMTrDataLoader(val_ds, config.batch_size, padd_idx, config.max_set_size, config.has_element, config.has_feat, shuffle=False, aux=True)
    test_dl = SetMTrDataLoader(test_sets, config.batch_size, padd_idx, config.max_set_size, config.has_element, config.has_feat, shuffle=False, aux=True, test=True)


    # model
    model = Model(ele_num, config.max_set_size, config.feat_dim, config.element_embedding,
                padd_idx, config.has_element, config.has_feat,
                config.encoder_layer, config.decoder_layer, config.model_dim, config.num_heads,
                config.dropout, layer_norm=True, element_query_method=config.element_query_method)
    model_aux = ModelAux(model, config.model_dim)

    metrics = [AuxAcc(), AuxF1(), AuxAUC()]
    if config.has_element:
        metrics.append(ElementReconstructAcc(ele_num, has_feat=config.has_feat, name='eacc', aux=True))
        metrics.append(Jaccard(config.has_feat, name='jcd', aux=True))

    # training
    loss = Loss(config.has_element, config.has_feat, epsilon=0.1)
    loss_aux = LossAux(loss)
    opt = Adam(model_aux.parameters(), lr=config.lr)
    trainer = Trainer(model_aux, loss_aux, opt, config.epochs, metrics=metrics)
    trainer.fit(train_dl, val_dl)

    rst = trainer.test(test_dl)
    test_results.append(rst[0])


for k, vs in merge_dict(test_results).items():
    print(k, '\t', [round(v, 6) for v in vs], '\taverage:', sum(vs)/len(vs))
