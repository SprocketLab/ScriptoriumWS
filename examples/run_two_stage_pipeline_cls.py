import logging
import torch
from wrench.dataset import load_dataset
from wrench._logging import LoggingHandler
from wrench.labelmodel import Snorkel, MajorityVoting, MajorityWeightedVoting, DawidSkene, FlyingSquid
from wrench.endmodel import EndClassifierModel

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)

logger = logging.getLogger(__name__)

device = torch.device('cuda:0')

#### Load dataset
dataset_path = '../datasets/'
data = 'youtube'
train_data, valid_data, test_data = load_dataset(
    dataset_path,
    data,
    use_codexLF=True,
    extract_feature=True,
    extract_fn='bert', # extract bert embedding
    model_name='bert-base-cased',
    cache_name='bert'
)
#### ========================================= ####

"""
#### Run label model: Flying Squid
flysquid_label_model = FlyingSquid()
flysquid_label_model.fit(
    dataset_train=train_data,
    dataset_valid=valid_data
)

acc = flysquid_label_model.test(test_data, 'acc')
logger.info(f'Flying Squid label model test acc: {round(acc, 4)}')

f1 = flysquid_label_model.test(test_data, 'f1_binary')
logger.info(f'Flying Squid label model test f1: {round(f1, 4)}')
#### ========================================= ####

#### Run label model: Dawid-Skene
ds_label_model = DawidSkene()
ds_label_model.fit(
    dataset_train=train_data,
    dataset_valid=valid_data
)

acc = ds_label_model.test(test_data, 'acc')
logger.info(f'Dawid-Skene label model test acc: {round(acc, 4)}')

f1 = ds_label_model.test(test_data, 'f1_binary')
logger.info(f'Dawid-Skene label model test f1: {round(f1, 4)}')
#### ========================================= ####
"""

#### Run label model: Majority Weighted Vote
majority_weighted_voter = MajorityWeightedVoting()
majority_weighted_voter.fit(
    dataset_train=train_data,
    dataset_valid=valid_data
)

acc = majority_weighted_voter.test(test_data, 'acc')
logger.info(f'majority weighted voter test acc: {round(acc, 4)}')

f1 = majority_weighted_voter.test(test_data, 'f1_binary')
logger.info(f'majority weighted voter test f1: {round(f1, 4)}')
#### ========================================= ####

#### Run label model: Majority Vote
majority_voter = MajorityVoting()
majority_voter.fit(
    dataset_train=train_data,
    dataset_valid=valid_data
)

acc = majority_voter.test(test_data, 'acc')
logger.info(f'majority voter test acc: {round(acc, 4)}')

f1 = majority_voter.test(test_data, 'f1_binary')
logger.info(f'majority voter test f1: {round(f1, 4)}')
#### ========================================= ####

#### Run label model: Snorkel
snorkel_label_model = Snorkel(
    lr=0.01,
    l2=0.0,
    n_epochs=300
)
snorkel_label_model.fit(
    dataset_train=train_data,
    dataset_valid=valid_data
)

acc = snorkel_label_model.test(test_data, 'acc')
logger.info(f'Snorkel label model test acc: {round(acc, 4)}')

f1 = snorkel_label_model.test(test_data, 'f1_binary')
logger.info(f'Snorkel label model test f1: {round(f1, 4)}')
#### ========================================= ####

#### Filter out uncovered training data
train_data = train_data.get_covered_subset()
aggregated_hard_labels = snorkel_label_model.predict(train_data)
aggregated_soft_labels = snorkel_label_model.predict_proba(train_data)

#### Run end model: MLP
model = EndClassifierModel(
    batch_size=512,
    test_batch_size=512,
    n_steps=10000,
    backbone='MLP',
    optimizer='Adam',
    optimizer_lr=1e-2,
    optimizer_weight_decay=0.0,
)
model.fit(
    dataset_train=train_data,
    y_train=aggregated_soft_labels,
    dataset_valid=valid_data,
    evaluation_step=10,
    metric='acc',
    patience=100,
    device=device
)

acc = model.test(test_data, 'acc')
logger.info(f'end model (MLP) test acc: {round(acc, 4)}')

f1 = model.test(test_data, 'f1_binary')
logger.info(f'end model (MLP) test f1: {round(f1, 4)}')

"""
#### Run end model: BERT
model = EndClassifierModel(
    batch_size=16,
    real_batch_size=16,  # for accumulative gradient update
    test_batch_size=256,
    n_steps=1000,
    backbone='BERT',
    backbone_model_name='bert-base-cased',
    backbone_max_tokens=128,
    backbone_fine_tune_layers=-1, # fine  tune all
    optimizer='AdamW',
    optimizer_lr=5e-5,
    optimizer_weight_decay=0.0,
)
model.fit(
    dataset_train=train_data,
    y_train=aggregated_soft_labels,
    dataset_valid=valid_data,
    evaluation_step=10,
    metric='acc',
    patience=-1,
    device=device
)

acc = model.test(test_data, 'acc')
logger.info(f'end model (BERT) test acc: {acc}')

f1 = model.test(test_data, 'f1_binary')
logger.info(f'end model (BERT) test f1: {f1}')
"""