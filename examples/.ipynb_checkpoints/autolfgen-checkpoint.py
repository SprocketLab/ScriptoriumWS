import fire
import logging
import torch
from wrench.dataset import load_dataset
from wrench._logging import LoggingHandler
from wrench.labelmodel import Snorkel, MajorityVoting, MajorityWeightedVoting, DawidSkene, FlyingSquid
from wrench.endmodel import EndClassifierModel, LogRegModel

#### Label Model Abbreviations: 
## Snorkel: Snorkel
## Majority Voting: MV
## Weighted Majority Voting: WMV
## Dawid-Skene: DS
## FlyingSquid: FS

#### End Model Abbreviations:
## MLP: MLP
## Logistic Regression: LR
## BERT: BERT

def LM_FS(train_data, valid_data, test_data):
    
    #### Run label model: Flying Squid
    flysquid_label_model = FlyingSquid()
    flysquid_label_model.fit(
        dataset_train=train_data,
        dataset_valid=valid_data
    )

    acc = flysquid_label_model.test(test_data, 'acc')
    f1 = flysquid_label_model.test(test_data, 'f1_binary')
    #### ========================================= ####
    
    return acc, f1, flysquid_label_model

def LM_DS(train_data, valid_data, test_data):
    
    #### Run label model: Dawid-Skene
    ds_label_model = DawidSkene()
    ds_label_model.fit(
        dataset_train=train_data,
        dataset_valid=valid_data
    )

    acc = ds_label_model.test(test_data, 'acc')
    f1 = ds_label_model.test(test_data, 'f1_binary')
    #### ========================================= ####
    
    return acc, f1, ds_label_model

def LM_WMV(train_data, valid_data, test_data):
    
    #### Run label model: Weighted Majority Voting
    majority_weighted_voter = MajorityWeightedVoting()
    majority_weighted_voter.fit(
        dataset_train=train_data,
        dataset_valid=valid_data
    )

    acc = majority_weighted_voter.test(test_data, 'acc')
    f1 = majority_weighted_voter.test(test_data, 'f1_binary')
    #### ========================================= ####
    
    return acc, f1, majority_weighted_voter

def LM_MV(train_data, valid_data, test_data):
        
    #### Run label model: Majority Vote
    majority_voter = MajorityVoting()
    majority_voter.fit(
        dataset_train=train_data,
        dataset_valid=valid_data
    )

    acc = majority_voter.test(test_data, 'acc')
    f1 = majority_voter.test(test_data, 'f1_binary')
    #### ========================================= ####
    
    return acc, f1, majority_voter

def LM_Snorkel(train_data, valid_data, test_data):
        
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
    f1 = snorkel_label_model.test(test_data, 'f1_binary')
    #### ========================================= ####
    
    return acc, f1, snorkel_label_model

def EM_MLP(train_data, soft_labels, valid_data, test_data, device):
    
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
        y_train=soft_labels,
        dataset_valid=valid_data,
        evaluation_step=10,
        metric='acc',
        patience=100,
        device=device
    )

    acc = model.test(test_data, 'acc')
    f1 = model.test(test_data, 'f1_binary')
    #### ========================================= ####
    
    return acc, f1, model

def EM_LR(train_data, soft_labels, valid_data, test_data, device):

    #### Run end model: Logistic Regression
    model = LogRegModel(
        batch_size=512,
        test_batch_size=512,
        n_steps=10000
    )
    model.fit(
        dataset_train=train_data,
        y_train=soft_labels,
        dataset_valid=valid_data,
        evaluation_step=10,
        metric='acc',
        patience=100,
        device=device
    )
    
    acc = model.test(test_data, 'acc')
    f1 = model.test(test_data, 'f1_binary')
    #### ========================================= ####
    
    return acc, f1, model


def main(dataset="youtube", codexLF=False, prompt_type=None, LM="Snorkel", EM="Stop"):

    ## Basic Config ##
    if codexLF == True: 
        LF_type = prompt_type + "_codex"
    else: 
        LF_type = "human"
        
    
    logging.basicConfig(
        filename="../exp_log/" + dataset + "/" + LF_type + "_" + LM + "_" + EM + ".txt",
        filemode="w",
        format='%(asctime)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=logging.INFO
    )
    logger = logging.getLogger(__name__)
    dataset_path = '../datasets/'
    device = torch.device('cuda:0')
    bert_feature = True
    if dataset == "census": bert_feature = False
    logger.info(f'Dataset: {dataset}')
    ##################
    
    ## Load Dataset ##
    train_data, valid_data, test_data = load_dataset(
        dataset_path,
        dataset,
        use_codexLF=codexLF,
        pt_type=prompt_type,
        extract_feature=bert_feature,
        extract_fn='bert', # extract bert embedding
        model_name='bert-base-cased',
        cache_name='bert'
    )
    ##################
    
    ## Train Label Model ##
    if LM == "Snorkel": 
        lm_acc, lm_f1, lm = LM_Snorkel(train_data, valid_data, test_data)
        logger.info(f'Snorkel label model test acc: {round(lm_acc, 4)}')
        logger.info(f'Snorkel label model test f1: {round(lm_f1, 4)}')
    elif LM == "MV": 
        lm_acc, lm_f1, lm = LM_MV(train_data, valid_data, test_data)
        logger.info(f'majority voter test acc: {round(lm_acc, 4)}')
        logger.info(f'majority voter test f1: {round(lm_f1, 4)}')
    elif LM == "WMV":
        lm_acc, lm_f1, lm = LM_WMV(train_data, valid_data, test_data)
        logger.info(f'weighted majority voter test acc: {round(lm_acc, 4)}')
        logger.info(f'weighted majority voter test f1: {round(lm_f1, 4)}')
    elif LM == "DS": 
        lm_acc, lm_f1, lm = LM_DS(train_data, valid_data, test_data)
        logger.info(f'Dawid-Skene label model test acc: {round(lm_acc, 4)}')
        logger.info(f'Dawid-Skene label model test f1: {round(lm_f1, 4)}')
    elif LM == "FS": 
        lm_acc, lm_f1, lm = LM_FS(train_data, valid_data, test_data)
        logger.info(f'Flying Squid label model test acc: {round(lm_acc, 4)}')
        logger.info(f'Flying Squid label model test f1: {round(lm_f1, 4)}')
    else: 
        logger.info(f'Cannot find this LM')
    ##################
    
    ## Filter out uncovered training data ##
    train_data_covered = train_data.get_covered_subset()
    lm_cov = len(train_data_covered) / len(train_data)
    logger.info(f'label model test cov: {round(lm_cov, 4)}')
    
    aggregated_hard_labels = lm.predict(train_data_covered)
    aggregated_soft_labels = lm.predict_proba(train_data_covered)
    ##################
    
    ## Train End Model ##
    if EM == "MLP":
        em_acc, em_f1, em = EM_MLP(train_data_covered, aggregated_soft_labels, valid_data, test_data, device)
        logger.info(f'End model (MLP) test acc: {round(em_acc, 4)}')
        logger.info(f'End model (MLP) test f1: {round(em_f1, 4)}')
    elif EM == "LR":
        em_acc, em_f1, em = EM_LR(train_data_covered, aggregated_soft_labels, valid_data, test_data, device)
        logger.info(f'End model (Logistic Regression) test acc: {round(em_acc, 4)}')
        logger.info(f'End model (Logistic Regression) test f1: {round(em_f1, 4)}')
    elif EM == "Stop":
        logger.info(f'Not going to run end model')
    else:
        logger.info(f'Cannot find this EM')
        
    return None
  
if __name__ == '__main__':
    fire.Fire(main)