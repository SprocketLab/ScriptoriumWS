"""
This code is to conduct an experiment about the integration of humanLF and codexLF to improve LF coverage and accuracy.
"""

import fire
import torch
import logging
import numpy as np
from tqdm import tqdm
from wrench.dataset import load_dataset
from wrench._logging import LoggingHandler
from wrench.labelmodel import Snorkel, MajorityVoting, MajorityWeightedVoting, DawidSkene, FlyingSquid
from wrench.endmodel import EndClassifierModel, LogRegModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

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

def main(dataset="youtube", codexLF=False, prompt_type="basic_mission", LM="Snorkel", EM="LR"):

    ## Basic Config ##
    LF_type = prompt_type + "_integration"
        
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
    
    ## Train Label Model (for datapoints labeled by humanLF) ##
    if LM == "Snorkel": 
        _, _, lm = LM_Snorkel(train_data, valid_data, test_data)
    elif LM == "MV": 
        _, _, lm = LM_MV(train_data, valid_data, test_data)
    elif LM == "WMV":
        _, _, lm = LM_WMV(train_data, valid_data, test_data)
    elif LM == "DS": 
        _, _, lm = LM_DS(train_data, valid_data, test_data)
    elif LM == "FS": 
        _, _, lm = LM_FS(train_data, valid_data, test_data)
    ##################
    
    ## Filter out uncovered training datapoints ##
    train_data_uncovered = train_data.get_uncovered_subset()
    logger.info(f'# of uncovered training datapoint: {len(train_data_uncovered)}')
    train_data_covered = train_data.get_covered_subset()
    logger.info(f'# of covered training datapoint: {len(train_data_covered)}')
    lm_cov = len(train_data_covered) / len(train_data)
    logger.info(f'label model coverage: {round(lm_cov, 4)}')
    ##################
    
    test_X = test_data.features
    test_y = test_data.labels
    aggregated_hard_labels_humanLF = lm.predict(train_data_covered)
    # aggregated_soft_labels_humanLF = lm.predict_proba(train_data_covered)
    
    ## Train End Model (for humanLF only) ##
    """
    train_X_humanLF = train_data_covered.features
    
    lgr_humanLF = LogisticRegression(C=1e4, solver="lbfgs", max_iter=2000)
    lgr_humanLF.fit(X=train_X_humanLF, y=aggregated_hard_labels_humanLF)
    prediction_humanLF = lgr_humanLF.predict(test_X)
    
    logger.info(f'End Model (for humanLF only) test acc: {round(accuracy_score(test_y, prediction_humanLF), 4)}')
    if dataset != "agnews":
        logger.info(f'End Model (for humanLF only) test f1: {round(f1_score(test_y, prediction_humanLF, average="binary"), 4)}')
    logger.info(f'End Model (for humanLF only) cov: {round(len(train_X_humanLF) / len(train_data), 4)}')
    """
    ##################
    
    ## Grab thouse uncovered datapoints back ##
    if dataset == "imdb":
        if prompt_type == "basic_mission":
            from datasets.imdb.LFs.basic_mission import LF1, LF2, LF3, LF4, LF5, LF6
            LF_list = [LF1, LF2, LF3, LF4, LF5, LF6]
        elif prompt_type == "task_description":
            from datasets.imdb.LFs.task_description import LF1, LF2, LF3, LF4, LF5
            LF_list = [LF1, LF2, LF3, LF4, LF5]
        elif prompt_type == "human_heuristic":
            from datasets.imdb.LFs.human_heuristic import LF1, LF2, LF3, LF4, LF5, LF6
            LF_list = [LF1, LF2, LF3, LF4, LF5, LF6]
        elif prompt_type == "human_label_function":
            from datasets.imdb.LFs.human_label_function import LF1, LF2, LF3, LF4, LF5
            LF_list = [LF1, LF2, LF3, LF4, LF5]
        elif prompt_type == "data_example":
            from datasets.imdb.LFs.data_example import LF1, LF2, LF3, LF4, LF5
            LF_list = [LF1, LF2, LF3, LF4, LF5]
            
        for i, example in tqdm(enumerate(train_data_uncovered.examples), total=len(train_data_uncovered.examples)):
            data = example["text"]
            weak_labels = []
            for LF in LF_list:
                weak_label = LF.is_positive(data)
                if weak_label == None:
                    weak_label = -1
                weak_labels.append(weak_label)
            train_data_uncovered.weak_labels[i] = weak_labels

    if dataset == "sms":
        if prompt_type == "basic_mission":
            from datasets.sms.LFs.basic_mission import LF1, LF2, LF3, LF4, LF5, LF6, LF7, LF8
            LF_list = [LF1, LF2, LF3, LF4, LF5, LF6, LF7, LF8]
        elif prompt_type == "task_description":
            from datasets.sms.LFs.task_description import LF1, LF2, LF3, LF4, LF5, LF6, LF7, LF8
            LF_list = [LF1, LF2, LF3, LF4, LF5, LF6, LF7, LF8]
        elif prompt_type == "human_heuristic":
            from datasets.sms.LFs.human_heuristic import LF1, LF2, LF3, LF4, LF5, LF6, LF7, LF8, LF9
            LF_list = [LF1, LF2, LF3, LF4, LF5, LF6, LF7, LF8, LF9]
        elif prompt_type == "human_label_function":
            from datasets.sms.LFs.human_label_function import LF1, LF2, LF3, LF4, LF5, LF6, LF7, LF8
            LF_list = [LF1, LF2, LF3, LF4, LF5, LF6, LF7, LF8]
        elif prompt_type == "data_example":
            from datasets.sms.LFs.data_example import LF1, LF2, LF3, LF4, LF5, LF6, LF7, LF8
            LF_list = [LF1, LF2, LF3, LF4, LF5, LF6, LF7, LF8]

        for i, example in tqdm(enumerate(train_data_uncovered.examples), total=len(train_data_uncovered.examples)):
            data = example["text"]
            weak_labels = []
            for LF in LF_list:
                weak_label = LF.is_spam(data)
                if weak_label == None:
                    weak_label = -1
                weak_labels.append(weak_label)
            train_data_uncovered.weak_labels[i] = weak_labels
    
    if dataset == "youtube":
        if prompt_type == "basic_mission":
            from datasets.youtube.LFs.basic_mission import LF1, LF2, LF3, LF4, LF5, LF6, LF7, LF8, LF9
            LF_list = [LF1, LF2, LF3, LF4, LF5, LF6, LF7, LF8, LF9]
        elif prompt_type == "task_description":
            from datasets.youtube.LFs.task_description import LF1, LF2, LF3, LF4, LF5, LF6, LF7, LF8, LF9
            LF_list = [LF1, LF2, LF3, LF4, LF5, LF6, LF7, LF8, LF9]
        elif prompt_type == "human_heuristic":
            from datasets.youtube.LFs.human_heuristic import LF1, LF2, LF3, LF4, LF5, LF6, LF7, LF8
            LF_list = [LF1, LF2, LF3, LF4, LF5, LF6, LF7, LF8]
        elif prompt_type == "human_label_function":
            from datasets.youtube.LFs.human_label_function import LF1, LF2, LF3, LF4, LF5, LF6
            LF_list = [LF1, LF2, LF3, LF4, LF5, LF6]
        elif prompt_type == "data_example":
            from datasets.youtube.LFs.data_example import LF1, LF2, LF3, LF4, LF5, LF6, LF7, LF8
            LF_list = [LF1, LF2, LF3, LF4, LF5, LF6, LF7, LF8]

        for i, example in tqdm(enumerate(train_data_uncovered.examples), total=len(train_data_uncovered.examples)):
            data = example["text"]
            weak_labels = []
            for LF in LF_list:
                weak_label = LF.is_spam(data)
                if weak_label == None:
                    weak_label = -1
                weak_labels.append(weak_label)
            train_data_uncovered.weak_labels[i] = weak_labels
    
    if dataset == "spouse":
        if prompt_type == "basic_mission":
            from datasets.spouse.LFs.basic_mission import LF1, LF2, LF3, LF4, LF5, LF6, LF7, LF8
            LF_list = [LF1, LF2, LF3, LF4, LF5, LF6, LF7, LF8]
        elif prompt_type == "task_description":
            from datasets.spouse.LFs.task_description import LF1, LF2, LF3, LF4, LF5, LF6, LF7, LF8, LF9
            LF_list = [LF1, LF2, LF3, LF4, LF5, LF6, LF7, LF8, LF9]
        elif prompt_type == "human_heuristic":
            from datasets.spouse.LFs.human_heuristic import LF1, LF2, LF3, LF4, LF5, LF6, LF7, LF8
            LF_list = [LF1, LF2, LF3, LF4, LF5, LF6, LF7, LF8]
        elif prompt_type == "human_label_function":
            from datasets.spouse.LFs.human_label_function import LF1, LF2, LF3, LF4, LF5
            LF_list = [LF1, LF2, LF3, LF4, LF5]
        elif prompt_type == "data_example":
            from datasets.spouse.LFs.data_example import LF1, LF2, LF3, LF4, LF5, LF6, LF7, LF8
            LF_list = [LF1, LF2, LF3, LF4, LF5, LF6, LF7, LF8]

        for i, example in tqdm(enumerate(train_data_uncovered.examples), total=len(train_data_uncovered.examples)):
            data = example["text"]
            weak_labels = []
            for LF in LF_list:
                weak_label = LF.is_spouse(data)
                if weak_label == None:
                    weak_label = -1
                weak_labels.append(weak_label)
            train_data_uncovered.weak_labels[i] = weak_labels
    
    if dataset == "yelp":
        if prompt_type == "basic_mission":
            from datasets.yelp.LFs.basic_mission import LF1, LF2, LF3, LF4, LF5, LF6, LF7, LF8, LF9, LF10, LF11
            LF_list = [LF1, LF2, LF3, LF4, LF5, LF6, LF7, LF8, LF9, LF10, LF11]
        elif prompt_type == "task_description":
            from datasets.yelp.LFs.task_description import LF1, LF2, LF3, LF4, LF5, LF6, LF7
            LF_list = [LF1, LF2, LF3, LF4, LF5, LF6, LF7]
        elif prompt_type == "human_heuristic":
            from datasets.yelp.LFs.human_heuristic import LF1, LF2, LF4, LF5, LF6
            LF_list = [LF1, LF2, LF4, LF5, LF6]
        elif prompt_type == "human_label_function":
            from datasets.yelp.LFs.human_label_function import LF1, LF3, LF4, LF5, LF6
            LF_list = [LF1, LF3, LF4, LF5, LF6]
        elif prompt_type == "data_example":
            from datasets.yelp.LFs.data_example import LF1, LF2, LF3, LF4, LF6, LF7
            LF_list = [LF1, LF2, LF3, LF4, LF6, LF7]

        for i, example in tqdm(enumerate(train_data_uncovered.examples), total=len(train_data_uncovered.examples)):
            data = example["text"]
            weak_labels = []
            for LF in LF_list:
                weak_label = LF.restaurant_rating(data)
                if weak_label == None:
                    weak_label = -1
                weak_labels.append(weak_label)
            train_data_uncovered.weak_labels[i] = weak_labels
            
    if dataset == "agnews":
        if prompt_type == "basic_mission":
            from datasets.agnews.LFs.basic_mission import LF1, LF2, LF3, LF4, LF5, LF6, LF7, LF8
            LF_list = [LF1, LF2, LF3, LF4, LF5, LF6, LF7, LF8]
        elif prompt_type == "task_description":
            from datasets.agnews.LFs.task_description import LF1, LF2, LF3, LF4
            LF_list = [LF1, LF2, LF3, LF4]
        elif prompt_type == "human_heuristic":
            from datasets.agnews.LFs.human_heuristic import LF1, LF2, LF3, LF4
            LF_list = [LF1, LF2, LF3, LF4]
        elif prompt_type == "human_label_function":
            from datasets.agnews.LFs.human_label_function import LF1, LF2, LF3, LF4, LF5, LF6, LF7, LF8
            LF_list = [LF1, LF2, LF3, LF4, LF5, LF6, LF7, LF8]
        elif prompt_type == "data_example":
            from datasets.agnews.LFs.data_example import LF1, LF2, LF3, LF4, LF5
            LF_list = [LF1, LF2, LF3, LF4, LF5]

        for i, example in tqdm(enumerate(train_data_uncovered.examples), total=len(train_data_uncovered.examples)):
            data = example["text"]
            weak_labels = []
            for LF in LF_list:
                weak_label = LF.news_category(data)
                if weak_label != -1 and weak_label != None:
                    weak_label -= 1
                if weak_label == None:
                    weak_label = -1
                weak_labels.append(weak_label)
            train_data_uncovered.weak_labels[i] = weak_labels
    
    logger.info(f'Load {dataset} {prompt_type} {len(LF_list)} codex LFs')
    ##################
    
    ## Train Label Model (for datapoints labeled by codexLF) ##
    if LM == "Snorkel": 
        lm2 = Snorkel(lr=0.01, l2=0.0, n_epochs=300)
    elif LM == "MV": 
        lm2 = MajorityVoting()
    elif LM == "WMV":
        lm2 = MajorityWeightedVoting()
    elif LM == "DS": 
        lm2 = DawidSkene()
    elif LM == "FS": 
        lm2 = FlyingSquid()
    lm2.fit(
        dataset_train=train_data_uncovered,
        dataset_valid=None
    )
    ##################
    
    ## Train End Model (integration) ##
    aggregated_hard_labels_codex = lm2.predict(train_data_uncovered)
    # aggregated_soft_labels_codex = lm2.predict_proba(train_data_uncovered)

    train_X_integration = np.vstack((train_data_covered.features, train_data_uncovered.features))
    train_y_integration = np.concatenate((aggregated_hard_labels_humanLF, aggregated_hard_labels_codex))

    lgr_integration = LogisticRegression(C=1e4, solver="lbfgs", max_iter=2000)
    lgr_integration.fit(X=train_X_integration, y=train_y_integration)
    prediction_integration = lgr_integration.predict(test_X)
    
    logger.info(f'End Model (integration) test acc: {round(accuracy_score(test_y, prediction_integration), 4)}')
    if dataset != "agnews":
        logger.info(f'End Model (integration) test f1: {round(f1_score(test_y, prediction_integration, average="binary"), 4)}')
    logger.info(f'End Model (integration) cov: {round(len(train_X_integration) / len(train_data), 4)}')
    ##################
    
    return None
  
if __name__ == '__main__':
    fire.Fire(main)