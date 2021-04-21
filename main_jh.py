import argparse
import glob
import os
import pickle
import random
import time
import warnings
from pathlib import Path

import numpy as np
# Import libraries
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
import wandb
from data_loader import load_and_cache_examples
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm
from trainer import Trainer

# Ignore Warnings
warnings.filterwarnings(action='ignore')

# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'device : {device}')
print(torch.cuda.get_device_properties(device))

# Set ROOT_PATH
ROOT_PATH = os.getcwd()
print(f'ROOT_PATH : {ROOT_PATH}')

# Set wandb
wandb.login()
CFG = wandb.config
# % env

WANDB_PROJECT = 'P2_new'
WANDB_LOG_MODEL = True
WANDB_SILENT = True
WANDB_WATCH = all


def main(CFG, args):
    random.seed(CFG.seed)
    np.random.seed(CFG.seed)
    torch.manual_seed(CFG.seed)
    torch.cuda.manual_seed_all(CFG.seed)
    os.environ["PYTHONHASHSEED"] = str(CFG.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    #Set Tokenizer and Model
    tokenizer = transformers.AutoTokenizer.from_pretrained(CFG.MODEL_NAME)
    tokenizer.add_tokens(['<e1>', '</e1>', '<e2>', '</e2>'], special_tokens=True)
    # model.resize_token_embeddings(tokenizer.vocab_size + 4)
    print(tokenizer)
    train_dataset = load_and_cache_examples(args, tokenizer, mode="train")
    test_dataset = load_and_cache_examples(args, tokenizer, mode="test")

    print(train_dataset)
    print(test_dataset)

    trainer = Trainer(CFG, args, train_dataset=train_dataset, test_dataset=test_dataset)

    if args.do_train:
        trainer.train()

    if args.do_eval:
        trainer.load_model()
        trainer.evaluate("test")

if __name__ == "__main__":
    # Set Experiment
    CFG.name = '365-entity-token-sun16-special'
    CFG.tag = ['roberta-large', 'max-length-250', 'entity-token', 'add-UNK-text']
    CFG.seed = 2021
    CFG.MODEL_NAME = "xlm-roberta-large"
    CFG.lr = 3e-5
    CFG.batch_size = 16
    CFG.epochs = 10
    CFG.tokenizer_max_length = 365
    CFG.warmup_steps = 500
    CFG.weight_decay = 0.001
    CFG.lr_scheduler_type = 'linear'
    CFG.set_valid_dataset = True
    CFG.fp16 = True
    CFG.label_smoothing_factor = 0.0
    # CFG.add_UNK_text_to_vocab = True
    run = wandb.init(project='P2_new', group=CFG.MODEL_NAME, name=CFG.name, tags=CFG.tag, config=CFG)

    # # Set Directory
    # if not os.path.isdir(f'custom_data/{CFG.name}') :
    #     os.chdir(os.path.join(ROOT_PATH, 'custom_data'))
    #     os.mkdir(f'{CFG.name}')
    #     os.chdir(ROOT_PATH)

    # Set Seed

    parser = argparse.ArgumentParser()

    parser.add_argument("--task", default="semeval", type=str, help="The name of the task to train")
    parser.add_argument(
        "--data_dir",
        default="./data",
        type=str,
        help="The input data dir. Should contain the .tsv files (or other data files) for the task.",
    )
    parser.add_argument("--model_dir", default="./model", type=str, help="Path to model")
    parser.add_argument(
        "--eval_dir",
        default="./eval",
        type=str,
        help="Evaluation script, result directory",
    )
    parser.add_argument("--train_file", default="RBERT-train.tsv", type=str, help="Train file")
    parser.add_argument("--test_file", default="RBERT-val.tsv", type=str, help="Test file")
    parser.add_argument("--label_file", default="label_new.txt", type=str, help="Label file")

    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default=CFG.MODEL_NAME,
        help="Model Name or Path",
    )

    parser.add_argument("--seed", type=int, default=CFG.seed, help="random seed for initialization")
    parser.add_argument("--train_batch_size", default=CFG.batch_size, type=int, help="Batch size for training.")
    parser.add_argument("--eval_batch_size", default=32, type=int, help="Batch size for evaluation.")
    parser.add_argument(
        "--max_seq_len",
        default=CFG.tokenizer_max_length, #384
        type=int,
        help="The maximum total input sequence length after tokenization.",
    )
    parser.add_argument(
        "--learning_rate",
        default=CFG.lr,
        type=float,
        help="The initial learning rate for Adam.",
    )
    parser.add_argument(
        "--num_train_epochs",
        default=CFG.epochs,
        type=float,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument("--weight_decay", default=CFG.weight_decay, type=float, help="Weight decay if we apply some.")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument("--warmup_steps", default=CFG.warmup_steps, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument(
        "--dropout_rate",
        default=0.1,
        type=float,
        help="Dropout for fully-connected layers",
    )
    parser.add_argument("--lr_scheduler_type", default=CFG.lr_scheduler_type, type=str, help="Linear")
    parser.add_argument("--set_valid_dataset", default=CFG.set_valid_dataset, type=bool, help="set_valid_dataset")
    parser.add_argument("--fp16", default=CFG.fp16, type=bool, help="fp16")
    parser.add_argument("--label_smoothing_factor", default=CFG.label_smoothing_factor, type=float, help="label_smoothing_factor")


    parser.add_argument("--logging_steps", type=int, default=300, help="Log every X updates steps.")
    parser.add_argument(
        "--save_steps",
        type=int,
        default=280,
        help="Save checkpoint every X updates steps.",
    )

    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the test set.")
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument(
        "--add_sep_token",
        action="store_true",
        help="Add [SEP] token at the end of the sentence",
    )

    args = parser.parse_args()

    main(CFG, args)