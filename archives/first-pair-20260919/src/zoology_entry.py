"""Pinned upstream code with CPU allocation and Windows seed integer fixes."""
import os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
UPSTREAM=ROOT/'third_party/zoology-1ad20d1'
os.environ['WANDB_MODE']='disabled'
sys.path.insert(0,str(UPSTREAM))
from zoology.experiments.basic_examples.basic import config as BASIC_CONFIG
from zoology.model import LanguageModel
from zoology.data.utils import prepare_data
from zoology.data.multiquery_ar import multiquery_ar
from zoology.train import Trainer
from zoology.utils import set_determinism

def configuration(sequence_length=64):
    if sequence_length < 16 or sequence_length >= 256 or sequence_length % 2:
        raise ValueError('This fixed-vocabulary MQAR baseline requires an even length from 16 to 254')
    config=BASIC_CONFIG.model_copy(deep=True)
    for segment in config.data.train_configs + config.data.test_configs:
        segment.input_seq_len=sequence_length
    # Absolute positions must cover the new input length; train from scratch.
    config.model.max_position_embeddings=sequence_length
    return config
