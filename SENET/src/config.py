import os
import logging
import sys

PROJECT_ROOT = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..')
RESULT_DIR = os.path.join(PROJECT_ROOT, "result")
SENET_DATA = os.path.join(PROJECT_ROOT, "data")
VOCAB_DIR = os.path.join(SENET_DATA, "vocab")

RNN_MODEL_DIR = os.path.join(SENET_DATA, "rnn_model")
RNN_ENCODER_PATH = "rnn_encoder.pickle"

FNN_MODEL_DIR = os.path.join(SENET_DATA, "fnn_model")
FNN_ENCODER_PATH = "fnn_encoder.pickle"

LOGS = os.path.join(PROJECT_ROOT, "logs")
FEATURE_VEC_DIR = os.path.join(SENET_DATA, "feature_vectors")

def write_csv(res, writer):
    res = [str(x) for x in res]
    content = ",".join(res)
    writer.write(content + "\n")
