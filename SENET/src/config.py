import os
import logging
import sys

PROJECT_ROOT = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..')
RESULT_DIR = os.path.join(PROJECT_ROOT, "result")
SENET_DATA = os.path.join(PROJECT_ROOT, "data")
VOCAB_DIR = os.path.join(SENET_DATA, "vocab")

RNN_MODEL_DIR = os.path.join(SENET_DATA, "model")
RNN_ENCODER_PATH = "encoder.pickle"
