from RNN import RNN
from config import *
from data_prepare import DataPrepare, SENETRawDataBuilder, PairBuilder
from feature_extractors import SENETFeaturePipe
import logging, sys

if __name__ == '__main__':
    try:
        cur_node_partition = int(sys.argv[1])
        total_partition = int(sys.argv[2])
    except Exception as e:
        total_partition = 1
        cur_node_partition = 1
    version = "v0.03_all"
    sql_file = os.path.join(PROJECT_ROOT, "..", "SENET", "data", "term_definitions.db")
    logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler()])
    logger = logging.getLogger(__name__)

    expension_file_path = os.path.join(SENET_DATA, "vocab", "expension_on_fly.txt")
    pair_builder = PairBuilder(expension_file_path, cur_node_partition, total_partition)
    data_builder = SENETRawDataBuilder(sql_file, pair_builder=pair_builder)
    raws = data_builder.raws
    documents = data_builder.documents
    senet_features = SENETFeaturePipe(documents)
    pickle_name = "classify_dataset_" + version + "_" + str(cur_node_partition) + ".pickle"
    data_set_pickle = os.path.join(FEATURE_VEC_DIR, pickle_name)
    data = DataPrepare(data_set_pickle, feature_pipe=senet_features,
                       raw_materials=raws, rebuild=True)
