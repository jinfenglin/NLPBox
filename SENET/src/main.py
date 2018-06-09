from FeedForwardNN import FNN
from RNN import RNN
from config import *
from data_prepare import DataPrepare, SENETRawDataBuilder, PairBuilder
from feature_extractors import SENETFeaturePipe
import logging, sys

if __name__ == '__main__':
    try:
        mode = sys.argv[1]
    except Exception as e:
        mode = "ten_fold"
    try:
        cur_node_partition = int(sys.argv[2])
        total_partition = int(sys.argv[3])
    except Exception as e:
        total_partition = 1
        cur_node_partition = 1
    version = "v0.03_all"
    sql_file = os.path.join(PROJECT_ROOT, "..", "SENET", "data", "term_definitions.db")
    log_fileanme = os.path.join(LOGS, version + "_" + str(cur_node_partition) + ".log")
    logging.basicConfig(level=logging.INFO, handlers=[logging.FileHandler(log_fileanme), logging.StreamHandler()])
    logger = logging.getLogger(__name__)
    logger.info("RNN Experiemtn in mode:{} Model Version: {}".format(mode, version))
    model_name = "fnn_v{}.ckpt".format(str(version))
    model_path = os.path.join(FNN_MODEL_DIR, model_name)
    if mode == "ten_fold":
        data_builder = SENETRawDataBuilder(sql_file)
        raws = data_builder.raws
        documents = data_builder.documents
        senet_features = SENETFeaturePipe(documents)
        data = DataPrepare("dataset_origin.pickle", feature_pipe=senet_features, raw_materials=raws,
                           rebuild=False)
        print("Experiment data is ready, size ", len(data.data_set))
        fnn = FNN(data.get_vec_length(), model_path, FNN_ENCODER_PATH)
        fnn.ten_fold_test(data)
    elif mode == "classify":
        expension_file_path = os.path.join(SENET_DATA, "vocab", "expension.txt")
        pair_builder = PairBuilder(expension_file_path, cur_node_partition, total_partition)
        data_builder = SENETRawDataBuilder(sql_file, pair_builder=pair_builder)
        raws = data_builder.raws
        documents = data_builder.documents
        senet_features = SENETFeaturePipe(documents)
        data_set_pickle = "classify_dataset" + version + "_" + str(cur_node_partition) + ".pickle"
        data = DataPrepare(data_set_pickle, feature_pipe=senet_features,
                           raw_materials=raws, rebuild=False)
        fnn = FNN(data.get_vec_length(), model_path, FNN_ENCODER_PATH)
        res, encoder = fnn.classify(data.all())
        result_file_name = "classify" + "_" + version + "_" + str(cur_node_partition) + ".txt"
        classify_res_path = os.path.join(RESULT_DIR, result_file_name)
        fnn.write_classify_res(classify_res_path, res, encoder)
