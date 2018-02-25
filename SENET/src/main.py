from RNN import RNN
from config import *
from data_prepare import DataPrepare, SENETRawDataBuilder, PairBuilder
from feature_extractors import SENETFeaturePipe
import logging, sys

if __name__ == '__main__':
    try:
        mode = sys.argv[1]
    except Exception as e:
        mode = "classify"
    # sql_file = os.path.join(PROJECT_ROOT, "..", "Scraper", "data", "term_definitions.db")
    sql_file = os.path.join(PROJECT_ROOT, "..", "SENET", "data", "term_definitions.db")
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    logger = logging.getLogger(__name__)
    version = "v0.02_all"
    logger.info("RNN Experiemtn in mode:{} Model Version: {}".format(mode, version))
    model_name = "rnn_v{}.ckpt".format(str(version))
    model_path = os.path.join(RNN_MODEL_DIR, model_name)
    if mode == "ten_fold":
        data_builder = SENETRawDataBuilder(sql_file)
        raws = data_builder.raws
        documents = data_builder.documents
        senet_features = SENETFeaturePipe(documents)
        data = DataPrepare("dataset.pickle", feature_pipe=senet_features, raw_materials=raws,
                           rebuild=True)
        print("Experiment data is ready, size ", len(data.data_set))
        res_file_name = "RNN_result{}.txt".format(len(os.listdir(RESULT_DIR)))
        result_file = os.path.join(RESULT_DIR, res_file_name)
        rnn = RNN(data.get_vec_length(), model_path, RNN_ENCODER_PATH)
        rnn.ten_fold_test(data, result_file)
    elif mode == "classify":
        expension_file_path = os.path.join(SENET_DATA, "vocab", "expension_debug.txt")
        pair_builder = PairBuilder(expension_list_txt=expension_file_path)
        data_builder = SENETRawDataBuilder(sql_file, pair_builder=pair_builder)
        raws = data_builder.raws
        documents = data_builder.documents
        senet_features = SENETFeaturePipe(documents)
        data = DataPrepare("classify_dataset.pickle", feature_pipe=senet_features,
                           raw_materials=raws, rebuild=True)
        rnn = RNN(data.get_vec_length(), model_path, RNN_ENCODER_PATH)
        res, encoder = rnn.classify(data.all())
        classify_res_path = os.path.join(RESULT_DIR, "classify.txt")
        rnn.write_classify_res(classify_res_path, res, encoder)
