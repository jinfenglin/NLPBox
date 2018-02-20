from RNN import RNN
from config import *
from data_prepare import DataPrepare, SENETRawDataBuilder
from feature_extractors import SENETFeaturePipe

if __name__ == '__main__':
    sql_file = os.path.join(PROJECT_ROOT, "..", "Scraper", "data", "term_definitions.db")
    raws = SENETRawDataBuilder().raws
    senet_features = SENETFeaturePipe()
    data = DataPrepare("dataset.pickle", feature_pipe=senet_features, raw_materials=raws, rebuild=True)
    print("Experiment data is ready, size ", len(data.data_set))
    res_file_name = "RNN_result{}.txt".format(len(os.listdir(RESULT_DIR)))
    result_file = os.path.join(RESULT_DIR, res_file_name)
    rnn = RNN(data.get_vec_length())
    rnn.ten_fold_test(data, result_file)
