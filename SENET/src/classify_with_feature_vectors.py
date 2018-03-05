from RNN import RNN
from config import *
from data_prepare import DataPrepare
import re

if __name__ == '__main__':
    try:
        cur_node_partition = int(sys.argv[1])
        total_partition = int(sys.argv[2])
    except Exception as e:
        total_partition = 1
        cur_node_partition = 1
    version = "v0.03_all"
    model_name = "rnn_v{}.ckpt".format(str(version))
    model_path = os.path.join(RNN_MODEL_DIR, model_name)
    file_names = os.listdir(FEATURE_VEC_DIR)
    for name in file_names:
        print("Processing file:{}".format(name))
        name_parts = re.split("[_\.]", name)
        partition_num = int(name_parts[-2])
        data_set_pickle = os.path.join(FEATURE_VEC_DIR, name)
        if partition_num % total_partition == (cur_node_partition - 1):
            try:
                data = DataPrepare(data_set_pickle, feature_pipe=None, raw_materials=None, rebuild=False)
                rnn = RNN(data.get_vec_length(), model_path, RNN_ENCODER_PATH)
                res, encoder = rnn.classify(data.all())
                result_file_name = name.replace(".pickle", ".res")
                classify_res_path = os.path.join(RESULT_DIR, result_file_name)
                rnn.write_classify_res(classify_res_path, res, encoder)
            except Exception as e:
                print(e)
    print("finished")
