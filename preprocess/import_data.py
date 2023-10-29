import h5py
from tqdm import tqdm

data_dir = "../data/hdf5_data"
train_sets = [
    "ply_data_train0.h5",
    "ply_data_train1.h5",
    "ply_data_train2.h5",
    "ply_data_train3.h5",
    "ply_data_train4.h5",
    "ply_data_train5.h5"
]
val_sets = ["ply_data_val0.h5"]
test_sets = ["ply_data_test0.h5", "ply_data_test1.h5"]

def load_data():
    sample_path = data_dir + '/' + train_sets[0]
    with h5py.File(sample_path, "r") as f:
        for key in f.keys():
            print(key, f[key])


if __name__ == '__main__':
    load_data()
