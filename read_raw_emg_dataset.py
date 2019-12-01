import numpy as np
import pandas as pd

file_name = "/home/tug18152/MyoArmbandDataset/PyTorchImplementation/formatted_datasets/saved_pre_training_dataset_spectrogram.npy"

dataset = np.load(file_name, allow_pickle = True)

df = pd.DataFrame(data=dataset, index = ['a','b','c'])

print(df)


