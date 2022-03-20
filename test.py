#%%
import tensorflow as tf
import tensorflow.keras as keras
from src.encoder_decoder.training_loop import TrainingLoop
from src.utils.dataset_creators import QADataset
from tqdm import tqdm
path = 'final_dataset_clean_v2 .tsv'

dataset_creator = QADataset(path)
# new_data = dataset_creator.create_false_positives(0.5, 0.1)
data, lang_tokenizer = dataset_creator.call_post(0.4, 0.1, 128)


for (batch, data_dict) in tqdm(enumerate(data.take(1))):
    print(batch, data_dict)
# optimizer = keras.optimizers.Adam()
# training_loop = TrainingLoop(dataset_creator, optimizer, D = 14, frac=0.5)


# %%
