from haystack.reader.farm import FARMReader
from haystack.document_store import ElasticsearchDocumentStore

reader = FARMReader(model_name_or_path="distilbert-base-uncased-distilled-squad", use_gpu=False)
reader.train(data_dir = 'data', train_filename="dataset.json", use_gpu=False, n_epochs=1, save_dir="my_model")
