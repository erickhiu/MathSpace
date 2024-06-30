
from create_model import create_qa_model
from datasets import load_dataset

model_name = "konstantindobler/xlm-roberta-base-focus-german"
dataset = load_dataset("deepset/germanquad")
# dataset['test'] = dataset['test'].select(range(100))

path_to_save = './'

tokenizer,model,trainer = create_qa_model(model_name,dataset,path_to_save)

trainer.train()

results = trainer.evaluate()

print(results)
