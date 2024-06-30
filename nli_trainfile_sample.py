
from create_model import create_nli_model
from datasets import load_dataset


model_name = "konstantindobler/xlm-roberta-base-focus-german"
dataset = load_dataset("xnli", 'de')
path_to_save = './'

tokenizer,model,trainer = create_nli_model(model_name,dataset,path_to_save)

trainer.train()

results = trainer.evaluate()

print(results['eval_accuracy'])
