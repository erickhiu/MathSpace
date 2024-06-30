
from model_card import ModelCard
from create_model import create_nli_model
from datasets import load_dataset

model_cards = [
    ModelCard("konstantindobler/xlm-roberta-base-focus-german",'xnli','de'),
    ModelCard("konstantindobler/xlm-roberta-base-focus-arabic",'xnli','ar'),
    ModelCard("konstantindobler/xlm-roberta-base-focus-kiswahili",'xnli','sw'),
]

final_results = {}

for model_card in model_cards:

    model_name = model_card.model_name
    dataset = load_dataset(model_card.dataset, model_card.dataset_lang)
    path_to_save = './'

    tokenizer,model,trainer = create_nli_model(model_name,dataset,path_to_save)

    trainer.train()

    results = trainer.evaluate()

    final_results[model_card] = results
    print(final_results)