
from transformers import Trainer
from transformers import AutoTokenizer
from transformers import TrainingArguments
from transformers import (AutoModelForSequenceClassification, 
                          AutoModelForTokenClassification, 
                          AutoModelForQuestionAnswering)

from utils import (preprocess_function_for_NLI, 
                   compute_metrics_for_NLI, 
                   preprocess_function_for_NER,
                   compute_metrics_for_NER,
                   preprocess_training_for_QA,
                   preprocess_validation_for_QA,
                   create_compute_metric_for_QA)

def create_nli_model(model_name, dataset, path_to_save):

    """
    Creates an NLI model

    Parameters
    ----------
    model_name
    dataset 
    path_to_save

    Returns
    -------
    Tokenizer, model and trainer for NLI model specified by parameters
    """

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels = 3)

    encoded = dataset.map(lambda examples: preprocess_function_for_NLI(examples, tokenizer), batched = True)
    train = encoded["train"].shuffle(seed=42).select(range(25000))


    training_args = TrainingArguments(
        output_dir=path_to_save,
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=128,
        per_device_eval_batch_size=128,
        num_train_epochs=0,
        weight_decay=0.01,
        logging_steps=196,
        save_strategy="epoch",
        save_total_limit=1,
        adam_epsilon=1e-8,
        adam_beta1=0.9,
        adam_beta2=0.999,
        lr_scheduler_type='linear',
        warmup_ratio=0.1
    )

    trainer_xnli = Trainer(
        model = model,
        args = training_args,
        train_dataset = train,
        eval_dataset = encoded["validation"],
        compute_metrics = compute_metrics_for_NLI,
    )

    return tokenizer,model,trainer_xnli


def create_ner_model(model_name, dataset, path_to_save):

    """
    Creates an NER model

    Parameters
    ----------
    model_name
    dataset 
    path_to_save

    Returns
    -------
    Tokenizer, model and trainer for NER model specified by parameters
    """

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels = 7)

    encoded = dataset.map(lambda examples: preprocess_function_for_NER(examples, tokenizer), batched = True)
    train = encoded["train"].shuffle(seed=42).select(range(len(encoded["train"])))


    training_args = TrainingArguments(
        output_dir = path_to_save,
        evaluation_strategy = "epoch",
        learning_rate = 2e-5,
        per_device_train_batch_size = 32,
        per_device_eval_batch_size = 32,
        num_train_epochs = 0,
        weight_decay = 0.01,
        logging_steps = 1250,
        save_strategy = "epoch",
        save_total_limit = 1,
        adam_epsilon = 1e-8,
        adam_beta1 = 0.9,
        adam_beta2 = 0.999,
        lr_scheduler_type = 'linear',
        warmup_ratio = 0.1
    )

    trainer = Trainer(
        model = model,
        args = training_args,
        train_dataset = train,
        eval_dataset = encoded["validation"],
        compute_metrics = compute_metrics_for_NER
    )

    return tokenizer,model,trainer



def create_qa_model(model_name, dataset, path_to_save):

    """
    Creates a QA model

    Parameters
    ----------
    model_name
    dataset 
    path_to_save

    Returns
    -------
    Tokenizer, model and trainer for QA model specified by parameters
    """


    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)

    encoded_for_model = dataset.map(lambda examples: preprocess_training_for_QA(examples, tokenizer), batched = True, remove_columns=dataset["train"].column_names)
    train = encoded_for_model["train"].shuffle(seed=42).select(range(len(encoded_for_model["train"])))
    test = encoded_for_model["test"]

    eval_set = dataset['test'].map(lambda examples: preprocess_validation_for_QA(examples, tokenizer), batched = True, remove_columns=dataset["test"].column_names)
    metric = create_compute_metric_for_QA(eval_set,dataset['test'])

    training_args = TrainingArguments(
        output_dir=path_to_save,
        eval_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=128,
        per_device_eval_batch_size=128,
        num_train_epochs=0,
        weight_decay=0.01,
        logging_steps=196,
        save_strategy="epoch",
        save_total_limit=1,
        adam_epsilon=1e-8,
        adam_beta1=0.9,
        adam_beta2=0.999,
        lr_scheduler_type='linear',
        warmup_ratio=0.1
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train,
        eval_dataset=test,
        compute_metrics=metric
    )

    return tokenizer,model,trainer
