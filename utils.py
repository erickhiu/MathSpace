
from datasets import load_metric
import numpy as np
import collections
import evaluate
import numpy as np
from tqdm.auto import tqdm

metric_for_NLI = load_metric("accuracy")
metric_for_NER = load_metric("f1")
metric_for_QA = evaluate.load("squad")


def preprocess_function_for_NLI(examples, tokenizer):
    return tokenizer(examples['premise'], examples['hypothesis'], truncation = True, padding = 'max_length', max_length = 128)

def preprocess_function_for_NER(examples, tokenizer):
  total_adjusted_labels = []
  tokenized_samples = tokenizer.batch_encode_plus(examples["tokens"], is_split_into_words = True, padding = 'max_length', truncation = True, max_length = 128)

  for k in range(0, len(tokenized_samples["input_ids"])):
    prev_wid = -1
    word_ids_list = tokenized_samples.word_ids(batch_index=k)
    existing_label_ids = examples["ner_tags"][k]

    i = -1
    adjusted_label_ids = []

    for wid in word_ids_list:
      if(wid is None):
        adjusted_label_ids.append(-100)
      elif(wid!=prev_wid):
        i = i + 1
        adjusted_label_ids.append(existing_label_ids[i])
        prev_wid = wid
      else:
        adjusted_label_ids.append(existing_label_ids[i])

    total_adjusted_labels.append(adjusted_label_ids)

  tokenized_samples["labels"] = total_adjusted_labels

  return tokenized_samples

max_length = 384
stride = 128

def preprocess_training_for_QA(examples,tokenizer):
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=max_length,
        truncation="only_second",
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    offset_mapping = inputs.pop("offset_mapping")
    sample_map = inputs.pop("overflow_to_sample_mapping")
    answers = examples["answers"]
    start_positions = []
    end_positions = []

    for i, offset in enumerate(offset_mapping):
        sample_idx = sample_map[i]
        answer = answers[sample_idx]
        start_char = answer["answer_start"][0]
        end_char = answer["answer_start"][0] + len(answer["text"][0])
        sequence_ids = inputs.sequence_ids(i)

        # Find the start and end of the context
        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1
        context_start = idx
        while sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1

        # If the answer is not fully inside the context, label is (0, 0)
        if offset[context_start][0] > start_char or offset[context_end][1] < end_char:
            start_positions.append(0)
            end_positions.append(0)
        else:
            # Otherwise it's the start and end token positions
            idx = context_start
            while idx <= context_end and offset[idx][0] <= start_char:
                idx += 1
            start_positions.append(idx - 1)

            idx = context_end
            while idx >= context_start and offset[idx][1] >= end_char:
                idx -= 1
            end_positions.append(idx + 1)

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs

def preprocess_validation_for_QA(examples,tokenizer):
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=max_length,
        truncation="only_second",
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_map = inputs.pop("overflow_to_sample_mapping")
    example_ids = []

    for i in range(len(inputs["input_ids"])):
        sample_idx = sample_map[i]
        example_ids.append(examples["id"][sample_idx])

        sequence_ids = inputs.sequence_ids(i)
        offset = inputs["offset_mapping"][i]
        inputs["offset_mapping"][i] = [
            o if sequence_ids[k] == 1 else None for k, o in enumerate(offset)
        ]

    inputs["example_id"] = example_ids
    return inputs




def compute_metrics_for_NLI(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis = -1)
    return metric_for_NLI.compute(predictions = predictions, references = labels)

def compute_metrics_for_NER(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis = 2)

    true_predictions = [
        [p for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [l for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    true_predictions = [item for sublist in true_predictions for item in sublist]
    true_labels = [item for sublist in true_labels for item in sublist]

    result = metric_for_NER.compute(predictions = true_predictions, references = true_labels, average = "micro")

    return {
        "f1": result["f1"]
    }


def create_compute_metric_for_QA(eval_set, examples):
    def compute_metric_for_QA(p):
        predictions, _ = p
        start_logits = predictions[0]
        end_logits = predictions[1]

        example_to_features = collections.defaultdict(list)
        for idx, feature in enumerate(eval_set):
            example_to_features[feature["example_id"]].append(idx)

        n_best = 20
        max_answer_length = 30
        predicted_answers = []

        for example in examples:
            example_id = example["id"]
            context = example["context"]
            answers = []

            for feature_index in example_to_features[example_id]:
                start_logit = start_logits[feature_index]
                end_logit = end_logits[feature_index]
                offsets = eval_set["offset_mapping"][feature_index]

                start_indexes = np.argsort(start_logit)[-1 : -n_best - 1 : -1].tolist()
                end_indexes = np.argsort(end_logit)[-1 : -n_best - 1 : -1].tolist()
                for start_index in start_indexes:
                    for end_index in end_indexes:
                        # Skip answers that are not fully in the context
                        if offsets[start_index] is None or offsets[end_index] is None:
                            continue
                        # Skip answers with a length that is either < 0 or > max_answer_length.
                        if (
                            end_index < start_index
                            or end_index - start_index + 1 > max_answer_length
                        ):
                            continue

                        answers.append(
                            {
                                "text": context[offsets[start_index][0] : offsets[end_index][1]],
                                "logit_score": start_logit[start_index] + end_logit[end_index],
                            }
                        )

            if len(answers) > 0:
                best_answer = max(answers, key=lambda x: x["logit_score"])
                predicted_answers.append(
                    {"id": str(example_id), "prediction_text": best_answer["text"]}
                )
            else:
                predicted_answers.append({"id": str(example_id), "prediction_text": ""})

        theoretical_answers = [
            {"id": str(ex["id"]), "answers": ex["answers"]} for ex in examples
        ]
        
        result = metric_for_QA.compute(predictions=predicted_answers, references=theoretical_answers)
        return result
    
    return compute_metric_for_QA