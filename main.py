from transformers import XLMRobertaTokenizerFast, XLMRobertaForSequenceClassification
from datasets import load_dataset
from sklearn.metrics import accuracy_score

NUM_TEST_CASES = 80

# load tokenizer and model weights
tokenizer = XLMRobertaTokenizerFast.from_pretrained('s-nlp/xlmr_formality_classifier')
model = XLMRobertaForSequenceClassification.from_pretrained('s-nlp/xlmr_formality_classifier')

id2formality = {0: "formal", 1: "informal"}

formality_dataset = load_dataset("osyvokon/pavlick-formality-scores", split="test").shuffle(2025).select(
    range(NUM_TEST_CASES))


def preprocess_function(dataset_item):
    if dataset_item["avg_score"] > 0:
        dataset_item["formality"] = "formal"
    else:
        dataset_item["formality"] = "informal"
    return dataset_item


data = formality_dataset.map(preprocess_function)

texts = data["sentence"]
formality = data["formality"]

# prepare the input
encoding = tokenizer(
    texts,
    add_special_tokens=True,
    return_token_type_ids=True,
    truncation=True,
    padding="max_length",
    return_tensors="pt",
)

# inference
output = model(**encoding)

formality_scores = [
    {id2formality[idx]: score for idx, score in enumerate(text_scores.tolist())}
    for text_scores in output.logits.softmax(dim=1)
]


def set_label(output_item):
    if output_item["formal"] > output_item["informal"]:
        return "formal"
    else:
        return "informal"


# set label according to scores
results = []
for item in formality_scores:
    results.append(set_label(item))

# calculate matches
accuracy = accuracy_score(results, formality)

print("Accuracy:", accuracy)
