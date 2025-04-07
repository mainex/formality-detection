# Evaluation of Formality Detection Model

## Overview

This project evaluates a formality detection model `s-nlp/xlmr_formality_classifier` (Dementieva _et al._, 2023) on the
dataset `osyvokon/pavlick-formality-scores` (Syvokon, 2023).

## How to Run

Before you begin, make sure you have all the necessary libraries installed:

```
pip install -r requirements.txt
```
Run the script to reproduce the results:

```
python main.py
```

## Tools and Techniques

### Formality Detection Model

This project evaluates the model presented by Dementieva _et al._ (2023). This model is based on XLM-Roberta and
fine-tuned on formality classification dataset XFORMAL (Briakou _et al._, 2021).

As classification model, it evaluates how formal is the text and returns scores for "formal" and "informal"
labels. `set_label` function in [main.py](main.py) chooses the label with the maxim score for future evaluation.

### Evaluation Dataset

The model was evaluated on the dataset from Hugging Face created by Oleksiy Syvokon (2023). He combined two datasets:

- the news and blog data collected by Shibamouli (2015);
- the news, blogs, email ad QA forums data collected by Pavlick _et al._ (2016).

Oleksiy also improved the data by detokenization of answers and emails for better readability.

Every item in the dataset has three columns: domain, average score and sentence. In this research only average score and
sentence was used.
`sentence` column contains texts from different resources. `avg_score` is a score that shows how formal is the sentence.
It ranges from -3 to 3, where most formal sentences are indicated with the highest score.

`preprocess_function` in the [main.py](main.py) script sets the labels to the items for future evaluation.

### Evaluation Metrics

As evaluation metric was `accuracy_score` from scikit-learn. It calculates the proportion of matches between two arrays.
The arrays formed from the output model and from the test dataset were compared.

## Implementation Challenges

In the original dataset, formality had not just a label, but a score. I initially wanted to normalize these values ​​and
compare them with the percentage predicted by the model. However, firstly, these numbers have slightly different
meanings, since the first one shows the degree of formality, and the second one shows the probability that the statement
is formal. I am not sure that comparing them is a good metric. Secondly, comparing numbers requires a more complex
metric for evaluation, which is not so easy to find.

## Results

The model was tested on 80 random sentences from the testing dataset. Its accuracy was 0.7, meaning it gave the
expected answer 70% of the time. This is a small size for a dataset, usually 3000-30000 data points are recommended (
Koshute et al., 2021). Unfortunately, this is the laptop limit, which could not be increased. To reproduce the results
with a different number, you should change the value of the `NUM_TEST_CASES` variable at the beginning of
the [main.py](main.py) file. It is possible that with a different dataset size the results will be significantly
different.

## References

Briakou, E., Lu, D., Zhang, K., Tetreault, J. (2021) 'Olá, Bonjour, Salve! XFORMAL: A Benchmark for Multilingual
Formality Style Transfer', in _Proceedings of the 2021 Conference of the North American Chapter of the Association for
Computational Linguistics: Human Language Technologies_, pp. 3199–3216. Association for Computational
Linguistics. Available at: https://aclanthology.org/2021.naacl-main.256/ (Accessed: 7 April 2025).

Dementieva, D., Babakov, N., Panchenko, A. (2023) 'Detecting Text Formality: A Study of Text
Classification Approaches', in _Proceedings of the 14th International Conference on Recent Advances in Natural Language
Processing_, pp. 274–284, Varna, Bulgaria. Shoumen, Bulgaria: INCOMA Ltd. Available
at: https://aclanthology.org/2023.ranlp-1.31/ (Accessed: 7 April 2025).

Koshute, P., Zook, J., McCulloh, I. (2021) 'Recommending Training Set Sizes for Classification', _arXiv preprint_.
Available at: https://doi.org/10.48550/arXiv.2102.09382 (Accessed: 7 April 2025).

Pavlick, E., Tetreault, J. (2016) 'An Empirical Analysis of Formality in Online Communication', _Transactions of the
Association for Computational Linguistics_, 4, pp. 61-74. Available at: https://aclanthology.org/Q16-1005/ (Accessed: 7
April 2025).

Shibamouli, L. (2015) 'SQUINKY! A Corpus of Sentence-level Formality, Informativeness, and Implicature', _arXiv_.
Available at: https://doi.org/10.48550/arXiv.1506.02306 (Accessed: 7 April 2025).

Syvokon, O. (2023) Pavlick Formality Scores Dataset. Available
at: https://huggingface.co/datasets/osyvokon/pavlick-formality-scores (Accessed: 7 April 2025).