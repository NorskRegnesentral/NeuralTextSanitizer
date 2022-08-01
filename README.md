# NeuralTextSanitizer
Text sanitization with explicit measures of privacy risk.

To run, first please:
* Download the three models from [this link](https://drive.google.com/drive/folders/1p9znczAIruZKvUxY0hLRy5YXyj0SfOYk?usp=sharing) and place them in the SampleData folder
* Install google-ortools with ```python -m pip install --upgrade --user ortools```
* Install transformers from Hugging Face: `pip install transformers`
* Install intervaltree: `pip install intervaltree`
* Install dill: `pip3 install dill`

The input is a file containing json objects to be sanitized. See *sample2.json* and *sample.json* in the SampleData folder for an example input.

| Field  | Description | |
| ------------- | ------------- | ------------- |
| text  | The text to be sanitized  | required |
| target  | The individual to be protected in the text | required |
| annotations| Manual annotated start and end offsets, and semantic label of PII in the text | optional |

To run the whole pipeline:
* ```python sanitize.py```

The output is a json file containing the masking decisions of each module of the pipeline.
