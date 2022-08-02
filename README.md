# NeuralTextSanitizer
Text sanitization with explicit measures of privacy risk.

For ```python>=3.7```:
* Download the three models from [this link](https://drive.google.com/drive/folders/1p9znczAIruZKvUxY0hLRy5YXyj0SfOYk?usp=sharing) and place them in the SampleData folder
* ```python -m pip install -r requirements.txt```

The input should be a file containing the text(s) to be sanitized. See *sample2.json* and *sample.json* in the SampleData folder for an example input.

| Field  | Description | |
| ------------- | ------------- | ------------- |
| text  | The text to be sanitized  | required |
| target  | The individual to be protected in the text | required |
| annotations| Manual annotated start and end offsets, and semantic label of PII in the text | optional |

To run the whole pipeline, provide the path to an input file as follows:
* ```python sanitize.py SampleData/sample2.json```

The output is a json file containing the masking decisions of each module of the pipeline. More specifically:

| Field  | Description |
| ------------- | ------------- |
| opt_decision  | The masking decisions after the Optimization Algorithm |
| PII  | Personally Identifiable Information in the text| 
| blacklist1| The masking decisions of the Language Model |
| blacklist2| The masking decisions of the Web Query model |
| blacklist3| The masking decisions of the Mask Classifier model |