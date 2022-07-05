# NeuralTextSanitizer
Text sanitization with explicit measures of privacy risk.

To run, first please:
* Download the tree models from [this link](https://drive.google.com/drive/folders/1p9znczAIruZKvUxY0hLRy5YXyj0SfOYk?usp=sharing) and place in SampleData folder
* Install google-ortools with ```python -m pip install --upgrade --user ortools```
* 
The input should be a json file. See *sample.json* and *sample2.json* for an example input.

To run the whole pipeline:
* ```python test.py```

The output is a json file containing the masking decisions of each module of the pipeline.
