# NLI4CT_2024
This repository contains my implementation for the [NLI4CT](https://sites.google.com/view/nli4ct/) task at SemEval 2024. The dataset, which was published together with the task, contains clinical trial reports (CTRs) and annotated statements. The task of the model is to decide if the statement follows from given information from up to two CTRs (see example below). For full details, see the [dataset paper](https://aclanthology.org/2023.emnlp-main.1041.pdf).

![image](https://github.com/OvrK12/NLI4CT_2024/assets/92592126/9fef46b3-60b3-4b08-8df8-1f0e23f74e28)


Image credit: [Julien et al., 2023](https://aclanthology.org/2023.emnlp-main.1041.pdf)


## Overview
To run the code you first have to execute the *preprocess_data.py* script which will create a *data* folder in which preprocessed json files are located for the respective data splits (train, val, test). These preprocessed json files are used as input for the implemented models. The *data_exploration.ipynb* notebook contains code which analyzes token lengths, label distribution, etc. of the dataset.

This repository contains the implementation of two approaches. Firstly, a **FLAN-T5 model** that is finetuned to the dataset of the task, and secondly, a **multi-agent approach**.
### Finetuned FLAN-T5
Recent research suggests that small, finetuned models (e.g. Phi2) can outperform larger general models in a zeroshot setting. We use the *huggingface* library to finetune a FLAN-T5
models on the provided dataset. The FLAN-T5 model was originally developed by Google and is available in different parameter sizes (small, base, large, xl and xxl). The *t5_finetuning* notebook contains the code to finetune the model on the dataset. The *t5_inference* script runs the evaluation of the finetuned model. The *openAi_test* notebook can be used to generate results from OpenAIs GPT models in a zeroshot setting (OpenAI API key required). The images below shows the explored hyperparameter space with the best performing parameters written in bold and the results (for a FLAN-T5 base model). We can see that the finetuned model considerably outperforms the base FLAN-T5 model and performs performs comparably well to the GPT 3.5 model, despite having far less parameters. The GPT4 model performs considerably better than all other models.

![image](https://github.com/OvrK12/NLI4CT_2024/assets/92592126/7d1cc96e-db6e-4720-b57f-ce806cf46b81)
![image](https://github.com/OvrK12/NLI4CT_2024/assets/92592126/50c36d48-2cc5-4186-a786-970974b88340)


### Multi-agent Approach
In recent research, multi-agent approaches have shown promising results on reasoning tasks. [Du et al., 2023](https://composable-models.github.io/llm_debate/) introduced a "debate" approach of multiple language models which significantly improved the performance on a range of quantitative and commonsense reasoning tasks compared to the output of just a single model. In this implementation, we aim to replicate these results using open-source models on the NLI4CT task.
