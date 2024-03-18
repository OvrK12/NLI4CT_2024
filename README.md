# NLI4CT_2024
This repository contains my implementation for the [NLI4CT](https://sites.google.com/view/nli4ct/) task at SemEval 2024. The dataset, which was published together with the task, contains clinical trial reports (CTRs) and annotated statements. The task of the model is to decide if the statement follows from given information from up to two CTRs (see example below). For full details, see the [dataset paper](https://aclanthology.org/2023.emnlp-main.1041.pdf).

![image](https://github.com/OvrK12/NLI4CT_2024/assets/92592126/9fef46b3-60b3-4b08-8df8-1f0e23f74e28)


Image credit: [Julien et al., 2023](https://aclanthology.org/2023.emnlp-main.1041.pdf)


## Overview
To run the code you first have to execute the *preprocess_data.py* script which will create a *data* folder in which preprocessed json files are located for the respective data splits (train, val, test). These preprocessed json files are used as input for the implemented models. The *data_exploration.ipynb* notebook contains code which analyzes token lengths, label distribution, etc. of the dataset.

This repository contains the implementation of two approaches. Firstly, a **FLAN-T5 model** that is finetuned to the dataset of the task, and secondly, a **multi-agent approach**.
## Finetuned FLAN-T5
Recent research suggests that small, finetuned models (e.g. Phi2) can outperform larger general models in a zeroshot setting. We use the *huggingface* library to finetune a FLAN-T5
models on the provided dataset. The FLAN-T5 model was originally developed by Google and is available in different parameter sizes (small, base, large, xl and xxl). The *t5_finetuning* notebook contains the code to finetune the model on the dataset. The *t5_inference* script runs the evaluation of the finetuned model. The *openAi_test* notebook can be used to generate results from OpenAIs GPT models in a zeroshot setting (OpenAI API key required). The images below shows the explored hyperparameter space with the best performing parameters written in bold and the results (for a FLAN-T5 base model). We can see that the finetuned model considerably outperforms the base FLAN-T5 model and performs performs comparably well to the GPT 3.5 model, despite having far less parameters. The GPT4 model performs considerably better than all other models.

![image](https://github.com/OvrK12/NLI4CT_2024/assets/92592126/7d1cc96e-db6e-4720-b57f-ce806cf46b81)
![image](https://github.com/OvrK12/NLI4CT_2024/assets/92592126/50c36d48-2cc5-4186-a786-970974b88340)


## Multi-agent Approach
In recent research, multi-agent approaches have shown promising results on reasoning tasks. [Du et al., 2023](https://composable-models.github.io/llm_debate/) introduced a "debate" approach of multiple language models which significantly improved the performance on a range of quantitative and commonsense reasoning tasks compared to the output of just a single model. In this implementation, we aim to replicate these results using open-source models on the NLI4CT task.

Each multi-agent "conversation" consists of at least one round and is structured as follows: In the first round, all agents receive prompts and generate a zeroshot answer. In the subsequent rounds, all agents receive the answers of the other agents from the previous round as an input prompt and are asked to reconsider their answer based on this. By exposing agents to different reasoning paths, we aim to improve the overall performance of the system. This structure is equivalent to the implementation of [Du et al., 2023](https://composable-models.github.io/llm_debate/).

To use the code from this repository follow these instructions:
- install [ollama](https://ollama.ai/download). As of now, ollama is only available for Linux and Mac. However, you can also use a WSL instance installed on Windows.
- install [litellm](https://litellm.vercel.app/docs/)
- download the required LLMs using ollama. For this run "ollama run <model>" (models used in this implementation: orca2:13b, wizard-math:13b, medllama2:latest)
- run "litellm --model ollama/<model>" to start a model on a proxy server. The first model is always started on port 8000. Ports for additional models are randomly assigned
- now run *multi_agent.ipynb*. You have to adjust the ports of the models, as they are randomly assigned by litellm

  The image below shows the result for the multi-agent approach

  ![image](https://github.com/OvrK12/NLI4CT_2024/assets/92592126/5ea6de37-3be8-4887-a2c4-86ea00bbb920)

Evaluating the results from the multi-agent conversations is not straightforward, as the used models are finetuned for reasoning tasks and therefore produce a lot of explaining text, even when prompted to just reply with a single word. In particular, Orca2 and Wizardmath produce lenghty outputs. However, they follow a structure in which they explain their reasoning and then produce a string with a final answer (e.g. "Final Answer: Contradiction"). Based on this, we extract the final answer from a model by matching a regular expression. There are cases in which the models do not want to decide on either entailment or contradiction (e.g. "Final Answer: Not enough information"). This is due to the fact that the models were finetuned on NLI tasks that contain such a third label. These instances could not be captured by the regular expression and were therefore counted as contradiction. Furthermore, there are some instances in which the models do not follow the structure of the prompt and produce answer strings, such as "Final Answer: The statement is true". Due to the multitude of possible answer phrases, these instances are not captured by the regular expression and are also counted as contradiction. These limitations have to be considered in the interpretation of the results.

The final answer to a statement is extracted by taking the majority vote of the models after the final round of the conversation.
After three rounds the multi-agent approach barely outperforms the TF-IDF baseline. However, the results also show that performance increases significantly with each round, highlighting the potential performance of this approach.

