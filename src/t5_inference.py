import torch

from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import load_dataset
from sklearn.metrics import precision_score, recall_score, f1_score

# HF id for model/tokenizer
MODEL_ID = "OvrK12/t5Seq2SeqBase1027"
# location of the data folder
DATA_LOCATION = "../data/"
# the data split on which you want to evaluate the model
DATA_SPLIT = 'val'
 
class T5Inference(): 
    def __init__(self, model_id) -> None:
        # init model and tokenizer
        self._model = AutoModelForSeq2SeqLM.from_pretrained(model_id, torch_dtype="auto", trust_remote_code=True)
        torch.set_default_device("cuda")
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model = self._model.to(self._device)
        self._model.eval()

        self._tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    # Load Dataset
    def load_and_split_dataset(self,datasplit: str):
        """Loads a datasplit from the given dataset

        Args:
            datasplit (str): the datasplit to be loaded. Defaults to val dataset

        Returns:
            the specified datasplit
        """
        try:
            return load_dataset('json', data_files={datasplit: [f'{DATA_LOCATION}{datasplit}.json']})[datasplit]
        except:
            # handle the exception
            print("Unknown data split. Using val data")
        return load_dataset('json', data_files={'val': [f'{DATA_LOCATION}val.json']})['val']


    def send_prompt(self, prompt: str, max_length=512) -> str:
        """Send a prompt to the model

        Args:
            prompt (str): Prompt to send to the model
            max_length (int, optional): Max generation length of the model. Defaults to 512.

        Returns:
            the reply of the model
        """    
        inputs = self._tokenizer.encode(prompt, return_tensors="pt").to(self._device)
        with torch.no_grad():
            outputs = self._model.generate(inputs, max_length=max_length)
        output_text = self._tokenizer.decode(outputs[0], skip_special_tokens=True)
        return output_text

    # Convert Label to ID
    def label_to_id(self,label: str):
        """Converts the output label of the model to ints, so we can analyze the performance.
        Entailment is converted to 0, contradiction is converted to 1

        Args:
            label (str): the label to be converted

        Returns:
            the int label
        """    
        pred = label.strip().lower()
        # sometimes the models reply yes/no -> translate this to entailment/contradiction
        if pred == "entailment" or pred == "yes":
            return 1
        if pred == "contradiction" or pred == "no":
            return 0
        return 0

# Main Function
def main():
    # Init
    print("Init model")
    t5_inference = T5Inference(MODEL_ID)

    # Load dataset
    dataset = t5_inference.load_and_split_dataset(DATA_SPLIT)

    # Generate predictions
    print("Generating predictions")
    predictions = []
    for datapoint in tqdm(dataset, desc="Processing"):
        predictions.append(t5_inference.label_to_id(t5_inference.send_prompt(datapoint["text"])))
    labels = [t5_inference.label_to_id(datapoint["label"]) for datapoint in dataset]

    # Compute evaluation metrics
    print(f"Precision: {precision_score(predictions, labels)}")
    print(f"Recall: {recall_score(predictions, labels)}")
    print(f"F1 Score: {f1_score(predictions, labels)}")

if __name__ == "__main__":
    main()
