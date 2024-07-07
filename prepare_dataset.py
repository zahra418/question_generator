# prepare_dataset.py
from pathlib import Path
import transformers
from datasets import load_dataset
from transformers import AutoTokenizer
import nltk
import yaml

nltk.download('punkt') # download punkt tokenizer

# Load configuration
current_path = Path(__file__).parent 
config_file_path = current_path / "config" / "config.yaml"

with open(config_file_path, 'r') as f:
    config = yaml.safe_load(f)

model_checkpoint = config['model']['checkpoint'] # the model
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint) # the tokenizer

# the dataset
train_dataset = load_dataset("squad", split=config['data']['train_split'])
valid_dataset = load_dataset("squad", split=config['data']['validation_split'])

prefix = "generate question: " if model_checkpoint in ["t5-small", "t5-base", "t5-large", "t5-3b", "t5-11b"] else ""

max_input_length = config['preprocessing']['max_input_length']
max_target_length = config['preprocessing']['max_target_length']

def preprocess_function(examples):
    inputs = [prefix + context for context in examples["context"]] # add the `prefix` to each context in the input `examples`
   
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True) # set the max\min size to the input # we tokenize the input
    
    # Setup the tokenizer for targets # we tokenize the answers to prepare targets for the model
    labels = tokenizer(text_target=[answer['text'][0] for answer in examples["answers"]], max_length=max_target_length, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Tokenize the datasets
tokenized_train_dataset = train_dataset.map(preprocess_function, batched=True)
tokenized_valid_dataset = valid_dataset.map(preprocess_function, batched=True)

# Save tokenized datasets
tokenized_train_dataset.save_to_disk(config['output']['tokenized_train_dataset'])
tokenized_valid_dataset.save_to_disk(config['output']['tokenized_valid_dataset'])
