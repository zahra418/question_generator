from pathlib import Path
import torch
import transformers
from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
from datasets import load_from_disk
from evaluate import load
import numpy as np
import nltk
import yaml

# Load configuration from config.yaml
current_path = Path(__file__).parent 
config_file_path = current_path / "config" / "config.yaml"

with open(config_file_path, 'r') as f:
    config = yaml.safe_load(f)

# Download the punkt tokenizer
nltk.download('punkt')

# Extract configurations
model_checkpoint = config['training']['model_checkpoint']
batch_size = config['training']['batch_size']
learning_rate = config['training']['learning_rate']
weight_decay = config['training']['weight_decay']
num_train_epochs = config['training']['num_train_epochs']
evaluation_strategy = config['training']['evaluation_strategy']
save_total_limit = config['training']['save_total_limit']
output_dir = config['training']['output_dir']
predict_with_generate = config['training']['predict_with_generate']
fp16 = config['training']['fp16']
push_to_hub = config['training']['push_to_hub']
tokenized_train_dataset_path = config['training']['tokenized_train_dataset']
tokenized_valid_dataset_path = config['training']['tokenized_valid_dataset']

# Load the model and tokenizer
model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
tokenizer = transformers.AutoTokenizer.from_pretrained(model_checkpoint)

# Check if CUDA is available and set device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Load tokenized datasets
tokenized_train_dataset = load_from_disk(tokenized_train_dataset_path)
tokenized_valid_dataset = load_from_disk(tokenized_valid_dataset_path)

# Load the rouge metric for evaluation
metric = load("rouge")

# Define training arguments
args = Seq2SeqTrainingArguments(
    output_dir=output_dir,
    evaluation_strategy=evaluation_strategy,
    learning_rate=learning_rate,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=weight_decay,
    save_total_limit=save_total_limit,
    num_train_epochs=num_train_epochs,
    predict_with_generate=predict_with_generate,
    fp16=fp16,
    push_to_hub=push_to_hub,
)

# Define data collator
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# Compute metrics function
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # Split the decoded predictions and labels into sentences for better evaluation
    decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]
    
    # Compute rouge scores
    result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True, use_aggregator=True)
    result = {key: value * 100 for key, value in result.items()}  # Convert scores to percentages
    
    # Avg length of generated predictions
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)
    
    return {k: round(v, 4) for k, v in result.items()}

# Initialize the trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_valid_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# Start training
trainer.train()

# Save the fine-tuned model & tokenizer
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
