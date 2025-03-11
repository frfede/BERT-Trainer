## Training script with K-Fold cross-validation for Sequence Classification tasks

# Imports
from datasets import load_dataset

# Tokenizer and model loader change based on the model
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, EarlyStoppingCallback

from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold
import numpy as np
import torch
import json
from time import time
import argparse
import os

def create_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Directory {directory} created")
    else:
        print(f"Directory {directory} already exists")

# Argument definitions
parser = argparse.ArgumentParser(description="Fine-tuning a BERT model with Transformers")
parser.add_argument("--model_name", type=str, required=True, help="Name of the pre-trained model (e.g., bert-base-uncased)")
parser.add_argument("--dataset_path", type=str, required=True, help="Path to the dataset (can be the name of a Huggingface dataset or a file path)")
parser.add_argument("--epochs", type=int, default=5, help="Number of epochs to train for (e.g., 3)")
parser.add_argument("--train_batch_size", type=int, default=16, help="Training batch size")
parser.add_argument("--eval_batch_size", type=int, default=32, help="Evaluation batch size")
parser.add_argument("--eval_percentage", type=int, default=1, help="Indicates how often to perform validation/save model steps as a percentage of completion")
parser.add_argument("--early_stopping_patience", type=int, default=6, help="Number of steps after which, if model performance continues to deteriorate, training is stopped")

# Argument parsing
args = parser.parse_args()

# Access arguments
model_name = args.model_name
dataset_path = args.dataset_path

# Verify dataset path
try:
    print(f"Loading dataset from: {dataset_path}")
    # Load dataset (either from Huggingface or local file)
    dataset = load_dataset(dataset_path, split='train')
except FileNotFoundError:
    print(f"Error: File or directory {dataset_path} not found.")
    exit(1)
except ValueError:
    print(f"Error: Dataset {dataset_path} is invalid. Verify the name is correct or check file format.")
    exit(1)
except Exception as e:
    print(f"Unknown error while loading dataset: {e}")
    exit(1)

# Loading tokenizer
try:
    print(f"Loading tokenizer for model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
except Exception as e:
    print(f"Error loading tokenizer for model {model_name}: {e}")
    exit(1)

max_length = tokenizer.model_max_length if (tokenizer.model_max_length < 2048) else 512 # fix for debertav3 model_max_length bug

# Dataset tokenization
def tokenize_function(batch):
    return tokenizer(batch["text"], padding='max_length', truncation=True, max_length=max_length)

tokenized_dataset = dataset.map(tokenize_function, batched=True)
tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

# K-Fold
k = 5
kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

dt_labels = tokenized_dataset['label']

# Metrics
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    return {"accuracy" : acc, "f1": f1, "precision" : prec, "recall" : rec}

# Initialize fold results lists
fold_accuracies = []
fold_f1_scores = []
fold_precisions = []
fold_recalls = []
fold_losses = []
fold_runtimes = []
fold_train_times = []

def unique_labels(labels):
    output = set()
    for x in labels:
        output.add(x)
    return len(output)    

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

# Training loop for each fold
for fold, (train_index, val_index) in enumerate(kf.split(np.zeros(len(dt_labels)), dt_labels)):

    print(f'Fold {fold + 1}/{k}')

    train_dataset = tokenized_dataset.select(train_index)
    val_dataset = tokenized_dataset.select(val_index)

    # Calculate eval_steps and save_steps values
    train_dataset_size = len(train_dataset)
    steps_per_epoch = train_dataset_size // args.train_batch_size
    total_steps = steps_per_epoch * args.epochs
    num_steps = (total_steps * args.eval_percentage) // 100

    # Load model
    try:
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=unique_labels(dataset['label'])).to(device)
    except Exception as e:
        print(f"Error loading model {model_name}: {e}")
        exit(1)

    # Define training hyperparameters
    training_args = TrainingArguments(
            output_dir=f'./results/{model_name}/{dataset_path}/fold_{fold + 1}',
            num_train_epochs=args.epochs,
            learning_rate=2e-5,
            weight_decay=0.01,
            per_device_train_batch_size=args.train_batch_size,
            per_device_eval_batch_size=args.eval_batch_size,
            eval_strategy='steps',
            eval_steps=num_steps,
            save_strategy='steps',
            save_steps=num_steps,
            save_total_limit=args.early_stopping_patience+1,
	        warmup_steps=total_steps*10//100,
            load_best_model_at_end=True,
            metric_for_best_model='loss',
            push_to_hub=False,
            report_to=None
        )

    # Trainer instantiation
    trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)]
        )
    
    # Start training
    start_time = time()
    trainer.train()

    metrics = trainer.evaluate()
    train_time = time() - start_time

    # Print and save results
    print(f"Fold {fold + 1} - Accuracy: {metrics['eval_accuracy']:.4f}, F1: {metrics['eval_f1']:.4f}, Precision: {metrics['eval_precision']:.4f}, Recall: {metrics['eval_recall']:.4f}, Loss : {metrics['eval_loss']:.4f}, Runtime : {metrics['eval_runtime']:.4f}, Train time : {train_time:.4f} secs")
    
    fold_accuracies.append(metrics['eval_accuracy'])
    fold_f1_scores.append(metrics['eval_f1'])
    fold_precisions.append(metrics['eval_precision'])
    fold_recalls.append(metrics['eval_recall'])
    fold_losses.append(metrics['eval_loss'])
    fold_runtimes.append(metrics['eval_runtime'])
    fold_train_times.append(train_time)

    part_res = {"model" : model_name, 
           "results" : {
               "accuracy" : f"{metrics['eval_accuracy']:.4f}",
               "f1" : f"{metrics['eval_f1']:.4f}",
               "precision" : f"{metrics['eval_precision']:.4f}",
               "recall" : f"{metrics['eval_recall']:.4f}",
               "loss" : f"{metrics['eval_loss']:.4f}",
                "runtime" : f"{metrics['eval_runtime']:.4f}",
                "train time" : f"{train_time:.4f}"
           }}
    
    save_path = f'/home/gsafractal/Desktop/Tesi-Francesco-Federico/Modelli addestrati/{model_name}/{dataset_path}'
    create_dir(save_path)

    with open(f'{save_path}/risultati-fold-{fold + 1}.json', 'w') as outfile:
        json.dump(part_res, outfile)
    
    # Save model and tokenizer
    trainer.save_model(f'{save_path}/fold-{fold+1}')
    tokenizer.save_pretrained(f'{save_path}/fold-{fold+1}')
    print(f"Model for fold {fold + 1} saved in {save_path}/fold-{fold+1}")
    
    del model
    torch.cuda.empty_cache()

print(f'Average Accuracy: {np.mean(fold_accuracies):.4f} ± {np.std(fold_accuracies):.4f}')
print(f'Average F1: {np.mean(fold_f1_scores):.4f} ± {np.std(fold_f1_scores):.4f}')
print(f'Average Precision: {np.mean(fold_precisions):.4f} ± {np.std(fold_f1_scores):.4f}')
print(f'Average Recall: {np.mean(fold_recalls):.4f} ± {np.std(fold_recalls)}')
print(f'Average Loss: {np.mean(fold_losses):.4f} ± {np.std(fold_losses):.4f}')
print(f'Average Runtime: {np.mean(fold_runtimes):.4f} ± {np.std(fold_runtimes):.4f}')
print(f'Average Train time: {np.mean(fold_train_times):.4f} ± {np.std(fold_train_times):.4f}')
print(f'Total train time: {np.sum(fold_train_times):.4f}')

# Save final overall results
results = {"model" : model_name, 
           "results" : {
               "accuracy" : f"{np.mean(fold_accuracies):.4f} ± {np.std(fold_accuracies):.4f}",
               "f1" : f"{np.mean(fold_f1_scores)::.4f} ± {np.std(fold_f1_scores):.4f}",
               "precision" : f"{np.mean(fold_precisions):.4f} ± {np.std(fold_precisions):.4f}",
               "recall" : f"{np.mean(fold_recalls):.4f} ± {np.std(fold_recalls):.4f}",
               "loss" : f"{np.mean(fold_losses):.4f} ± {np.std(fold_losses)::.4f}",
               "runtime" : f"{np.mean(fold_runtimes):.4f} ± {np.std(fold_runtimes):.4f}",
               "Train time media" : f"{np.mean(fold_train_times):.4f} ± {np.std(fold_train_times):.4f}",
               "Total train time" : f"{np.sum(fold_train_times):.4f}"
           }}

with open(f'{save_path}/risultati.json', 'w') as outfile:
    json.dump(results, outfile)
