import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import classification_report
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.optim import AdamW
from sklearn.utils.class_weight import compute_class_weight
import pickle

# Training configuration
class_number = 2
base_model = 'climatebert/distilroberta-base-climate-f' #model is automatically downloaded at runtime if it is not already stored locally
n_words = 250
batch_size = 64
learning_rate = 1e-6
epochs = 35
weight_decay = 0.01

training_args = {
    "learning_rate": learning_rate,
    "per_device_train_batch_size": batch_size,
    "per_device_eval_batch_size": batch_size,
    "num_train_epochs": epochs,
    "weight_decay": weight_decay,
    "n_words": n_words,
    "class_number": class_number
}

# Automatically use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load datasets
train_df = pd.read_csv('Classifiers/Data/relevance_classifier_climatetech_train_set.csv')
validation_df = pd.read_csv('Classifiers/Data/relevance_classifier_climatetech_validation_set.csv')
test_df = pd.read_csv('Classifiers/Data/relevance_classifier_climatetech_test_set.csv')

# Extract inputs and labels
train_text = train_df['text']
train_labels = train_df['label']
val_text = validation_df['text']
val_labels = validation_df['label']
test_text = test_df['text']
test_labels = test_df['label']

# Load model and tokenizer
auto_model = AutoModelForSequenceClassification.from_pretrained(base_model, num_labels=class_number) #model is automatically downloaded at runtime if it is not already stored locally
tokenizer = AutoTokenizer.from_pretrained(base_model)

# Tokenization helper
def tokenize(texts):
    return tokenizer.batch_encode_plus(
        texts,
        max_length=n_words,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )

# Tokenize all sets
tokens_train = tokenize(train_text.tolist())
tokens_val = tokenize(val_text.tolist())
tokens_test = tokenize(test_text.tolist())

# Convert to tensors
train_seq, train_mask = tokens_train['input_ids'], tokens_train['attention_mask']
val_seq, val_mask = tokens_val['input_ids'], tokens_val['attention_mask']
test_seq, test_mask = tokens_test['input_ids'], tokens_test['attention_mask']

train_y = torch.tensor(train_labels.values)
val_y = torch.tensor(val_labels.values)
test_y = torch.tensor(test_labels.values)

# DataLoaders
train_data = TensorDataset(train_seq, train_mask, train_y)
train_dataloader = DataLoader(train_data, sampler=RandomSampler(train_data), batch_size=batch_size)

val_data = TensorDataset(val_seq, val_mask, val_y)
val_dataloader = DataLoader(val_data, sampler=SequentialSampler(val_data), batch_size=batch_size)

# Custom model wrapper
class BERT_Arch(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.bert = model
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, sent_id, mask):
        output = self.bert(sent_id, attention_mask=mask, return_dict=False)
        return self.softmax(output[0])

# Build and push model to device
model = BERT_Arch(auto_model).to(device)

# Optimizer
optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

# Compute class weights
class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(train_labels), y=train_labels)
weights = torch.tensor(class_weights, dtype=torch.float).to(device)
cross_entropy = torch.nn.CrossEntropyLoss(weight=weights)

# Training loop
def train():
    model.train()
    total_loss = 0
    total_preds = []

    for step, batch in enumerate(train_dataloader):
        if step % 50 == 0 and step != 0:
            print(f'  Batch {step:>5}  of  {len(train_dataloader):>5}.')

        batch = [r.to(device) for r in batch]
        sent_id, mask, labels = batch

        model.zero_grad()
        preds = model(sent_id, mask)
        loss = cross_entropy(preds, labels)

        total_loss += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        preds = preds.detach().cpu().numpy()
        total_preds.append(preds)

    avg_loss = total_loss / len(train_dataloader)
    total_preds = np.concatenate(total_preds, axis=0)
    return avg_loss, total_preds

# Evaluation loop
def evaluate():
    print("\nEvaluating...")
    model.eval()
    total_loss = 0
    total_preds = []

    for step, batch in enumerate(val_dataloader):
        if step % 50 == 0 and step != 0:
            print(f'  Batch {step:>5}  of  {len(val_dataloader):>5}.')

        batch = [t.to(device) for t in batch]
        sent_id, mask, labels = batch

        with torch.no_grad():
            preds = model(sent_id, mask)
            loss = cross_entropy(preds, labels)
            total_loss += loss.item()
            preds = preds.detach().cpu().numpy()
            total_preds.append(preds)

    avg_loss = total_loss / len(val_dataloader)
    total_preds = np.concatenate(total_preds, axis=0)
    return avg_loss, total_preds

# Training process
best_valid_loss = float('inf')
train_losses, valid_losses = [], []

for epoch in range(epochs):
    print(f'\n Epoch {epoch + 1} / {epochs}')

    train_loss, _ = train()
    valid_loss, _ = evaluate()

    if valid_loss < best_valid_loss:
        print("Saving best model...")
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'relevance_classifier_climatetech_saved_weights.pt')

    train_losses.append(train_loss)
    valid_losses.append(valid_loss)

    print(f'Training Loss: {train_loss:.3f}')
    print(f'Validation Loss: {valid_loss:.3f}')

# Load best model and evaluate on test data
model.load_state_dict(torch.load('relevance_classifier_climatetech_saved_weights.pt'))

with torch.no_grad():
    preds = model(test_seq.to(device), test_mask.to(device))
    preds = preds.detach().cpu().numpy()

preds = np.argmax(preds, axis=1)

clf_report = classification_report(test_y, preds)
pickle.dump(clf_report, open('relevance_classifier_climatetech_classification_report.txt', 'wb'))
print(clf_report)

