import pandas as pd
import numpy as np
import time
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from transformers import BertTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset


import logging
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.FileHandler("sentiment_analysis_training_log.log"),
                              logging.StreamHandler()])


random_seed = 42
np.random.seed(random_seed)
torch.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True


logging.info("Loading and preprocessing dataset..")
df = pd.read_csv('Tweets.csv')
df = df[['text', 'airline_sentiment']]
df = df.drop_duplicates()
df = df.drop(df[df.airline_sentiment == 'negative'].iloc[:6000].index)
logging.info(f"Dataset Shape: {df.shape}")


label_counts = df['airline_sentiment'].value_counts()
logging.info("Label distribution:\n", label_counts)


def clean_text(text):
    """
    Clean and preprocess the given text by applying a series of transformations.

    Args:
        text (str): The input text to be cleaned.

    Returns:
        str: The cleaned text with various transformations applied.
    """
    
    tweet = text.lower()                                                # Convert text to lowercase
    tweet = re.sub(r'((www\.[^\s]+)|(https?://[^\s]+))', '', tweet)     # Remove links (URLs)
    tweet = re.sub(r'@[^\s]+', '', tweet)                               # Remove usernames (@mentions)
    tweet = re.sub(r'\s+', ' ', tweet)                                  # Remove additional whitespaces
    tweet = re.sub(r'#([^\s]+)', r'\1', tweet)                          # Replace hashtags with words
    tweet = tweet.strip('\'"')                                          # Trim surrounding single and double quotes
    tweet = re.sub(r'[^a-zA-Z\s]', '', tweet)                           # Remove non-alphabetic characters
    return " ".join(tweet.split())


def deEmojify(text):
    """
    Remove emojis from the given text.

    Args:
        text (str): The input text containing emojis.

    Returns:
        str: The input text with emojis removed.
    """
    return text.encode('ascii', 'ignore').decode('ascii')


logging.info("Performing text cleaning..")
df['cleaned_text'] = df['text'].apply(clean_text)
df['cleaned_text'] = df['cleaned_text'].apply(deEmojify)


label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['airline_sentiment'])


logging.info("Splitting into train and test dataset..")
train_data, test_data = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])


logging.info("Finetuning BERT Model for Sentiment Analysis..")
model_name = 'bert-base-uncased'
max_length = 128
batch_size = 32
num_epochs = 10
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=len(label_encoder.classes_))


class CustomDataset(Dataset):
    def __init__(self, data, tokenizer, max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        text = self.data.iloc[index]['text']
        label = self.data.iloc[index]['label']
        
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }
    

train_dataset = CustomDataset(train_data, tokenizer, max_length)
test_dataset = CustomDataset(test_data, tokenizer, max_length)


train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

model.to(device)

optimizer = AdamW(model.parameters(), lr=2e-5)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(train_dataloader) * num_epochs)
criterion = nn.CrossEntropyLoss()


logging.info("Started Training..")
for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    
    start_time = time.time()  
    
    for batch_num, batch in enumerate(train_dataloader, 1):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask)[0]
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        
        _, predicted_labels = torch.max(outputs, dim=1)
        correct_predictions += (predicted_labels == labels).sum().item()
        total_samples += labels.size(0)
    
    average_loss = total_loss / len(train_dataloader)
    accuracy = correct_predictions / total_samples
    
    end_time = time.time()  
    epoch_time = end_time - start_time  
    
    logging.info(f"Epoch [{epoch + 1}/{num_epochs}] - Training Loss: {average_loss:.4f} - Training Accuracy: {accuracy:.4f} - Time: {epoch_time:.2f} seconds")

logging.info("Training complete!!")


logging.info("Evaluating the model on Test Data..")
model.eval()
total_correct = 0
total_samples = 0

predicted_all = []
labels_all = []

with torch.no_grad():
    for batch in test_dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        outputs = model(input_ids, attention_mask=attention_mask)[0]
        _, predicted_labels = torch.max(outputs, dim=1)
        
        total_correct += (predicted_labels == labels).sum().item()
        total_samples += labels.size(0)
        
        predicted_all.extend(predicted_labels.cpu().tolist())
        labels_all.extend(labels.cpu().tolist())

accuracy = total_correct / total_samples
logging.info(f"Validation Accuracy: {accuracy:.2f}")


cm_plot_labels = label_encoder.classes_
cm = confusion_matrix(labels_all, predicted_all)
cm = pd.DataFrame(cm, columns = cm_plot_labels, index = cm_plot_labels)
plt.figure(figsize=(12, 6))
ax = sns.heatmap(cm, annot = True, fmt = 'g' ,vmin = 0, vmax = 250,cmap = 'Blues')
ax.set_xlabel('Predicted',fontsize = 14,weight = 'bold')
ax.set_xticklabels(ax.get_xticklabels(),rotation =0)
ax.set_ylabel('Actual',fontsize = 14,weight = 'bold') 
ax.set_yticklabels(ax.get_yticklabels(),rotation =0)
ax.set_title('Confusion Matrix - Test Set',fontsize = 16,weight = 'bold',pad=20)

confusion_plot_path = 'confusion_matrix.png'
plt.savefig(confusion_plot_path, bbox_inches='tight', pad_inches=0.1)
plt.close()
logging.info(f"Confusion Matrix plot saved as {confusion_plot_path}")


logging.info("Saving the model endpoint..")
model_checkpoint = {
    'model_state_dict': model.state_dict(),
    'tokenizer': tokenizer,
    'label_encoder': label_encoder,
    'max_length': max_length,
    'model_name': model_name
}

checkpoint_path = 'model_checkpoint.pth'
torch.save(model_checkpoint, checkpoint_path)
logging.info(f"Model checkpoint saved as {checkpoint_path}")

