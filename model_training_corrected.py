import warnings

warnings.filterwarnings("ignore", category=UserWarning, message=r".*contains `beta`.*")
warnings.filterwarnings("ignore", category=UserWarning, message=r".*contains `gamma`.*")



from tqdm import tqdm
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import AdamW, SGD
from torch.utils.data import Dataset, DataLoader, Subset
from transformers import get_scheduler, AutoModel, AutoTokenizer, AutoModelForSequenceClassification, BertTokenizerFast, BertForSequenceClassification
from torch.profiler import profile, record_function, ProfilerActivity
from sklearn.model_selection import train_test_split

class CustomTextDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        text = item['Ticket Description']
        label = item['Ticket Priority']

        encoding = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        encoding = {key: val.squeeze(0) for key, val in encoding.items()}

        return {
            **encoding,
            'labels': label
        }

class BERTWithClassifierHead(nn.Module):
    def __init__(self, num_classes):
        super(BERTWithClassifierHead, self).__init__()
        self.bert = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_classes)
    
    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        return self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, labels=labels)

priority_mapping = {
    "Low": 0,
    "Medium": 1,
    "High": 2,
    "Critical": 3
}
try:
    file_path = 'customer_support_tickets.csv'
    data = pd.read_csv(file_path)
    data = data[['Ticket Description', 'Ticket Priority']]
    data['Ticket Priority'] = data['Ticket Priority'].map(priority_mapping)
    data['Ticket Priority'] = data['Ticket Priority'].apply(lambda x: torch.tensor(x))
except FileNotFoundError as e:
    print(f"Error: {e}")
except KeyError as e:
    print(f"Error: {e}")


tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')


try:
    ds = CustomTextDataset(data, tokenizer)
except NameError as e:
    print(f"Error: {e}")

BATCH_SIZE = 20


try:
    train_indices, test_indices = train_test_split(range(len(ds)), test_size=0.1)
    train_split = Subset(ds, train_indices)
    test_split = Subset(ds, test_indices)

    train_batches = DataLoader(train_split, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
    test_batches = DataLoader(test_split, batch_size=BATCH_SIZE)
except NameError as e:
    print(f"Error: {e}")


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


try:
    model = BERTWithClassifierHead(num_classes=4)
    model.to(device)
except NameError as e:
    print(f"Error: {e}")
epochs = 3
lr = 5e-5
try:
    optimizer = AdamW(model.parameters(), lr=lr)
    num_training_steps = epochs * len(train_batches)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps
    )

    model.train()
    for epoch in range(epochs):
        for batch in tqdm(train_batches):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()


    res = tokenizer(text="Test Text. Hi, Hello!", padding='max_length', max_length=500, truncation=True, return_tensors='pt')
    res = {k: v.to(device) for k, v in res.items()}
    output = model(input_ids=res['input_ids'], attention_mask=res['attention_mask'])
    print(output)

    m = nn.Softmax(dim=1)
    scaled = m(output.logits)
    print(scaled)
    prediction = torch.argmax(scaled, dim=1).item()
    print(prediction)


    labels = ["Low", "Medium", "High", "Critical"] 
    print(labels[prediction])


    correct = 0
    total = 0

    with torch.no_grad():
        for batch in test_batches:
            X = {k: batch[k] for k in batch.keys() if k != "labels"}
            y = batch["labels"]

            X = {k: v.to(device) for k, v in X.items()}
            y = y.to(device)

            outputs = model(**X).logits
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == y).sum().item()
            total += y.size(0)

    print(f'Model Accuracy: {100 * correct // total} %')


    import evaluate

    metric = evaluate.load("accuracy")
    model.eval()
    for batch in test_batches:
        X = {k: batch[k] for k in batch.keys() if k != 'labels'}
        labels = batch['labels'].to(device)
        
        with torch.no_grad():
            outputs = model(**X)

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        metric.add_batch(predictions=predictions, references=labels)

    print(metric.compute())

except NameError as e:
    print(f"Error: {e}")
