from torchinfo import summary
from transformers import AutoModel, BertModel, AutoTokenizer, AdamW, get_linear_schedule_with_warmup, BertForSequenceClassification
from torch import nn, optim
import torch as T
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
#from transformers.models.auto.tokenization_auto import AutoTokenizer

BATCH_SIZE = 64
TEST_SPLIT = .1
EPOCHS = 5
layers = 6
GRU_OUTPUT_SIZE = 512
DENSE1_OUTPUT_SIZE = 16
OUTPUT_SIZE = 2
MODEL_NAME = 'bert-base-uncased'
DEVICE = T.device("cuda:0" if T.cuda.is_available() else "cpu")
MAX_LEN = 100
TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME)


# A dataset class for formatting training and testing data to go into a pytorch model
class CreateDataset(Dataset):
  def __init__(self, data, tokenizer=TOKENIZER,
               max_len=MAX_LEN):
    self.text = data['text'].to_numpy()
    self.labels = data.labels.apply(lambda x: 1 if x==4 else 0).to_numpy()
    self.tokenizer = TOKENIZER
    self.max_len = max_len
  def __len__(self):
    return len(self.text)
  def __getitem__(self, item):
    text = str(self.text[item])
    labels = self.labels[item]
    encoding = TOKENIZER.encode_plus(
      text,
      add_special_tokens=True,
      padding='max_length',
      max_length=self.max_len,
      return_token_type_ids=False,
      return_attention_mask=True,
      truncation=True,
      return_tensors='pt',
    )
    return {
      'text': text,
      'input_ids': encoding['input_ids'].flatten(),
      'attention_mask': encoding['attention_mask'].flatten(),
      'labels': T.tensor(labels, dtype=T.long)
    }
    
# A data loader to interface to our dataset class
def LoadData(data):
  ds = CreateDataset(data=data)
  return DataLoader(
    ds,
    batch_size=BATCH_SIZE,
    num_workers=4
  )
  
  
# To get the training and testing data broken down into data loaders
def get_data(PATH = 'training.1600000.processed.noemoticon.csv'):
    
  data = pd.read_csv(PATH, encoding ='latin', names = ['labels','id','date','query','user','text'])
  train, test = train_test_split(data, test_size=.1)
  data_loader_train = LoadData(train)
  data_loader_test = LoadData(test)
  
  return len(train), len(test), data_loader_train, data_loader_test
  

# This is a custom sentiment analysis model built from a base of a BERT transformer model
class SentimentModel(nn.Module):
  
  def __init__(self, n_classes, len_data):
    super().__init__()
    self.bert = BertModel.from_pretrained(MODEL_NAME)
    for param in self.bert.parameters():
        param.requires_grad=False
    embedding_dim = self.bert.config.to_dict()['hidden_size']
    self.rnn = nn.GRU(embedding_dim,
                      GRU_OUTPUT_SIZE,
                      num_layers = layers,
                      batch_first = True)
    self.dense1 = nn.Linear(GRU_OUTPUT_SIZE, DENSE1_OUTPUT_SIZE)
    self.output = nn.Linear(DENSE1_OUTPUT_SIZE, n_classes)
    self.dropout = nn.Dropout(.1)
    self.loss_function = nn.CrossEntropyLoss()
    self.optimizer = AdamW(self.parameters(), lr = .0001, correct_bias=False)
    self.scheduler =  get_linear_schedule_with_warmup(self.optimizer,
                                                      num_warmup_steps=10,
                                                      num_training_steps=len_data*EPOCHS)
  
  # Run a forward propagation step 
  def forward(self, input_ids, attention_mask):
    with T.no_grad():
      embedding, pooled = self.bert(
          input_ids,
          attention_mask,
          return_dict=False)
    X = self.rnn(embedding)[1]  
    X = self.dropout(X[-1,:,:])
    X = F.relu(self.dense1(X))
    X = self.output(X)
    #Do we want this activation relu here?
    return X 

  # Run one step with training data
  def train_step(self, d):
    input_ids = d['input_ids'].to(DEVICE)
    attention_mask = d['attention_mask'].to(DEVICE)
    labels = d['labels'].to(DEVICE)

    model = self.train()
    # The raw output of shape [Batch_size, 2]
    raw_output = model(input_ids, attention_mask)
    # Apply a softmax to the raw output
    refined_output = F.softmax(raw_output, dim=1)
    # Get the class of the correct prediciton by finding the max value of each output
    _, predictions = T.max(refined_output, axis=1)
    # Add to the correct predictions
    correct_predictions = T.sum(predictions == labels).cpu().numpy()
    #batch_accuracy_arr.append(correct_predictions_batch.double()/BATCH_SIZE)
    #correct_predictions +=correct_predictions_batch

    # Compute the loss
    loss = self.loss_function(raw_output, labels)
    loss.backward()
    # Clip the gradients if they exceed a threshold
    nn.utils.clip_grad_norm_(self.parameters(), max_norm = 1.0)
    # Update stuff
    self.optimizer.step()
    self.scheduler.step()
    #Reset the grads to zero because otherwise pytorch sums the grads over each training step
    self.optimizer.zero_grad()

    return correct_predictions

  # Run one step with testing data
  def test_step(self, data):
    input_ids = data['input_ids'].to(DEVICE)
    attention_mask = data['attention_mask'].to(DEVICE)
    labels = data['labels'].to(DEVICE)

    model = self.eval()
    output = F.softmax(model(input_ids, attention_mask), dim=1)
    _, predictions = T.max(output, axis=1)
    correct_predictions = T.sum(predictions == labels).cpu().numpy()
    return correct_predictions

  # Takes an input string of text and returns the sentiment of it
  def get_sentiment(self, text):
  
    encoding = TOKENIZER.encode_plus(
      text,
      add_special_tokens=True,
      padding='max_length',
      max_length=MAX_LEN,
      return_token_type_ids=False,
      return_attention_mask=True,
      truncation=True,
      return_tensors='pt',
    )
  
    input_ids = encoding['input_ids']
    attention_mask = encoding['attention_mask']
    model = self.eval()
    # Get the raw output
    with T.no_grad():
      outputs = model(
          input_ids,
          attention_mask
      )
    # Find which of the two outputs is greater and make that the predicted class
    _, predictions = T.max(outputs, dim=1)

    return predictions.item()
    
  
# Get training and test data and then train the model for a given amount of epochs
def train_model():
    len_train, len_test, data_loader_train, data_loader_test = get_data()
    model = SentimentModel(2, len_train).to(DEVICE)
    
    for i in range(EPOCHS):
        
        correct_train = 0
        correct_test = 0
        
        for train, test in zip(data_loader_train,
                           data_loader_test):
            correct_train += model.train_step(train)
            correct_test += model.test_step(test)
            
        print(f'Train {i}: {correct_train/len_train}')
        print(f'Test {i}: {correct_test/len_test}')
        
    return model

# Load a pretrained model
def load_trained_model():
    model = SentimentModel(2, 0)
    model.load_state_dict(T.load('twitter_sentiment_model', map_location='cpu'), strict=False)
    return model

# Save a model
def save_model(model, name):
    T.save(model.state_dict(),name)
    
# A function to get the sentiment and count of any given ticker in a list of data
# We will look in data for sentiment and count
def get_ticker_sent(data, ticker):
    # Create a dataframe with all the occurences of a ticker in data
    data_ticker = data.loc[data['tickers'] == ticker]
    count = len(data_ticker)
    # Find the average sentiment of this ticker
    average_sentiment = np.sum(data_ticker['sentiment'].to_numpy())/count
    return pd.Series([average_sentiment, count])

# Get the sentiment for each entry of data. Then use get_tickers_sentiment to find the most popular
# tickers
def get_most_popular_tickers(model, data):
      
    data = data.loc[data['text modified'] != 'none']
    data = data.loc[data['text modified'].str.len() > 1]
    data['sentiment'] = data['text modified'].apply(model.get_sentiment)
    print('Done with sentiment dectection!')
    
    # Extract the tickers from data, making sure there are no duplicates, and create a new dataframe to hold them
    tickers_sentiment = pd.DataFrame(columns=['ticker', 'sentiment', 'count'])
    tickers_list = []
    [tickers_list.append(ticker) for ticker in data['tickers'].to_list() if ticker not in tickers_list]
    tickers_sentiment['ticker'] = tickers_list

    tickers_sentiment[['sentiment', 'count']] = tickers_sentiment['ticker'].apply(lambda x: get_ticker_sent(data,
                                                                                                            x))
    tickers_sentiment = tickers_sentiment.sort_values(by='count', ascending=False)
    tickers_sentiment = tickers_sentiment.reset_index(drop=True)
    tickers_sentiment.to_csv('data/most_popular_tickers.csv', index = False)
    
    return tickers_sentiment