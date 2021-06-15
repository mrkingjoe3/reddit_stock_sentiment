# reddit_stock_sentiment
This project scans reddit and reports the most commonly mentioned stock tickers in comments and posts. Then sentiment analysis is performed, and the final results is a list of the top tickers, the amount of times they appear, and their average sentiment. To view the results, go to src/data/most_common_tickers.

For the sentiment model, I used a BERT model with 6 LSTM layers and two linears layers on top of it. The model was trained using the data here: https://www.kaggle.com/kazanova/sentiment140. This dataset contains tweets with a positive or negative sentiment. Using this dataset, the model has a testing accuracy of 86% from training for 6 epochs. This is a huge model, and it was hard to optmize it, as I do not have a great laptop. I might not be using the best model architecture, but this is the best time vs perforamce archtieture I found.

Their are two traditional python files: one for text collection/processing and another for the sentiment model. The main program is a IPython notebook that can easily be modified for different purposes, such as loading a pretrained model or using data that is already collected. 

The only problem with this project is the time it takes to collect data, process it, and then analyze it. The size of the model, dataset, training data etc. make this project somewhat inefficient.
