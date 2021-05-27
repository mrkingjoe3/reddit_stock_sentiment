import data_collection_processing as data_stuff
import sentiment_model as SentimentModel
import pandas as pd

if __name__ == '__main__':


    model = SentimentModel.load_trained_model()
    print('start')
    #my_username = input("Enter reddit username: ")
    #my_password = input("enter reddit password: ")
    my_username = 'thebigjay5'
    my_password = 'EvK82rP$'
    
    #data = data_stuff.get_processed_data(my_username, my_password, True, False)
    data = pd.read_csv('data/data.csv')
    data = model.get_most_popular_tickers(data)
    data.to_csv('data/most_popular_tickers.csv')
    
    print(data)
