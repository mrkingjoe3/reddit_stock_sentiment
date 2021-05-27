import praw
import pandas as pd
import re
import nltk
import spacy
from nltk.stem.wordnet import WordNetLemmatizer

#nltk.download('wordnet')
nlp = spacy.load('en_core_web_sm')

emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags 
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
#problem with spacing
bad_word = re.compile('bearish |bears |bear |red |puts |put |shorted |shorting ')
good_word = re.compile('bullish |bulls |bull |green |calls |call ')
remove_special_char = re.compile(r"[,'()\]$%]")
user_names = re.compile(r'(@.*?\s)|(@.*?$)')
remove_not_contractions = re.compile(r"n't")
remove_iam = re.compile(r"i\'m")
remove_link = re.compile(r'http[\S]*')
lemmatizer = WordNetLemmatizer()


BLACKLIST = ['bleeding', 'itm', 'congrats', 'fda', 'cnbc', 'cathie', 'fidelity', 'ev', 'wsb', 
             'sec', 'rh', 'etf', ' ', 'irs', 'r/', 'u/', 'iphone', 'app', 'fed', 'op',
             'vanguard', 'covid', 'robinhood', 'usd', 'yahoo', 'berkshire', 'meme', 'yolo', 'ha', 'time', 
             'otm', 'mf', 'medicare', 'wtf', 'ath', 'congress', 'elon', 'tech', 'webull', 'af', 'house',
             'eps', 'apes', 'ira', 'fair', 'ice', 'net', 'cramer', 'etrade', 'kinda', 'musk', 'eu',
             'fyi', 'aws', 'td', 'gotcha', 'guh', 'nah', 'way', 'moon', 'u', 'gl', 'faamg', 'faang', 'karma']


# This function scans specific subreddits for a specific number of posts and the comments of the post
# We will only get post/comments containing a word in query
def get_reddit_data(my_username, my_password, num_submissions = 500, get_comments = True, query = [], save_to_csv = True, 
                    subreddit_list = ['Wallstreetbets', 'Stocks', 'Investing', 'StockMarket'],
                    submissions_path = 'data/submissions.csv', comments_path = 'data/comments.csv'):
      
    reddit = praw.Reddit(client_id='WX8tsO3brrseeA', 
                       client_secret='qp3mJC41c4z4IG2UeJ_wZ2J2RdLlaQ', 
                       username=my_username,
                       password=my_password,
                       user_agent='Scanner', 
                       check_for_async=False)
    
    
    submissions = []
    comments = []
    query = [s.lower() for s in query] 
    subreddits = [reddit.subreddit(subreddit) for subreddit in subreddit_list]
    
    #Loop through all the subreddits
    for subreddit in subreddits:
    
        #Loop through the hot submissions of the selected subreddit
        for submission in subreddit.hot(limit=num_submissions):
            submissions.append([submission.title, submission.subreddit])
      
            if get_comments: 
                #Get rid of non-comments
                submission.comments.replace_more(limit=0)
                for comment in submission.comments.list():
                    if not query:
                        comments.append([comment.body, submission.subreddit])
                    else:
                        comment_words = [word.lower() for word in comment.body.split()]
                        if any(word in query for word in comment_words):
                            comments.append([comment.body, submission.subreddit])
                
    submissions = pd.DataFrame(submissions, columns=['text', 'subreddit'])
    comments = pd.DataFrame(comments, columns=['text', 'subreddit'])
    
    if save_to_csv:
        submissions.to_csv(submissions_path, index=False)
        comments.to_csv(comments_path, index=False)
                          
    return submissions, comments
      
    
def process_and_remove_text(s, stock_data = True):

  # Convert to lower case
  s = s.lower()

  # Remove any entries that are too long, have emojis, or are equal to [removed]
  if len(s) > 100 or\
     len(s) < 5 or\
     s == '[removed]' or\
     emoji_pattern.search(s) != None:
    good_string = False
  else:
    good_string = True

  s = remove_special_char.sub(r'', s)
  if stock_data:
    s = re.sub('$', ' ', s)
    # Change common stock sentiment words to positive/ negative
    s = bad_word.sub(r'bad ', s)
    s = good_word.sub(r'good ', s)
  # Change 't to 'not'
  s = remove_not_contractions.sub(" not", s)
  # Remove any @user names
  s = user_names.sub('', s)
  # Remove any links
  s = remove_link.sub('', s)
  # Remove useless words
  #s = " ".join([x for x in s.split() if x not in stopwords])
  # Remove trailing whitespace
  #s = re.sub(r'\s+', ' ', s).strip()
  s = [lemmatizer.lemmatize(w) for w in s.split()]
  s = ' '.join(s)
    
  return s, good_string


def get_tickers(text):
  list = []
  ents = nlp(text).ents

  for ent in ents:
    if ent.label_ == "ORG" and not any(substring in ent.text.lower() for substring in BLACKLIST):
      list.append(ent.text.lower())
  #list = list(set(list))
  return list

def get_processed_data(my_username, my_password, save_data=True, load_data=True, data_path='data/data.csv'):
    
    if load_data:
        try:
            data = pd.read_csv(data_path)
        except:
            comments, submissions = get_reddit_data(my_username, my_password)
            data = [comments, submissions]
            data = pd.concat(data)
            data = data.reset_index(drop=True)
    
    else:
        comments, submissions = get_reddit_data(my_username, my_password)
        data = [comments, submissions]
        data = pd.concat(data)
        data = data.reset_index(drop=True)
        
    data[['text modified', 'keep']] = data['text'].apply(lambda x: pd.Series(process_and_remove_text(x), 
                                                                         index=['text modified', 'keep']))
    data = data[data['keep'] == True]
    data = data.reset_index(drop=True)
    data = data.drop(labels=['keep'], axis=1)
    print('Done with processing data. Now to get those tickers!')

    data['tickers'] = data['text modified'].apply(get_tickers)
    #data = data[data['tickers'].str.len() == 1]
    print('Done with the tickers now!')
    
    if save_data:
        data.to_csv(data_path, index=False)
        
    return data