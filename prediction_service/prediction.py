import yaml
import os
import json
import joblib
import numpy as np
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from tqdm import tqdm
import nltk
import re
import numpy as np

params_path = "params.yaml"

def read_params(config_path=params_path):
    with open(config_path) as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config

def predict(data):
    config = read_params(params_path)
    model_dir_path = config["webapp_model_dir"]
    model = joblib.load(model_dir_path)
    prediction = model.predict(np.array(data))
    return prediction

def clean_input(dict_request):
  config = read_params(params_path)
  w2v_path = config["w2v_path"]
  w2v_model = Word2Vec.load(w2v_path)
  w2v_words = list(w2v_model.wv.vocab)
  snow = nltk.stem.SnowballStemmer('english')
  stopwords= set(['br', 'the', 'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've",\
              "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', \
              'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their',\
              'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', \
              'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', \
              'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', \
              'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after',\
              'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further',\
              'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',\
              'most', 'other', 'some', 'such', 'only', 'own', 'same', 'so', 'than', 'too', 'very', \
              's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', \
              've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn',\
              "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn',\
              "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", \
              'won', "won't", 'wouldn', "wouldn't"])

  temp=[]
  temp.append(dict_request['review'])
  final_x = temp
  clean_data=[]
  for sentence in final_x:
      sentence = sentence.lower()
      cleanr = re.compile('<.*?>')
      sentence = re.sub(cleanr,'',sentence)
      sentence = re.sub(r'[?|!|\'|*|#]' ,r'' ,sentence)
      sentence = re.sub(r'[.|,|)|(|\|/]' ,r' ',sentence)
      sentence = re.sub( r'[0-9]', '', sentence)
      words = [snow.stem(word) for word in sentence.split() if word not in stopwords]
      clean_data.append(' '.join(words))

  t_vectors = []
  for sent in tqdm(clean_data): # for each review/sentence
      sent_vec = np.zeros(50) # as word vectors are of zero length 50, 
      cnt_words =0
      for word in sent:
          if word in w2v_words:
              vec = w2v_model.wv[word]
              sent_vec += vec
              cnt_words += 1
      if cnt_words != 0:
          sent_vec /= cnt_words
      t_vectors.append(sent_vec)
      return t_vectors



def form_response(dict_request):
  data=clean_input(dict_request)
  response = predict(data)
  return response

'''def api_response(dict_request):
    if validate_input(dict_request):
        data = np.array([list(dict_request.values())])
        response = predict(data)
        response = {"response": response}
        return response'''