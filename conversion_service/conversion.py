'''r=data
d=[]
d.append(r)
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
final_x = d
for sentence in final_x:
    sentence = sentence.lower()
    cleanr = re.compile('<.*?>')
    sentence = re.sub(cleanr,'',sentence)
    sentence = re.sub(r'[?|!|\'|*|#]' ,r'' ,sentence)
    sentence = re.sub(r'[.|,|)|(|\|/]' ,r' ',sentence)
    sentence = re.sub( r'[0-9]', '', sentence)
    words = [snow.stem(word) for word in sentence.split() if word not in stopwords]
    temp.append(' '.join(words))
r=temp
# average Word2Vec
# compute average word2vec for each review.
t_vectors = []; # the avg-w2v for each sentence/review is stored in this list
for sent in tqdm(temp): # for each review/sentence
    sent_vec = np.zeros(50) # as word vectors are of zero length 50, 
    #you might need to change this to 300 if you use google's w2v
    cnt_words =0; # num of words with a valid vector in the sentence/review
    for word in sent: # for each word in a review/sentence
        if word in w2v_words:
            vec = w2v_model.wv[word]
            sent_vec += vec
            cnt_words += 1
    if cnt_words != 0:
        sent_vec /= cnt_words
    t_vectors.append(sent_vec)
    
def validate_input(dict_request):
    def _validate_cols(col):
        schema = get_schema()
        actual_cols = schema.keys()
        if col not in actual_cols:
            raise NotInCols

    def _validate_values(col, val):
        schema = get_schema()

        if not (schema[col]["min"] <= float(dict_request[col]) <= schema[col]["max"]) :
            raise NotInRange

    for col, val in dict_request.items():
        _validate_cols(col)
        _validate_values(col, val)
    
    return True'''

def form_response(dict_request):
    '''if validate_input(dict_request):
        data = dict_request.values()
        data = [list(map(float, data))]
        response = predict(data)
        return response'''
        return clean(dict_request)

def api_response(dict_request):
    '''try:
        if validate_input(dict_request):
            data = np.array([list(dict_request.values())])
            response = predict(data)
            response = {"response": response}
            return response'''
            return clean(dict_request)