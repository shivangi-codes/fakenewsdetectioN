import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 
from tqdm import tqdm 
import re 
import nltk 
nltk.download(’punkt’) 
nltk.download(’stopwords’) 
from nltk.corpus import stopwords 
from nltk. tokenize import wordtokenize 26
from nltk.stem.porter import PorterStemmer 
from wordcloud import WordCloud 
data = pd.read_csv(’News.csv’,index-col=0,nrows=1000) 
data.head() 
print(data) 
data = data.drop(["title", "subject","date"], axis = 1) 
data.isnull().sum() 
data = data.sample(frac=1) 
data.resetindex(inplace=True) 
data.drop(["index"], axis=1, inplace=True) 
sns.countplot(data=data, x=’class’, order=data[’class’].valuecounts().index) 
def preprocess_text(text-data): 
preprocessed_text = () 
for sentence in tqdm(text-data): 
sentence = re.sub(r’(w s)’, ”, sentence) 
preprocessedtext.append(’ ’.join(token.lower()))
for token in str(sentence).split() 
if token not in
stopwords.words(’english’))) 
return preprocessed_text 
preprocessed_review = preprocess_text(data[’text’].values) 
data[’text’] = preprocessed_review 
Real consolidated = ’ ’.join(word for word in data[’text’][data[’class’] == 1].astype(str)) 
wordCloud = WordCloud(width=1600, height=800, random-state=21, max-font-size=110, collocations=False) 
plt.figure(figsize=(15, 10)) 
plt.imshow(wordCloud.generate(consolidated), interpolation=’bilinear’) plt.axis(’off’) 
plt.show() 
consolidated = ’ ’.join( 
word for word in data[’text’][data[’class’] == 0].astype(str)) 
wordCloud = WordCloud(width=1600, 
height=800,
random-state=21, 
max-font-size=110, 
collocations=False) 
plt.figure(figsize=(15, 10)) 
plt.imshow(wordCloud.generate(consolidated), interpolation=’bilinear’) plt.axis(’off’) 
plt.show() 
from sklearn.feature-extraction.text import CountVectorizer 
def get-top-n-words(corpus, n=None): 
vec = CountVectorizer().fit(corpus) 
bag_of_words = vec.transform(corpus) 
sum_words = bag_of_words.sum(axis=0) 
words_freq = [(word, sum_words[0, idx]) 
for word, idx in vec.vocabulary.items()] 
words-freq = sorted(words-freq, key=lambda, x: x[1], 
reverse=True) 
return words-freq[:n] 
common_words = get_top_n_words(data[’text’], 20) 
df1 = pd.DataFrame(common-words, columns=[’Review’, ’count’])
df1.groupby(’Review’).sum()[’count’].sort_values(ascending=False).plot( kind=’bar’, 
figsize=(10, 6), 
xlabel="Top Words", 
ylabel="Count", 
title="Bar Chart of Top Words Frequency" ) 
from sklearn.model_selection import train-test-split 
from sklearn.metrics import accuracy-score 
from sklearn.linear_model import LogisticRegression 
x_train, x_test, y_train, y_test = train_test_split(data[’text’], 
data[’class’], test_size=0.25) 
from sklearn.feature_extraction.text import TfidfVectorizer 
vectorization = TfidfVectorizer() 
x_train = vectorization.fit-transform(x_train) 
xtest = vectorization.transform(x_test) 
from sklearn.linear_model import LogisticRegression 
model = LogisticRegression() 
model.fit(x_train, y_train) 
#Testing the model
print(accuracyscore(y-_rain, model.predict(x,train))) 
print(accuracyscore(y_test, model.predict(x,test))) 
from sklearn.tree import DecisionTreeClassifier 
model = DecisionTreeClassifier() 
model.fit(x_train, y_train) 
#Testing the model 
print(accuracy_score(y_train, model.predict(x_train))) 
print(accuracy_score(y_test, model.predict(x_test))) 
Confusion matrix of Results from Decision Tree classification 
from sklearn import metrics 
cm=metrics.confusionmatrix(y_test, model. predict(x,test)) 
cmdisplay=metrics.ConfusionMatrixDisplay(confusionmatrix=cm, displaylabels=[False, True]) 
cmdisplay.plot() 
plt.show() 
