import nltk 
import pickle 
from nltk.corpus import stopwords
import re 
from nltk.stem.porter import PorterStemmer 
app = Flask(name) 
ps = PorterStemmer() 
#Load model and vectorizer 
with open(r"model.pkl", ’rb’) as f: 
model = pickle.load(f) 
with open(tfidfvect2.pkl’, ’rb’) as f: 
tfidfvect = pickle.load(f) 
#Build functionalities 
@app.route(’/’, methods=[’GET’]) 
def home(): 
return rendertemplate(′index.html′) 
def predict(text): 
review = re.sub(’[a-zA-Z]’, ’ ’, text) 
review = review.lower() 
review = review.split() 
review = [ps.stem(word) for word in review if not word in 
stopwords.words(’english’)]
review = ’ ’.join(review) 
reviewvect = tfidfvect.transform([review]).toarray() 
prediction = ’FAKE’ if model.predict(reviewvect) == 0else′REAL′ 
print(model.predict(reviewvect)) 
return prediction 
@app.route(’/’, methods=[’POST’]) 
def webapp(): 
text = request.form[’text’] 
prediction = predict(text) 
return render-template(’index.html’, text=text, result=prediction) 
@app.route(’/predict/’, methods=[’GET’,’POST’]) 
def api(): 
text = request.args.get("text") 
prediction = predict(text) 
print(prediction) 
return jsonify(prediction=prediction) 
if name == "main": 
app.debug = True 
app.run()
