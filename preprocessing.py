import nltk
import string
import re
import io
import random


from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn import svm
from sklearn import linear_model
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import KFold
from sklearn import model_selection


#opening files.
f_neg = io.open('rt-polarity.neg', encoding = "ISO-8859-1", errors='ignore')
f_pos = io.open('rt-polarity.pos', encoding = "ISO-8859-1", errors='ignore')

#load nltk stop words (english)
stopwords = set(stopwords.words('english'))

corpus_neg = []			#negative corpus with stopwords
corpus_pos = []			#positive corpus with stopwords

corpus_neg_nostop = []	#this will contain the negative corpus with no stopwords
corpus_pos_nostop = []	#this will contain the positive corpus with no stopwords

#iterate through lines of file
for line in f_neg:
		#removing punctuation
	line_no_punctuation = re.sub(r'['+string.punctuation+']+', ' ', line)
		#tokenize the sentences
	tokens = word_tokenize(line_no_punctuation)	#tokenize each line	

	joined = " ".join(tokens)

		#removing stop words
	result = []
	for word in tokens:
		if word not in stopwords:
			result.append(word);
	joined_nostop = " ".join(result)
	
	corpus_neg.append(joined);
	corpus_neg_nostop.append(joined_nostop);

#print(corpus_neg[:5])
#print(corpus_neg_nostop[:5])

for line in f_pos:
		#removing punctuation
	line_no_punctuation = re.sub(r'['+string.punctuation+']+', ' ', line)
		#tokenize the sentences
	tokens = word_tokenize(line_no_punctuation)	#tokenize each line
		#removing stop words
	
	joined = " ".join(tokens);

	result = []
	for word in tokens:
		if word not in stopwords:
			result.append(word);
	joined_nostop = " ".join(result)
	
	corpus_pos.append(joined);
	corpus_pos_nostop.append(joined_nostop);
#print(corpus_pos[:5])


#At this point, we have corpus_pos and corpus_neg.
X_pos = corpus_pos
y_pos = [1] * len(corpus_pos)
X_neg = corpus_neg
y_neg = [0] * len(corpus_neg)

X_pos_nostop = corpus_pos_nostop
y_pos_nostop = [1] * len(corpus_pos_nostop)
X_neg_nostop = corpus_neg_nostop
y_neg_nostop = [0] * len(corpus_neg_nostop)

#combining the whole corpus
X = X_pos + X_neg
y = y_pos + y_neg

X_nostop = X_pos_nostop + X_neg_nostop;
y_nostop = y_pos_nostop + y_neg_nostop;

#using count vectorizer to extract features
cv = CountVectorizer();
bigram_cv = CountVectorizer(ngram_range=(2, 2));
uni_bi_cv = CountVectorizer(ngram_range=(1, 2));

X_cv = cv.fit_transform(X);
X_bigram_cv = bigram_cv.fit_transform(X);
X_uni_bi_cv = uni_bi_cv.fit_transform(X);

X_nostop_cv = cv.fit_transform(X_nostop);
X_nostop_bigram_cv = bigram_cv.fit_transform(X_nostop);
X_nostop_uni_bi_cv = uni_bi_cv.fit_transform(X_nostop);



#split into training and testing sets
rand_state = 180 #this is used as seed for random benchmark too


kf = KFold(n_splits=5, shuffle = True, random_state = rand_state);

model1 = svm.SVC(kernel='linear', C = 1.0)
model2 = linear_model.LogisticRegression()
model3 = MultinomialNB()

for  m in [model2, model3]:
	results = model_selection.cross_val_score(m, X_cv, y, cv=kf)
	print("Accuracy: %.3f%% (%.3f%%)") % (results.mean()*100.0, results.std()*100.0)


'''
X_train, X_test, y_train, y_test = train_test_split( X_cv, y, test_size = 0.2, random_state = rand_state)
X_train2, X_test2, y_train2, y_test2 = train_test_split( X_bigram_cv, y, test_size = 0.2, random_state = rand_state)
X_train3, X_test3, y_train3, y_test3 = train_test_split( X_uni_bi_cv, y, test_size = 0.2, random_state = rand_state)

X_train4, X_test4, y_train4, y_test4 = train_test_split( X_nostop_cv, y, test_size = 0.2, random_state = rand_state)
X_train5, X_test5, y_train5, y_test5 = train_test_split( X_nostop_bigram_cv, y, test_size = 0.2, random_state = rand_state)
X_train6, X_test6, y_train6, y_test6 = train_test_split( X_nostop_uni_bi_cv, y, test_size = 0.2, random_state = rand_state)

'''

#print(y_train)
print("Train set size: {}".format(len(y_train)))
print("Test set size: {}".format(len(y_test)))

print(y_train.count(0) , y_train.count(1))
print(y_test.count(0) , y_test.count(1))


#print(X_traincv.toarray());

#print(cv.get_feature_names())




#Classification

	#Benchmark random guessing
random.seed(rand_state)
good_count = 0
for res in y_test:
	random_predict = random.randint(0,1)
	if(random_predict == res):
		good_count = good_count + 1;


	#Declaring a new Linear Kernel SVM
clf_SVC_lin = svm.SVC(kernel='linear', C = 1.0)
	#Training the model
clf_SVC_lin.fit(X_train, y_train);
print("Model Trained!")


	#declaring a Naive Bayes
clf_NB_Multinomial = MultinomialNB();
	#Training the model
clf_NB_Multinomial.fit(X_train, y_train);
print("Model Trained!")

	#declaring a Logistic Regression
clf_Logistic = linear_model.LogisticRegression();
clf_Logistic.fit(X_train, y_train);
print("Model Trained!")

	#the random guessing, we take correct / total (the *1.0 is to force float div)
print("Random Guessing: {}".format((good_count * 1.0) / len(y_test)))
print("SVM (Linear): {}".format(clf_SVC_lin.score(X_test, y_test)))
print("Multinomial Naive Bayes: {}".format(clf_NB_Multinomial.score(X_test, y_test)))
print("Logistic Regression: {}".format(clf_Logistic.score(X_test, y_test)))





