###  NLP (Natural language processing)

- Natural language processing (NLP) is a machine learning technology that gives computers the ability to interpret, manipulate, and comprehend human language.

- !pip install nltk [( https://tedboy.github.io/nlps/index.html )]
- !pip install svgling

#### Tokenization
```bash
!pip install nltk

from nltk.tokenize import sent_tokenize, word_tokenize

curpus = """The elephant is the world's largest living creature."""
document = sent_tokenize(curpus)
type(document)
for doc in document:
    print(doc)

```

#### Stemming 
```bash
from nltk.stem import PorterStemmer
stemmer = PorterStemmer()

words = [ "eat","eats","eaten","eattte","eater","run","running","ran","play","playing", "history","hist"]

for word in words:
    print(word + " -- "+ porterStemmer.stem(word))

```

#### Lemmatizer
```bash
from nltk.stem import WordNetLemmatizer
wnl = WordNetLemmatizer()
print(wnl.lemmatize('dogs'))

words = [ "eat","eats","eaten","eattte","eater","run","running","ran","play","playing", "history","hist"]
for word in words:
    print(word + " -- "+ wnl.lemmatize(word))

```
#### Stopword
```bash
from nltk.corpus import stopwords
stopwords.words()
stopwords.words('english')

```
#### Parts of speach tagging
```bash
import nltk
nltk.pos_tag(['India', 'is', 'a', 'beautiful', 'country'])

```
#### Named Entity Recognization 
```bash
import nltk

sentences = """ The Jagannath Temple is an important Hindu temple dedicated to Jagannath, a form of Vishnu – one of the trinity of supreme divinity in Hinduism. Puri is in the state of Odisha, on the eastern coast of India.
King Indradyumna of Avanti has built the main temple of Jagannath at Puri."

words = nltk.word_tokenize(sentences)
elements = nltk.pos_tag(words)

nltk.ne_chunk(elements)

nltk.ne_chunk(elements).draw()

```
#### NLP Text to vector technique
- One HOT Encoding
- Bag of words (BOW)
- TF-IDF
- Word2Vect
- AvgWord2Vect

### One HOT Encoding model
#### Advantages 
- Easy to implement ( from sklearn.preprocessing import OneHotEncoder  & pd.get_dummies() )

#### Dis-advantage
- Sparse Matrix (huge no of 1 & 0), Word to vector -> each word convert to one and zero for a unique word sentence.
- Overfitting ( very good at train data not for new data )
- Fixed text size ( sentences might contains multiple word )
- No semantic meaning is getting capured ( How each word is related or similar )
- New data set might get different set of data which is not present in unique word sentences.
- Example 50 K unique word it leads to problem represent 1 & 0

### Bag-of-words model
- Since bag-of-words approach works based on the frequency count of the words.
- All unique words from the corpus
- Step 1: Lowercase the input data, Tokenize the data, remove stop words and perform stemming or lemmatization.
- Step 2: List all unique words
- Step 3: Sorted words on frequency of occurrence in descending order. (may ignore single occurance)
- 
#### Advantages 
- Easy to implement
- Fixed size input help ML algorithm

#### Dis-advantage
- Sparce Matrix (Overfitting)
- Semantic meaning is not captured 
- New test data(if additional word)is going to ignore.
- Ordering of word changed 

- Bag of Words: Converting words to numbers with no semantic information. 
- TF-IDF: It is also converting the words to numbers or vectors with some weighted information.

### TF-IDF
- Term Frequency-Inverse Document Frequency (TF-IDF)
- Term Frequency (TF) - (No. of repeated words in sentence) / (No. of words in sentence)
- Inverse Document Frequency (IDF) - log[ (No. of sentences) / (No. of sentences containing word)]
#### Advantages 
- Input is fixed in size
- Word importance is captured 

#### Dis-advantage
- Sparce Matrix (Overfitting)
- Out of vocabulary , slow for large documents.

### Word2Vect
- Feature representation 
- Words that occur in similar contexts and have similar meanings. 
- Two main architectures: Continuous Bag of Words (CBOW) and Skip-gram.
- CBOW - the algorithm predicts the target word based on its surrounding context words.( Small dataset )
- Skip-gram - the algorithm predicts the context words given the target word. ( Huge dataset )
- More training data will resume better accuracy 

- !pip install gensim

- word2vec-google-news-300 { Pre-trained vectors trained on a part of the Google News dataset (about 100 billion words). The model contains 300-dimensional vectors for 3 million words and phrase }

#### Advantages 
- Improved semantic capture leading to hopefully better modeling results.
- 300 vector dimensiom/ Window size by google.

#### Dis-advantage
- representation for out-of-vocabulary words.

### AvgWord2Vect
- Avg Word2Vec is an extension of the Word2Vec model that generates vector representations for sentences or documents instead of individual words. It works by taking the average of the vector representations of all the words in a sentence or document to generate a single vector representation for the entire text.


### Use Cases ##############

- TFIDF, Word2Vec, AvgWord2Vec used in various NLP applications.

- Sentiment Analysis- These models can be used to classify the sentiment of a piece of text as positive, negative, or neutral.
- Document Similarity- These models can be used to compare the similarity between two documents or to cluster similar documents together.
- Chatbots- These models can be used to build chatbots that can understand natural language input from users and provide appropriate responses.



  
