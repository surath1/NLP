###  NLP (Natural language processing)

- Natural language processing (NLP) is a machine learning technology that gives computers the ability to interpret, manipulate, and comprehend human language.

- !pip install nltk [( https://tedboy.github.io/nlps/index.html )]
- !pip install svgling

## Tokenization

## Stemming 

## Lemmatizer

## Stopword

## Parts of speach tagging

## Named Entity Recognization 

### Text to vector
- One HOT Encoding
- Bag of words (BOW)
- TF-IDF
- Word2Vect
- AvgWord2Vect

## One HOT Encoding
# Advantages 
- Easy to implement ( from sklearn.preprocessing import OneHotEncoder  & pd.get_dummies() )

# Dis-advantage
- Sparse Matrix (huge no of 1 & 0), Word to vector -> each word convert to one and zero for a unique word sentence.
- Overfitting ( very good at train data not for new data )
- Fixed text size ( sentences might contains multiple word )
- No semantic meaning is getting capured ( How each word is related or similar )
- New data set might get different set of data which is not present in unique word sentences.
- Example 50 K unique word it leads to problem represent 1 & 0

## BOW

## TF-IDF

## Word2Vect

## AvgWord2Vect



  
