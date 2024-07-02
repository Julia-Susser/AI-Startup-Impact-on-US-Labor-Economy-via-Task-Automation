import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet, stopwords
from nltk.stem import WordNetLemmatizer
import string

def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)

def lemmatize(sentence):
    # Initialize lemmatizer
    lemmatizer = WordNetLemmatizer()
    
    # Get the list of stopwords
    stop_words = set(stopwords.words('english'))

    # Tokenize the sentence
    tokens = word_tokenize(sentence)

    # Remove punctuation, stop words, and lemmatize each word
    lemmatized_tokens = [
        lemmatizer.lemmatize(token, get_wordnet_pos(token)) 
        for token in tokens if token not in string.punctuation and token.lower() not in stop_words
    ]

    # Join the tokens back into a sentence
    lemmatized_sentence = ' '.join(lemmatized_tokens)

    return lemmatized_sentence


