import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer
import pickle
import re
import math
import urllib.parse
import pandas as pd
ps = PorterStemmer()


def transform_text_email(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

def transform_text_url(url):
    # Parse the URL
    parsed_url = urllib.parse.urlparse(url)
    
    features = {}
    
    # 1. url-length: Number of characters in the URL
    features['url_length'] = len(url)
    
    # 2. has_subscribe: Whether the URL contains the word 'subscribe'
    features['has_subscribe'] = int('subscribe' in url.lower())
    
    # 3. contains_hash: Whether the URL contains the hash '#' symbol
    features['contains_hash'] = int('#' in url)
    
    # 4. num_digits: The number of digits in the URL
    features['num_digits'] = len(re.findall(r'\d', url))
    
    # 5. non_https: Whether the URL uses a non-HTTPS connection
    features['non_https'] = int(parsed_url.scheme != 'https')
    
    # 6. num_words: The number of words in the URL (split by '/' and '-')
    features['num_words'] = len(re.findall(r'[\w]+', parsed_url.path))
    
    # 7. entropy: Measure of entropy (disorder/uncertainty) in the URL
    def calculate_entropy(url):
        # Frequency of each character
        prob = [float(url.count(c)) / len(url) for c in set(url)]
        # Shannon entropy formula
        entropy = -sum([p * math.log2(p) for p in prob])
        return entropy
    
    features['entropy'] = calculate_entropy(url)
    
    # 8. num_params: Number of query parameters in the URL
    query_params = urllib.parse.parse_qs(parsed_url.query)
    features['num_params'] = len(query_params)
    
    # 9. num_fragments: Number of fragments in the URL (after '#')
    features['num_fragments'] = len(parsed_url.fragment.split('&')) if parsed_url.fragment else 0
    
    # 10. num_subdomains: Number of subdomains (split by '.')
    features['num_subdomains'] = len(parsed_url.netloc.split('.')) - 2  # -2 for 'domain' and 'tld'
    
    # 11. num_%20: Number of encoded white spaces ('%20') in the URL
    features['num_%20'] = url.count('%20')
    
    # 12. num_@: Number of '@' symbols in the URL
    features['num_@'] = url.count('@')
    
    # 13. has_ip: Check if the URL has an IP address instead of a domain name
    ip_pattern = re.compile(r'(\d{1,3}\.){3}\d{1,3}')  # Pattern to match IP addresses
    features['has_ip'] = int(bool(ip_pattern.search(parsed_url.netloc)))
    
    # Convert the features dictionary into a Pandas DataFrame for model prediction
    feature_df = pd.DataFrame([features])
    
    return feature_df

def is_valid_url(url):
    # Regular expression pattern for matching URLs
    url_pattern = re.compile(
        r'^(https?|ftp):\/\/'               # Protocol (http, https, ftp)
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+'  # Domain name
        r'[A-Z]{2,6}\.?|localhost|'         # Domain extensions (like .com, .net) or localhost
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}|'  # OR IPv4 address
        r'\[?[A-F0-9]*:[A-F0-9:]+\]?)'      # OR IPv6 address
        r'(?::\d+)?'                        # Optional port number
        r'(?:\/?|[\/?]\S+)$', re.IGNORECASE # Rest of the URL with optional query parameters
    )
    
    # Check if the URL matches the pattern
    if re.match(url_pattern, url):
        return True
    else:
        return False
    
tfidf = pickle.load(open('vectorizer.pkl','rb'))
model_sms = pickle.load(open('model.pkl','rb'))
model_url = pickle.load(open('voting_classifier.pkl', 'rb'))

st.title("url/SMS Spam Classifier")

input = st.text_area("Enter the message or url")

if st.button('Predict'):
    if is_valid_url(input):
            # 1. Preprocess the URL (extract features)
        transformed_url = transform_text_url(input)

        # 2. Make the prediction using the pre-trained model
        result = model_url.predict(transformed_url)[0]

        # 3. Display the result
        if result == 1:
            st.header("Spam URL")
        else:
            st.header("Legitimate URL")

    else:    
        # 1. preprocess
        transformed_sms = transform_text_email(input)
        # 2. vectorize
        vector_input = tfidf.transform([transformed_sms])
        # 3. predict
        result = model_sms.predict(vector_input)[0]
        # 4. Display
        if result == 1:
            st.header("Spam")
        else:
            st.header("Not Spam")