from flask import Flask, jsonify, request
import json
import logging
import nltk.corpus
import nltk.stem.snowball
import nltk.tokenize
from nltk.corpus import wordnet
import string

# Get default English stopwords and extend with punctuation
stopwords = nltk.corpus.stopwords.words('english')
stopwords.extend(string.punctuation)
stopwords.append('')

# Create tokenizer and stemmer
lemmatizer = nltk.stem.wordnet.WordNetLemmatizer()


def tokenize(text):
    tokens = nltk.tokenize.word_tokenize(text)
    return tokens


def get_wordnet_pos(pos_tag):
    if pos_tag[1].startswith('J'):
        return (pos_tag[0], wordnet.ADJ)
    elif pos_tag[1].startswith('V'):
        return (pos_tag[0], wordnet.VERB)
    elif pos_tag[1].startswith('N'):
        return (pos_tag[0], wordnet.NOUN)
    elif pos_tag[1].startswith('R'):
        return (pos_tag[0], wordnet.ADV)
    else:
        return (pos_tag[0], wordnet.NOUN)


def is_match(a, b):
    """Check if a and b are matches."""
    pos_a = map(get_wordnet_pos, nltk.pos_tag(tokenize(a)))
    pos_b = map(get_wordnet_pos, nltk.pos_tag(tokenize(b)))
    lemmae_a = [
        lemmatizer.lemmatize(token.lower().strip(string.punctuation), pos)
        for token, pos in pos_a
        if pos == wordnet.NOUN and token.lower().strip(string.punctuation)
        not in stopwords
    ]
    lemmae_b = [
        lemmatizer.lemmatize(token.lower().strip(string.punctuation), pos)
        for token, pos in pos_b
        if pos == wordnet.NOUN and token.lower().strip(string.punctuation)
        not in stopwords
    ]

    # Calculate Jaccard similarity
    intersect = set(lemmae_a).intersection(lemmae_b)
    union = set(lemmae_a).union(lemmae_b)
    return len(intersect) / float(len(union))


app = Flask(__name__)
app.config.update(DEBUG=True)


@app.route('/favicon.ico')
def favicon():
    return send_from_directory(
        os.path.join(app.root_path, 'static'),
        'favicon.ico',
        mimetype='image/vnd.microsoft.icon')


@app.route('/')
def index():
    data = {'name': 'Marking API', 'version': 0.1}
    return jsonify(data)


@app.route('/mark', methods=['POST'])
def mark():
    try:
        data = request.get_json(force=True)
        percent = is_match(data.get('scheme', ''), data.get('answer', ''))
        logging.info(data)
        logging.info(percent)
        return jsonify({'percent':percent * 100})
    except Exception as e:
        logging.error(e)
        return jsonify({'error':'An error occured.'})


if __name__ == '__main__':
    app.run(host='0.0.0.0')
