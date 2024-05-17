from spacy import load
import logging

# Load the French model
__SPACY_NLP = load('fr_core_news_sm')

def normalizeAccent(string):
    """
    @brief Normalizes the accents in a string.

    This function normalizes the accents in a string by replacing them with their non-accented counterparts.

    @param string The string to be normalized.
    @return The normalized string.
    """
    REPLACE_DICT = {
        'á': 'a',
        'â': 'a',
        'é': 'e',
        'è': 'e',
        'ê': 'e',
        'ë': 'e',
        'î': 'i',
        'ï': 'i',
        'ö': 'o',
        'ô': 'o',
        'ò': 'o',
        'ó': 'o',
        'ù': 'u',
        'û': 'u',
        'ü': 'u',
        'ç': 'c'
    }
    for key, value in REPLACE_DICT.items():
        string = string.replace(key, value)
    return string


def tokenize(string, is_normalized=False):
    """
    @brief Tokenizes a string.

    This function tokenizes a string by splitting it into words and normalizing the accents.

    @param string The string to be tokenized.
    @return A list of tokens.
    """
    if not is_normalized:
        string = normalizeAccent(string.lower())
    spacy_tokens = __SPACY_NLP(string)
    return ' '.join([token.orth_ for token in spacy_tokens if not token.is_punct if not token.is_stop])


def tokenizeDataset(dataset):
    """
    @brief Tokenizes a dataset.

    This function tokenizes the designation column of a pandas dataset by tokenizing each string in the dataset.
    Warning, changes the dataset in place, meaning that the original dataset is modified.

    @param dataset The dataset to be tokenized.
    @return None
    """
    logging.debug('Tokenizing dataset (may take a while depending on the dataset size)')
    dataset['designation'] = dataset['designation'].apply(tokenize)
