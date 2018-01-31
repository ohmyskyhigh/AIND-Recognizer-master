import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []
    # TODO implement the recognizer
    X_lengths = test_set.get_all_Xlengths()
    for X, lengths in X_lengths.values():
        likely = {}
        max_score = float('-inf') #get the maximum score in this block
        guess = None # get the best guess in this block
        for w, m in models.items():
            try:
                #score the data
                score = m.score(X, lengths)
                likely[w] = score

                if score > max_score:
                    max_score = score
                    guess = w
            except:
                # Unable to process word
                likely[w] = float("-inf")

        guesses.append(guess)
        probabilities.append(likely)

    return probabilities, guesses
