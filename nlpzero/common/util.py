import numpy as np


def preprocess(text):
    """
    >>> text = "You say goodbye and I say hello."
    >>> corpus, word_to_id, id_to_word = preprocess(text)
    >>> word_to_id
    {'you': 0, 'say': 1, 'goodbye': 2, 'and': 3, 'i': 4, 'hello': 5, '.': 6}
    >>> id_to_word
    {0: 'you', 1: 'say', 2: 'goodbye', 3: 'and', 4: 'i', 5: 'hello', 6: '.'}
    """
    text = text.lower()
    text = text.replace(".", " .")  # ピリオドを単語から切り離す
    words = text.split(" ")

    word_to_id = {}
    id_to_word = {}
    for word in words:
        # 内包表記だと重複する単語を除きにくいのでこの書き方になると思われる
        if word not in word_to_id:
            new_id = len(word_to_id)
            word_to_id[word] = new_id
            id_to_word[new_id] = word

    corpus = np.array([word_to_id[w] for w in words])

    return corpus, word_to_id, id_to_word
