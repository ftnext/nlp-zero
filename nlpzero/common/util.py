import numpy as np


def preprocess(text):
    """Convert text to a list of word ids; the list is called corpus.

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


def create_co_matrix(corpus, vocab_size, window_size=1):
    """Create co-occurrence matrix from corpus."""
    corpus_size = len(corpus)
    co_matrix = np.zeros((vocab_size, vocab_size), dtype=np.int32)

    for idx, word_id in enumerate(corpus):
        for i in range(1, window_size + 1):
            left_idx = idx - i
            right_idx = idx + i

            # はみ出していない添字の語のカウントを1増やす
            if left_idx >= 0:
                left_word_id = corpus[left_idx]
                co_matrix[word_id, left_word_id] += 1

            if right_idx < corpus_size:
                right_word_idx = corpus[right_idx]
                co_matrix[word_id, right_word_idx] += 1

    return co_matrix
