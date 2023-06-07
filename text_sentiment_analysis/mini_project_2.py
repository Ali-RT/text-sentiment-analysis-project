from collections import Counter

import numpy as np
from mini_project_1 import (count_words, pretty_print_review_and_label,
                            read_file)


def update_input_layer(review, layer_0, word2index):
    """Modify the layer_0 to represent the vector form of review.
    The element at a given index of layer_0 should represent
    how many times the given word occurs in the review.
    Args:
        review(string) - the string of the review
        layer_0(numpy array) - the initial layer of zeroes
        word2index(dict) - dictionary mapping words to indices
    Returns:
        layer_0(numpy array) - the updated input layer
    """
    # clear out previous state by resetting the layer to be all 0s
    layer_0.fill(0)

    # Count how many times each word is used in the given review and store the results in layer_0
    review_counts = Counter(review.split(" "))
    for word, count in review_counts.items():
        if word in word2index:
            layer_0[0, word2index[word]] = count
    return layer_0


def get_target_for_label(label):
    """Convert a label to `0` or `1`.
    Args:
        label(string) - Either "POSITIVE" or "NEGATIVE".
    Returns:
        `0` or `1`.
    Raises:
        ValueError: If label is not 'POSITIVE' or 'NEGATIVE'
    """
    if label not in ["POSITIVE", "NEGATIVE"]:
        raise ValueError("Label must be either 'POSITIVE' or 'NEGATIVE'")

    return 1 if label == "POSITIVE" else 0


def main():
    reviews = read_file("reviews.txt")
    labels = [label.upper() for label in read_file("labels.txt")]

    for i in range(len(reviews[:5])):
        pretty_print_review_and_label(i, reviews, labels)

    total_counts, label_counts = count_words(reviews, labels, ["POSITIVE", "NEGATIVE"])

    vocab = set(total_counts.keys())
    vocab_size = len(vocab)
    layer_0 = np.zeros((1, vocab_size))

    assert layer_0.shape == (1, 74074)

    word2index = {}
    for i, word in enumerate(vocab):
        word2index[word] = i

    layer = update_input_layer(reviews[0], layer_0, word2index)
    print(layer)


if __name__ == "__main__":
    main()
