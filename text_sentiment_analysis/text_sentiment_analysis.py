import os
import sys
import time
from collections import Counter
from enum import Enum
from typing import List

import numpy as np


class Label(Enum):
    POSITIVE = 1
    NEGATIVE = 0


def read_file(file_path):
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"No such file or directory: '{file_path}'")

    with open(file_path, "r") as file:
        return file.read().split("\n")


# Encapsulate our neural network in a class
class SentimentNetwork:
    def __init__(
        self,
        reviews,
        labels,
        min_count=10,
        polarity_cutoff=0.1,
        hidden_nodes=10,
        learning_rate=0.1,
    ):
        """Create a SentimenNetwork with the given settings
        Args:
            reviews(list) - List of reviews used for training
            labels(list) - List of POSITIVE/NEGATIVE labels associated with the given reviews
            min_count(int) - Words should only be added to the vocabulary
                             if they occur more than this many times
            polarity_cutoff(float) - The absolute value of a word's positive-to-negative
                                     ratio must be at least this big to be considered.
            hidden_nodes(int) - Number of nodes to create in the hidden layer
            learning_rate(float) - Learning rate to use while training

        """
        np.random.seed(1)

        self.pre_process_data(reviews, labels, polarity_cutoff, min_count)

        self.init_network(len(self.review_vocab), hidden_nodes, 1, learning_rate)

    def pre_process_data(self, reviews, labels, polarity_cutoff, min_count):
        """
        This method preprocesses the review data, calculating positive-to-negative word ratios,
        and building vocabularies for reviews and labels.
        """
        # Calculate positive-to-negative ratios for words before building vocabulary
        pos_neg_ratios, total_counts = self._calculate_pos_neg_ratios(reviews, labels, min_count)

        # populate review_vocab with all of the words in the given reviews
        review_vocab = set()
        for review in reviews:
            for word in review.split(" "):
                # only add words that occur at least min_count times and meet the polarity_cutoff
                if total_counts[word] > min_count:
                    if word in pos_neg_ratios:
                        if (pos_neg_ratios[word] >= polarity_cutoff) or (pos_neg_ratios[word] <= -polarity_cutoff):
                            review_vocab.add(word)
                    else:
                        review_vocab.add(word)

        self.review_vocab = list(review_vocab)
        self.review_vocab_size = len(self.review_vocab)
        # Map words in the vocabulary to index positions
        self.word2index = {word: i for i, word in enumerate(self.review_vocab)}

        # populate label_vocab with all of the words in the given labels.
        label_vocab = set(labels)
        self.label_vocab = list(label_vocab)
        self.label_vocab_size = len(self.label_vocab)
        # Map labels to index positions
        self.label2index = {label: i for i, label in enumerate(self.label_vocab)}

    def _calculate_pos_neg_ratios(self, reviews, labels, min_count):
        """
        This helper method calculates the positive to negative word ratios.
        """
        positive_counts = Counter()
        negative_counts = Counter()
        total_counts = Counter()

        for i, review in enumerate(reviews):
            counts = positive_counts if labels[i] == "POSITIVE" else negative_counts
            for word in review.split(" "):
                counts[word] += 1
                total_counts[word] += 1

        pos_neg_ratios = Counter()

        for term, cnt in total_counts.items():
            if cnt >= min_count:
                pos_neg_ratio = positive_counts[term] / float(negative_counts[term] + 1)
                pos_neg_ratios[term] = pos_neg_ratio

        for word, ratio in pos_neg_ratios.items():
            if ratio > 1:
                pos_neg_ratios[word] = np.log(ratio)
            else:
                pos_neg_ratios[word] = -np.log((1 / (ratio + 0.01)))

        return pos_neg_ratios, total_counts

    def init_network(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        # Set number of nodes in input, hidden and output layers.
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        # Store the learning rate
        self.learning_rate = learning_rate

        # Initialize weights

        # These are the weights between the input layer and the hidden layer.
        self.weights_0_1 = np.zeros((self.input_nodes, self.hidden_nodes))

        # These are the weights between the hidden layer and the output layer.
        self.weights_1_2 = np.random.normal(
            0.0,
            self.output_nodes**-0.5,
            (self.hidden_nodes, self.output_nodes),
        )

        # Removed self.layer_0; added self.layer_1
        # The input layer, a two-dimensional matrix with shape 1 x hidden_nodes
        self.layer_1 = np.zeros((1, hidden_nodes))

    # Removed update_input_layer function

    def get_target_for_label(self, label: str) -> int:
        """
        Given a label, return 1 if it's 'POSITIVE', otherwise return 0.

        Args:
            label (str): The label to evaluate.

        Returns:
            int: 1 if label is 'POSITIVE', else 0.
        """
        return 1 if label == "POSITIVE" else 0

    def sigmoid(self, x: np.ndarray) -> np.ndarray:
        """
        Calculates the sigmoid function.

        Args:
            x (np.ndarray): Input array.

        Returns:
            np.ndarray: The output after applying the sigmoid function.
        """
        return 1 / (1 + np.exp(-x))

    def sigmoid_output_2_derivative(self, output: np.ndarray) -> np.ndarray:
        """
        Computes the derivative of the sigmoid function at output.
        Used for backpropagation in training neural networks.

        Args:
            output (np.ndarray): The output from the sigmoid function.

        Returns:
            np.ndarray: The derivative of the sigmoid function at output.
        """
        return output * (1 - output)

    # changed name of first parameter form 'training_reviews' to 'training_reviews_raw'

    def pre_process_reviews(self, reviews: List[str]) -> List[List[int]]:
        """
        Pre-process reviews to deal directly with the indices of non-zero inputs
        """
        return [[self.word2index[word] for word in review.split() if word in self.word2index] for review in reviews]

    def train(self, training_reviews_raw: List[str], training_labels: List[str]):
        training_reviews = self.pre_process_reviews(training_reviews_raw)

        assert len(training_reviews) == len(training_labels)

        correct_so_far = 0
        start = time.time()

        for i, (review, label) in enumerate(zip(training_reviews, training_labels)):
            self.layer_1 *= 0
            for index in review:
                self.layer_1 += self.weights_0_1[index]

            layer_2 = self.sigmoid(self.layer_1.dot(self.weights_1_2))

            layer_2_error = layer_2 - Label[label].value
            layer_2_delta = layer_2_error * self.sigmoid_output_2_derivative(layer_2)

            layer_1_error = layer_2_delta.dot(self.weights_1_2.T)
            layer_1_delta = layer_1_error

            self.weights_1_2 -= self.layer_1.T.dot(layer_2_delta) * self.learning_rate

            for index in review:
                self.weights_0_1[index] -= layer_1_delta[0] * self.learning_rate

            if (layer_2 >= 0.5 and label == "POSITIVE") or (layer_2 < 0.5 and label == "NEGATIVE"):
                correct_so_far += 1

            elapsed_time = float(time.time() - start)
            reviews_per_second = i / elapsed_time if elapsed_time > 0 else 0

            sys.stdout.write(
                "\rProgress:"
                + str(100 * i / float(len(training_reviews)))[:4]
                + "% Speed(reviews/sec):"
                + str(reviews_per_second)[0:5]
                + " #Correct:"
                + str(correct_so_far)
                + " #Trained:"
                + str(i + 1)
                + " Training Accuracy:"
                + str(correct_so_far * 100 / float(i + 1))[:4]
                + "%"
            )
            if i % 2500 == 0:
                print("")

    def run(self, review: str) -> Label:
        """
        Returns a POSITIVE or NEGATIVE prediction for the given review.
        """
        self.layer_1 *= 0
        unique_indices = {self.word2index[word] for word in review.lower().split() if word in self.word2index}

        for index in unique_indices:
            self.layer_1 += self.weights_0_1[index]

        layer_2 = self.sigmoid(self.layer_1.dot(self.weights_1_2))

        return Label.POSITIVE if layer_2[0] >= 0.5 else Label.NEGATIVE

    def test(self, testing_reviews: List[str], testing_labels: List[str]):
        """
        Attempts to predict the labels for the given testing_reviews,
        and uses the test_labels to calculate the accuracy of those predictions.
        """
        correct = 0
        start = time.time()

        for i, (review, label) in enumerate(zip(testing_reviews, testing_labels)):
            prediction = self.run(review)

            if prediction.name == label:
                correct += 1

            elapsed_time = float(time.time() - start)
            reviews_per_second = i / elapsed_time if elapsed_time > 0 else 0

            sys.stdout.write(
                "\rProgress:"
                + str(100 * i / float(len(testing_reviews)))[:4]
                + "% Speed(reviews/sec):"
                + str(reviews_per_second)[0:5]
                + " #Correct:"
                + str(correct)
                + " #Tested:"
                + str(i + 1)
                + " Testing Accuracy:"
                + str(correct * 100 / float(i + 1))[:4]
                + "%"
            )


def main():
    reviews = read_file("reviews.txt")
    labels = [label.upper() for label in read_file("labels.txt")]

    mlp = SentimentNetwork(
        reviews[:-1000],
        labels[:-1000],
        min_count=20,
        polarity_cutoff=0.8,
        learning_rate=0.01,
    )
    mlp.train(reviews[:-1000], labels[:-1000])


if __name__ == "__main__":
    main()
