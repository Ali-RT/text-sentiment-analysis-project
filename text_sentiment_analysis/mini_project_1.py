from collections import Counter

import numpy as np


def pretty_print_review_and_label(i, reviews, labels):
    print(f"{labels[i]} :\t {reviews[i][:80]}...")


def read_file(file_path):
    with open(file_path, "r") as file:
        content = [line.strip() for line in file]
    return content


def count_words(texts, labels, label_to_count):
    total_counts = Counter()
    label_counts = {label: Counter() for label in label_to_count}

    for text, label in zip(texts, labels):
        words = text.split(" ")
        total_counts.update(words)
        label_counts[label].update(words)

    return total_counts, label_counts


def filter_counts(counters, min_count):
    return {word: count for word, count in counters.items() if count >= min_count}


def calculate_ratios(positive_counts, negative_counts):
    ratios = Counter()
    for word in positive_counts:
        ratios[word] = positive_counts[word] / float(negative_counts.get(word, 0) + 1)
    return ratios


def log_ratios(ratios):
    for word, ratio in ratios.items():
        ratios[word] = np.log(ratio)
    return ratios


def main():
    reviews = read_file("reviews.txt")
    labels = [label.upper() for label in read_file("labels.txt")]

    for i in range(len(reviews[:5])):
        pretty_print_review_and_label(i, reviews, labels)
    total_counts, label_counts = count_words(reviews, labels, ["POSITIVE", "NEGATIVE"])

    filtered_positive_counts = filter_counts(label_counts["POSITIVE"], 100)
    filtered_negative_counts = filter_counts(label_counts["NEGATIVE"], 100)

    ratios = calculate_ratios(filtered_positive_counts, filtered_negative_counts)
    log_ratios_values = log_ratios(ratios)
    print(log_ratios_values)


if __name__ == "__main__":
    main()
