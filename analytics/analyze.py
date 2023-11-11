from dataclasses import dataclass
import collections
import re

PUNCTUATION = [",", ";", "!", "?", "(", ")", "{", "}", "[", "]"]
confusing_punctuation = [".", "/"]
token_separators = [" ", "\n"]

#Things to look at:
# punctuation
# each token is within a whitespace
# math. symbols, operators
@dataclass
class StringAnalytics:
    num_tokens: int
    token_histogram: collections.Counter

    def print_histogram(self):
        for token, count in self.token_histogram.items():
            print(token, count)


def get_separator_index(string: str, separators: list) -> (str, int):
    best_index = -1
    best_separator = separators[0]
    for separator in separators:
        index = string.find(separator)
        if best_index == -1 or (0 <= index < best_index):
            best_index = index
            best_separator = separator
    return best_separator, best_index

def split_string(orig_string):
    string = (orig_string + '.')[:-1] # Create a copy
    separator, end_index = get_separator_index(string, token_separators)
    split = []
    while end_index != -1:
        # Extract that word from the string
        word = string[:end_index]
        string = string[end_index+1:]

        # Look at if need to subdivide the word
        subsplits = split_word(word)

        split.extend(subsplits)

        end_index = string.find(" ")
    split.extend(string)
    return split

def split_word(orig_word):
    word = (orig_word + '.')[:-1] # Create a copy
    split = []

    separator, index = get_separator_index(word, PUNCTUATION + confusing_punctuation)
    while index >= 0:
        if index != 0:
            first_split = word[:index]
            split.append(first_split)

        punctuation = word[index:index+1]
        split.append(punctuation)

        if index+1 <= len(word):
            word = word[index+1:]

        separator, index = get_separator_index(word, PUNCTUATION + confusing_punctuation)
    if len(word) > 0:
        split.append(word)
    print(split)
    return split

# Analytics for a single page
def analyze_string(string: str):
    tokens = string.split()
    num_tokens = len(tokens)
    token_histogram = collections.Counter(tokens)
    return StringAnalytics(num_tokens=num_tokens, token_histogram=token_histogram)


if __name__ == "__main__":
    # text = "Several mathematical operations are provided for combining Counter objects to produce multisets (counters that have counts greater than zero). Addition and subtraction combine counters by adding or subtracting the counts of corresponding elements."
    text = "Canada Oil and Gas Operations Act ( R.S., 1985, c. O-7 )"
    # analytics = analyze_string(text)
    # analytics.print_histogram()
    splits = split_string(text)
    print(splits)

