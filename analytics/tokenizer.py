#Things to look at:
# punctuation
# each token is within a whitespace
# math. symbols, operators

PUNCTUATION = [",", ";", "!", "?", "(", ")", "{", "}", "[", "]", "\n", "\"", "&", "$", "#", "@", "<", ">"]
CONFUSING_PUNCTUATION = [".", "/"]
TOKEN_SEPARATORS = [" "]
ALL_PUNCTUATION = PUNCTUATION + CONFUSING_PUNCTUATION


def get_separator_index(string: str, separators: list, start_index=0) -> (str, int):
    best_index = -1
    best_separator = separators[0]

    for separator in separators:
        index = string.find(separator, start_index)
        if best_index == -1 or (0 <= index < best_index):
            best_index = index
            best_separator = separator
    return best_separator, best_index

def split_text(orig_text):
    text = (orig_text + '.')[:-1] # Create a copy
    separator, end_index = get_separator_index(text, TOKEN_SEPARATORS)
    split = []
    while end_index != -1:
        # Extract that word from the string
        word = text[:end_index]
        text = text[end_index+1:]

        # Look at if need to subdivide the word
        subsplits = split_word(word)

        split.extend(subsplits)

        end_index = text.find(" ")
    subsplits = split_word(text)
    split.extend(subsplits)
    return split

def split_word(orig_word):
    word = (orig_word + '.')[:-1] # Create a copy
    split = []

    separator, index = get_separator_index(word, PUNCTUATION + CONFUSING_PUNCTUATION)
    while index >= 0:
        start_index = 0
        special_punctuation = False
        first_split = word[:index]
        punctuation = word[index:index+1]
        end_split = word[index+1:]

        # if there is a number after a period, combine as single word
        if punctuation == "." and len(end_split) > 0:
            if end_split[0] != "\n":
                start_index = index + 1
                separator, index = get_separator_index(word, PUNCTUATION + CONFUSING_PUNCTUATION,
                                                       start_index=start_index)
                special_punctuation = True

        # Special operation: skip this punctuation
        if special_punctuation:
            continue

        # Normal operation: Save word
        if index != 0:
            split.append(first_split)
        split.append(punctuation)
        if index+1 <= len(word):
            word = end_split

        separator, index = get_separator_index(word, PUNCTUATION + CONFUSING_PUNCTUATION, start_index=start_index)

    # Last end of word
    if len(word) > 0:
        split.append(word)

    return split


if __name__ == "__main__":
    # text = "Several mathematical operations are provided for combining Counter objects to produce multisets (counters that have counts greater than zero). Addition and subtraction combine counters by adding or subtracting the counts of corresponding elements."
    text = "Canada Oil and Gas Operations Act ( R.S., 1985, c. O-7 )\n\n The following are numbers: 1.42, -3, 483, 1E-4, -23.1. This is a newline test.\nHello. asdf"
    splits = split_text(text)
    print(splits)
