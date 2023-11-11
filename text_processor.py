import re


# TODO: only remove nl if next to single character
def remove_whitespace(text):
    new_text = ""
    # check for newline
    # check for next newline
    # if space is empty in between, remove everything to the next newline
    # if space is not empty, change index
    next_i = text.find("\n")
    last_contained_length = 0
    while next_i != -1:
        contained = text[:next_i]
        contained_length = len(contained.strip())
        if contained_length == 1:
            # remove all whitespace
            new_text += contained.strip()
            text = text[next_i + 1:]
        # contained value is empty
        elif re.match(r'^\s*$', contained) and last_contained_length == 1:
            # move to next index
            text = text[next_i+1:]
        # Normal: keep searching
        else:
            new_text += "\n"
            new_text += contained
            text = text[next_i + 1:]
        next_i = text.find("\n")
        if contained_length > 0:
            last_contained_length = contained_length
    return new_text



if __name__ == "__main__":
    text = "1\n2\n \n3\n4\n"
    print("orig")
    print(text)
    print("remove")
    print(remove_whitespace(text))