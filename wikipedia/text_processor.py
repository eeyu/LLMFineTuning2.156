import re


# TODO: only remove nl if next to single character
# If last item was single character or this item was sngle character, do not append nl
def remove_whitespace(text):
    new_text = ""
    # check for newline
    # check for next newline
    # if space is empty in between, remove everything to the next newline
    # if space is not empty, change index
    next_i = text.find("\n")
    last_contained_length = 0
    last_nonempty_contained_length = 0
    while next_i != -1:
        contained = text[:next_i]
        contained_length = len(contained.strip())
        if contained_length == 1:
            # remove all whitespace
            new_text += contained.strip()
            text = text[next_i + 1:]
        # contained value is empty
        elif re.match(r'^\s*$', contained) and last_nonempty_contained_length == 1:
            # move to next index
            text = text[next_i+1:]
        # Normal: keep searching
        else:
            if not (last_nonempty_contained_length == 1 or contained_length == 1 or (last_contained_length == 0 and contained_length == 0)):
                new_text += "\n"
            new_text += contained
            text = text[next_i + 1:]
        next_i = text.find("\n")
        if contained_length > 0:
            last_nonempty_contained_length = contained_length
        last_contained_length = contained_length
    return new_text

def remove_text_after_sections(words, text):
    for word in words:
        index = text.find(word)
        text = text[0:index]
    return text

def remove_content_in_brackets(text):
    start_i = text.find("{")

    while start_i != -1:
        end_i = text.find("}")
        # Check if need to exit inner loop
        if end_i < start_i:
            return text
        # Check for inner loop
        next_start_i = text.find("{", start_i+1)
        # Inner loop exists
        if next_start_i != -1 and next_start_i < end_i:
            text = text[0:next_start_i] + remove_content_in_brackets(text[next_start_i:])
        # No inner loop - proceed as normal
        else:
            text = text[0:start_i] + text[end_i+1:]
        start_i = text.find("{")
    return text

if __name__ == "__main__":
    text = "1\n2\n \n3\n4\n"
    print("orig")
    print(text)
    print("remove")
    print(remove_whitespace(text))