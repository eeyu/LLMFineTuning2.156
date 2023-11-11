import os

import wikipediaapi
import random
import re
import text_processor as tp


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

def print_categories(page):
    categories = page.categories

    for title in sorted(categories.keys()):
        print("%s: %s" % (title, categories[title]))

def get_n_links_of_page(page, n):
    links = page.links
    link_titles = list(links.keys())
    link_titles = list(filter(lambda t: "Category" not in t, link_titles))
    random.shuffle(link_titles)

    n_links = []
    titles = []
    for i in range(n):
        page = links[link_titles[i]]
        n_links.append(page)
        titles.append(page.title)
    return n_links, titles

def get_n_categories_of_page(page, n):
    categories = page.categories
    category_titles = list(categories.keys())
    category_titles = list(filter(normal_category_filter, category_titles))
    random.shuffle(category_titles)

    n_categories = []
    titles = []
    for i in range(n):
        category = categories[category_titles[i]]
        n_categories.append(category)
        titles.append(category.title)
    return n_categories, titles

def get_n_pages_from_category(category, n):
    members = category.categorymembers
    member_titles = list(members.keys())

    random.shuffle(member_titles)
    member_titles = list(filter(lambda t: "Category" not in t, member_titles))
    n_members = []
    titles = []
    n = min(n, len(member_titles))
    for i in range(n):
        page = members[member_titles[i]]
        n_members.append(page)
        titles.append(page.title)
    return n_members, titles

def save_page(folder, page, index=0):
    subpath = "./texts/" + str(index) + "/"
    if not os.path.isdir(subpath):
        os.mkdir(subpath)
    path = subpath + clean_name(folder) + "/"
    if not os.path.isdir(path):
        os.mkdir(path)
    name = page.title
    with open(path + clean_name(name) + "_u.txt", 'w') as f:
        f.write(page.text)

    with open(path + clean_name(name) + ".txt", 'w') as f:
        text = remove_text_after_sections(["References", "See also"], page.text)
        text = remove_content_in_brackets(text)
        text = tp.remove_whitespace(text)
        f.write(text)


def normal_category_filter(c_title):
    bad_words = ["article", "Article", "Wikipedia", "dates", "Pages", "Wikidata", "reference errors"]
    condition = True
    for word in bad_words:
        condition = condition and (word not in c_title)
    return condition

def get_links_from_categories(page, n):
    num_cat = 5
    num_extract_per_cat = int(n / num_cat)

    categories, c_titles = get_n_categories_of_page(page, num_cat)

    links = []
    link_titles = []

    for c in categories:
        print("category: " + c.title)
        subpages, p_titles = get_n_pages_from_category(c, num_extract_per_cat)
        links += subpages
        link_titles += p_titles
    return links, link_titles

def clean_name(name):
    return name.replace("/", "", 10)

if __name__ == "__main__":
    # text = "1{2{5}6{7}}3{4}8"
    # print(remove_content_in_brackets(text))
    wiki_wiki = wikipediaapi.Wikipedia('MyProjectName (merlin@example.com)',
                                       'en',
                                       extract_format=wikipediaapi.ExtractFormat.WIKI)
    #
    # p_wiki = wiki_wiki.page('Digital image processing')
    # save_page("test", p_wiki, 9)


    p_wiki = wiki_wiki.page('Python_(programming_language)')

    print("Origin Page - Summary: %s" % p_wiki.summary[0:60])

    links_c, l_c_titles = get_links_from_categories(p_wiki, 10)
    links, l_titles = get_n_links_of_page(p_wiki, 10)

    random_save = random.randint(0, 100000)

    for l in links_c:
        print("link_c: " + l.title)
        save_page("links_c", l, index=random_save)

    for l in links:
        print("link: " + l.title)
        save_page("links", l, index=random_save)


