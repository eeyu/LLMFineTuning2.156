import os

import wikipediaapi
import random
import re
import text_processor as tp
import datetime
import paths

# Returns n linked pages from a given page
def get_n_links_from_page(page, n):
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

# returns N categories associated with a page
# Removes irrelevant categories
def get_n_categories_of_page(page, n):
    categories = page.categories
    category_titles = list(categories.keys())
    category_titles = list(filter(normal_category_filter, category_titles))
    random.shuffle(category_titles)
    n = min(len(category_titles), n)

    n_categories = []
    titles = []
    for i in range(n):
        category = categories[category_titles[i]]
        n_categories.append(category)
        titles.append(category.title)
    return n_categories, titles


# Returns False for any generic category ex. "articles containing ___"
def normal_category_filter(c_title):
    bad_words = ["article", "Article", "Wikipedia", "dates", "Pages", "Wikidata", "reference errors", "Webarchive", "errors:"]
    condition = True
    for word in bad_words:
        condition = condition and (word not in c_title)
    return condition


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


# Returns n pages that share categories with the given page
def get_n_links_from_page_categories(page, n):
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

def save_page(folder, page, index="0"):
    subpath = paths.WIKIPEDIA_DATA_PATH + str(index) + "/"
    if not os.path.isdir(subpath):
        os.mkdir(subpath)
    path = subpath + clean_name(folder) + "/"
    if not os.path.isdir(path):
        os.mkdir(path)
    name = page.title

    with open(path + clean_name(name) + "_u.txt", 'w') as f:
        f.write(page.text)
    with open(path + clean_name(name) + ".txt", 'w') as f:
        text = tp.remove_text_after_sections(["References", "See also"], page.text)
        text = tp.remove_content_in_brackets(text)
        text = tp.remove_whitespace(text)
        f.write(text)

def clean_name(name):
    return name.replace("/", "", 10)

if __name__ == "__main__":
    wiki_wiki = wikipediaapi.Wikipedia('MyProjectName (merlin@example.com)',
                                       'en',
                                       extract_format=wikipediaapi.ExtractFormat.WIKI)

    p_wiki = wiki_wiki.page('gravity')

    print("Origin Page - Summary: %s" % p_wiki.summary[0:60])

    links_c, l_c_titles = get_n_links_from_page_categories(p_wiki, 10)
    links, l_titles = get_n_links_from_page(p_wiki, 10)

    now = datetime.datetime.now()
    save_index = now.strftime('%h_%m_%s')

    # random_save = random.randint(0, 100000)

    for l in links_c:
        print("link_c: " + l.title)
        save_page("", l, index=save_index)

    for l in links:
        print("link: " + l.title)
        save_page("", l, index=save_index)


