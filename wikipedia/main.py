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
    bad_words = ["article", "Article", "Wikipedia", "dates", "Pages", "Wikidata",
                 "reference errors", "Webarchive", "errors:", "Source attribution",
                 "maint:", "pages", "CS1"]
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
def get_n_links_from_page_categories(page, n, num_categories=5):
    num_extract_per_cat = int(n / num_categories)

    categories, c_titles = get_n_categories_of_page(page, num_categories)

    links = []
    link_titles = []

    for c in categories:
        print("category: " + c.title)
        subpages, p_titles = get_n_pages_from_category(c, num_extract_per_cat)
        links += subpages
        link_titles += p_titles
    return links, link_titles

def save_page(folder, page, index="0"):
    # Subfolder for this save
    subpath = paths.WIKIPEDIA_DATA_PATH + str(index) + "/"
    if not os.path.isdir(subpath):
        os.mkdir(subpath)
    # holds extractions
    path = subpath + clean_name(folder) + "/"
    if not os.path.isdir(path):
        os.mkdir(path)
    # Holds uneditted extractions
    path_uneditted = path + paths.UNALTERED_FOLDER_NAME + "/"
    if not os.path.isdir(path_uneditted):
        os.mkdir(path_uneditted)

    name = page.title
    with open(path_uneditted + clean_name(name) + "_u.txt", 'w') as f:
        f.write(page.text)
    with open(path + clean_name(name) + ".txt", 'w') as f:
        text = tp.remove_text_after_sections(["References", "See also"], page.text)
        text = tp.remove_content_in_brackets(text)
        text = tp.remove_whitespace(text)
        f.write(text)

def clean_name(name):
    return name.replace("/", "", 10)

# returns n^depth pages
# Algorithm:
# At each depth, sample n linked pages. linked pages will be from related categories and direct page links
# For each linked page, save and sample n more pages
def get_n_links_from_page_with_depth(page, n, exploit_explore_ratio=0.5, max_depth=1, cur_depth = 0, save_index="0"):
    n_direct_links = int(exploit_explore_ratio * n)
    n_category_links = n - n_direct_links

    links_c, _ = get_n_links_from_page_categories(page, n_category_links)
    links_d, _ = get_n_links_from_page(page, n_direct_links)
    links = links_d + links_c

    for link in links:
        save_page("", link, index=save_index)
    if cur_depth + 1 == max_depth:
        return
    else:
        for link in links:
            get_n_links_from_page_with_depth(link, n, exploit_explore_ratio, max_depth, cur_depth + 1, save_index)


if __name__ == "__main__":
    wiki_wiki = wikipediaapi.Wikipedia('MyProjectName (merlin@example.com)',
                                       'en',
                                       extract_format=wikipediaapi.ExtractFormat.WIKI)

    origin_page_titles = ['Technical standard', 'IEEE Standards Association', 'List of technical standard organizations']

    now = datetime.datetime.now()
    save_index = now.strftime('%h_%m_%s')

    for title in origin_page_titles:
        p_wiki = wiki_wiki.page(title)
        print("Origin Page - Summary: %s" % p_wiki.summary[0:60])

        get_n_links_from_page_with_depth(page=p_wiki, n=7, exploit_explore_ratio=0.5, max_depth=2, cur_depth=0, save_index=save_index)

