import sys

sys.path.append("../../Cleaner")
import urllib.request
from lxml import html
from glossary_config import *
import cleaner
import re

from urllib3 import make_headers


class OnlineGlossaryExtractor:
    def clean_doc(self, doc):
        doc = re.sub("[\n\t\r:]", " ", doc)
        doc = cleaner.remove_content_in_bracket(doc)
        doc = cleaner.keep_only_given_chars(doc)
        doc = doc.strip("\n\t\r ")
        doc = cleaner.merge_white_space(doc)
        return doc

    def __init__(self, link):
        headers = make_headers()
        request = urllib.request.Request(link, None, headers)
        with urllib.request.urlopen(request, timeout=20) as url:
            html_page = url.read()
            if html_page == "" or html_page == None:
                return []
            self.html_tree = html.fromstring(html_page)


class FFSEExtractor(OnlineGlossaryExtractor):
    def __init__(self):
        link = "http://soft.vub.ac.be/FFSE/SE-contents.html"
        super().__init__(link)
        self.output_file_path = os.path.join(DATA_DIR, "software_engineering", "FFSE_glossary.txt")

    def parse(self):
        with open(self.output_file_path, "w") as fout:
            components = self.html_tree.xpath("//p")
            cur_index = self.next_term_index(components, 0)
            if cur_index is None:
                return
            next_index = self.next_term_index(components, cur_index + 1)
            while cur_index is not None:
                term_chunk = components[cur_index:next_index]
                term = term_chunk[0].text_content()
                definition = " ".join([x.text_content() for x in term_chunk[1:]])
                definition = super().clean_doc(definition)
                term = super().clean_doc(term)
                fout.write("{}:{}\n".format(term, definition))
                cur_index = next_index
                next_index = self.next_term_index(components, cur_index + 1)
                if next_index == cur_index + 1:
                    break

    def next_term_index(self, components_list, cur_index):
        for i in range(cur_index, len(components_list)):
            cur_component = components_list[i]
            class_name = cur_component.xpath(".//@class")
            if len(class_name) > 0:
                class_name = class_name[0]
            if class_name == "Term" or class_name == "Reference":
                return i
        return len(components_list)


class REGlossory1(OnlineGlossaryExtractor):
    def __init__(self):
        link = "http://www.processimpact.com/UC/Module_3/data/downloads/glossary.html"
        self.output_file_path = os.path.join(DATA_DIR, "requirement_engineering", "processimpact.txt")
        super().__init__(link)

    def parse(self):
        with open(self.output_file_path, "w") as fout:
            components = self.html_tree.xpath("//p")
            for component in components:
                texts = component.xpath(".//text()")
                term = super().clean_doc(texts[0])
                definition = super().clean_doc(" ".join(texts[1:]))
                fout.write("{}:{}\n".format(term, definition))


class AgileGlossory1(OnlineGlossaryExtractor):
    def __init__(self):
        link = "https://www.agilealliance.org/agile101/agile-glossary/"
        self.output_file_path = os.path.join(DATA_DIR, "agile_development", "agile_alliance.txt")
        super().__init__(link)

    def parse(self):
        with open(self.output_file_path, "w") as fout:
            components = self.html_tree.xpath(
                "//div[@class='wpb_text_column wpb_content_element  aa_unordered-list-padding']//div[@class='wpb_wrapper']")
            for component in components:
                term = component.xpath(".//h2/a")
                definition = component.xpath(".//p")
                if len(term) > 0:
                    term = term[0].text_content()
                    if len(term) == 1:  # A-Z which is not a valid result
                        continue
                    definition = definition[0].text_content()
                    term = self.clean_doc(term)
                    term = term.lower()
                    definition = self.clean_doc(definition)
                    fout.write("{}:{}\n".format(term, definition))


if __name__ == "__main__":
    # ffs_extractor = FFSEExtractor()
    # ffs_extractor.parse()
    # re1 = REGlossory1()
    # re1.parse()
    agile1 = AgileGlossory1()
    agile1.parse()
