from common import *
from lxml import html
import json


class StackOverflowParser:
    def __get_text_of_post_block(self, post_element):
        """
        Get the text in a post block html element
        :param post_element: An html element object
        :return: A string of the answer/question as a post
        """
        paragraphs_element = post_element.xpath(".//p")
        paragraph_texts = []
        for paragraph in paragraphs_element:
            para_text = paragraph.text_content()
            paragraph_texts.append(para_text)
        post_text = "\n".join(paragraph_texts)
        return post_text

    def get_question(self, html_page):
        if html_page == "" or html_page == None:
            return []
        html_tree = html.fromstring(html_page)
        text_blocks = html_tree.xpath('//td[@class="postcell"]')
        questions = []
        for element in text_blocks:
            questions.append(self.__get_text_of_post_block(element))
        return questions

    def get_answers(self, html_page):
        if html_page == "" or html_page == None:
            return []
        html_tree = html.fromstring(html_page)
        text_blocks = html_tree.xpath('//td[@class="answercell"]')
        answers = []
        for element in text_blocks:
            answers.append(self.__get_text_of_post_block(element))
        return answers

    def get_related_question_links(self, html_page):
        """
        Get the related question and its links
        :return:
        """
        if html_page == "" or html_page == None:
            return []
        html_tree = html.fromstring(html_page)
        related_questions = []
        related_links_div = html_tree.xpath('//div[contains(@class,"related")]//a[@class="question-hyperlink"]')
        for link in related_links_div:
            related_questions.append((link.text_content(), link.xpath("@href")[0]))
        return related_questions

    def parse(self, html, query):
        """
        Get parse the html page with all the methods available. And convert the result into Json which can be stored in db.
        All parser should implement this function.
        :param html:
        :param query
        :return: Json string of the parse result.
        """
        questions = self.get_question(html)
        answers = self.get_answers(html)
        related = self.get_related_question_links(html)
        res_book = {}
        res_book["questions"] = questions
        res_book["answers"] = answers
        res_book["related_questions"] = related
        return json.dumps(res_book)


class QuoraParser:
    def get_question(self, html_page):
        if html_page == "" or html_page == None:
            return []
        html_tree = html.fromstring(html_page)
        text_blocks = html_tree.xpath('//h1//span[@class="rendered_qtext"]')
        questions = []
        for element in text_blocks:
            questions.append(element.text_content())
        return questions

    def get_answers(self, html_page):
        if html_page == "" or html_page == None:
            return []
        html_tree = html.fromstring(html_page)
        text_blocks = html_tree.xpath('//span[@class="ui_qtext_rendered_qtext"]')
        answers = []
        for element in text_blocks:
            answers.append(element.text_content())
        return answers

    def get_related_question_links(self, html_page):
        if html_page == "" or html_page == None:
            return []
        html_tree = html.fromstring(html_page)
        related_links_a = html_tree.xpath("//a[@class ='question_link']")
        related_questions = []
        for link in related_links_a:
            related_questions.append((link.text_content(), link.xpath("@href")[0]))
        return related_questions

    def parse(self, html_page, query):
        questions = self.get_question(html_page)
        answers = self.get_answers(html_page)
        related = self.get_related_question_links(html_page)
        res_book = {}
        res_book["questions"] = questions
        res_book["answers"] = answers
        res_book["related_questions"] = related
        return json.dumps(res_book)


class PcMagParser:
    def get_definition(self, html_page):
        if html_page == "" or html_page == None:
            return ""
        html_tree = html.fromstring(html_page)
        def_element = html_tree.xpath("//div[@class ='cde_definition']")
        definition = ""
        if len(def_element) > 0:
            definition = def_element[0].text_content()
        return definition

    def get_title(self, html_page):
        if html_page == "" or html_page == None:
            return ""
        html_tree = html.fromstring(html_page)
        title_element = html_tree.xpath("//span[@class ='term_title']")
        title = ""
        if len(title_element) > 0:
            title = title_element[0].text_content()
            prefix = "Definition of:"
            title = title[len(prefix):].strip("\n\t\r ")
        return title

    def parse(self, html_page, query):
        res_book = {}
        res_book["term"] = ""
        res_book["definition"] = ""
        title = self.get_title(html_page)
        definition = self.get_definition(html_page)
        if title in query:
            res_book["term"] = title
            res_book["definition"] = definition
        return json.dumps(res_book)


class RegularParagraphParser:
    """
    Parse all text in <p> from any given html page. This is used for pages we don't know the format
    """
    def get_paragraph(self, html_page):
        if html_page == "" or html_page == None:
            return ""
        html_tree = html.fromstring(html_page)
        text_blocks = html_tree.xpath('//p')
        texts = []
        for element in text_blocks:
            texts.append(element.text_content())
        return texts

    def parse(self, html_page, query):
        res_book = {}
        res_book["content"] = self.get_paragraph(html_page)
        return json.dumps(res_book)
