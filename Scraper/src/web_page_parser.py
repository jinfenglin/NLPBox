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
        html_tree = html.fromstring(html_page)
        text_blocks = html_tree.xpath('//td[@class="postcell"]')
        questions = []
        for element in text_blocks:
            questions.append(self.__get_text_of_post_block(element))
        return questions

    def get_answers(self, html_page):
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
        html_tree = html.fromstring(html_page)
        related_questions = []
        related_links_div = html_tree.xpath('//div[contains(@class,"related")]//a[@class="question-hyperlink"]')
        for link in related_links_div:
            related_questions.append((link.text_content(), link.xpath("@href")[0]))
        return related_questions

    def parse(self, html):
        """
        Get parse the html page with all the methods available. And convert the result into Json which can be stored in db.
        All parser should implement this function.
        :param html:
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


if __name__ == "__main__":
    with open(os.path.join(DATA_DIR, "stackoverflow_test.html"), encoding="utf8") as fin:
        html_page = fin.read()
        stkP = StackOverflowParser()
        print("question:", stkP.get_question(html_page))
        print("answer:", stkP.get_answers(html_page))
        print("Related Links:", stkP.get_related_question_links(html_page))
        stkP.parse(html_page)
