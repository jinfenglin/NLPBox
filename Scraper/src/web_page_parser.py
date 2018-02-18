from common import *
from lxml import html


class HtmlPageParser:
    def __init__(self, html_page):
        self.html_page = html_page
        self.html_tree = html.fromstring(self.html_page)


class StackOverflowParser(HtmlPageParser):
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

    def get_question(self):
        text_blocks = self.html_tree.xpath('//td[@class="postcell"]')
        questions = []
        for element in text_blocks:
            questions.append(self.__get_text_of_post_block(element))
        return questions

    def get_answers(self):
        text_blocks = self.html_tree.xpath('//td[@class="answercell"]')
        answers = []
        for element in text_blocks:
            answers.append(self.__get_text_of_post_block(element))
        return answers

    def get_related_question_links(self):
        """
        Get the related question and its links
        :return:
        """
        related_questions = {}
        related_links_div = self.html_tree.xpath('//div[contains(@class,"related")]//a[@class="question-hyperlink"]')
        for link in related_links_div:
            related_questions[link.text_content()] = link.xpath("@href")[0]
        return related_questions

    def debug(self):
        print("question:", self.get_question())
        print("answer:", self.get_answers())
        print(self.get_related_question_links())


if __name__ == "__main__":
    with open(os.path.join(DATA_DIR, "stackoverflow_test.html"), encoding="utf8") as fin:
        html_page = fin.read()
        stkP = StackOverflowParser(html_page)
        print("question:", stkP.get_question())
        print("answer:", stkP.get_answers())
        print("Related Links:", stkP.get_related_question_links())
