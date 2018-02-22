import unittest
from common import *
from web_page_parser import StackOverflowParser, QuoraParser, PcMagParser, RegularParagraphParser, InnolutionParser


class TestPagerParser(unittest.TestCase):
    def test_stackoverflow_parser(self):
        with open(os.path.join(DATA_DIR, "stackoverflow_test.html"), encoding="utf8") as fin:
            html_page = fin.read()
            stkP = StackOverflowParser()
            print("question:", stkP.get_question(html_page))
            print("answer:", stkP.get_answers(html_page))
            print("Related Links:", stkP.get_related_question_links(html_page))
            stkP.parse(html_page, "")

    def test_quora_parser(self):
        with open(os.path.join(DATA_DIR, "quora_test.html"), encoding="utf8") as fin:
            html_page = fin.read()
            quoraP = QuoraParser()
            print("question:", quoraP.get_question(html_page))
            print("answer:", quoraP.get_answers(html_page))
            print("Related Links:", quoraP.get_related_question_links(html_page))
            quoraP.parse(html_page, "")

    def test_pcMag(self):
        with open(os.path.join(DATA_DIR, "pc_mag.html"), encoding="utf8") as fin:
            html_page = fin.read()
            pcMagP = PcMagParser()
            query = "definition of 'ambient lighting' site:pcmag.com/encyclopedia/"
            print(pcMagP.parse(html_page, query))

    def test_innolution(self):
        with open(os.path.join(DATA_DIR, "innolution.html"), encoding="utf8") as fin:
            html_page = fin.read()
            pcMagP = InnolutionParser()
            print(pcMagP.parse(html_page, ""))

    def test_paragraphParser(self):
        with open(os.path.join(DATA_DIR, "pc_mag.html"), encoding="utf8") as fin:
            html_page = fin.read()
            regParser = RegularParagraphParser()
            print(regParser.parse(html_page, ""))
