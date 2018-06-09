import os
import unittest

from mission import Mission
from scrap_query import ScrapQuery
from scraper import GoogleScraperWraper, DATA_DIR
from web_page_parser import StackOverflowParser, QuoraParser


class MissionExecutions(unittest.TestCase):
    def setUp(self):
        self.sql_db = os.path.join(DATA_DIR, "term_definitions.db")
        vocab_path = os.path.join(DATA_DIR, "vocabulary.txt")
        self.scraper = GoogleScraperWraper()
        self.terms = []

        with open(vocab_path, encoding="utf8") as fin:
            for line in fin.readlines():
                term = line.strip("\n\t\r ")
                self.terms.append(term)

    def stackoverflow_execution(self):
        stk_querys = []
        for term in self.terms:
            scrap_query = ScrapQuery([term], template="what is {}", domain="stackoverflow.com")
            stk_querys.append(scrap_query)
        stk_parser = StackOverflowParser()
        mission = Mission(self.sql_db, "stackoverflow", stk_querys, stk_parser, self.scraper)
        mission.run(delay=1)

    def quora_execution(self):
        quora_queries = []
        for term in self.terms:
            scrap_query = ScrapQuery([term], template="what is {}", domain=".quora.com")
            quora_queries.append(scrap_query)
        quora_parser = QuoraParser()
        mission = Mission(self.sql_db, "quora", quora_queries, quora_parser, self.scraper)
        mission.run(delay=1)
