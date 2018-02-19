from mission import Mission
from scrap_query import ScrapQuery
from scraper import GoogleScraperWraper
from common import *
from web_page_parser import StackOverflowParser

if __name__ == "__main__":
    sql_db = os.path.join(DATA_DIR, "term_definitions.db")
    vocab_path = os.path.join(DATA_DIR, "vocabulary.txt")
    scraper = GoogleScraperWraper()
    terms = []

    with open(vocab_path, encoding="utf8") as fin:
        for line in fin.readlines():
            term = line.strip("\n\t\r ")
            terms.append(term)

    stk_querys = []
    for term in terms:
        scrap_query = ScrapQuery([term], template="what is {}", domain="stackoverflow.com")
        stk_querys.append(scrap_query)
    stk_parser = StackOverflowParser()
    mission = Mission(sql_db, "stackoverflow", stk_querys, stk_parser, scraper)
    mission.run(delay=1)
