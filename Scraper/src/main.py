from mission import Mission
from scrap_query import ScrapQuery
from scraper import GoogleScraperWraper
from common import *
from web_page_parser import StackOverflowParser, QuoraParser, PcMagParser, RegularParagraphParser


def load_terms(vocab_path):
    terms = []
    with open(vocab_path, encoding="utf8") as fin:
        for line in fin.readlines():
            term = line.strip("\n\t\r ")
            terms.append(term)
    return terms


def run_stackoverflow_mission(sql_db, terms, scraper, use_proxy):
    stk_querys = []
    for term in terms:
        scrap_query = ScrapQuery([term], template="what is \"{}\"", domain="stackoverflow.com")
        stk_querys.append(scrap_query)

    stk_parser = StackOverflowParser()
    mission = Mission(sql_db, "stackoverflow", stk_querys, stk_parser, scraper, use_proxy)
    mission.run(delay=0.2, thread_num=4)


def run_quora_mission(sql_db, terms, scraper, use_proxy):
    quora_querys = []
    for term in terms:
        scrap_query = ScrapQuery([term], template="what is \"{}\"", domain="quora.com")
        quora_querys.append(scrap_query)

    quora_parser = QuoraParser()
    mission = Mission(sql_db, "quora", quora_querys, quora_parser, scraper, use_proxy)
    mission.run(delay=0.2, thread_num=4)


def run_pcMag_mission(sql_db, terms, scraper, use_proxy):
    pcMag_queries = []
    for term in terms:
        scrap_query = ScrapQuery([term], template="definition of \"{}\"", domain="pcmag.com/encyclopedia/")
        pcMag_queries.append(scrap_query)
    pcMag_parser = PcMagParser()
    mission = Mission(sql_db, "pcMag", pcMag_queries, pcMag_parser, scraper, use_proxy)
    mission.run(delay=0, thread_num=4, link_limit=1)


def run_regularParse_mission(sql_db, terms, scraper, use_proxy):
    regular_terms = []
    for term in terms:
        scrap_query = ScrapQuery([term], template="definition of \"{}\"")
        regular_terms.append(scrap_query)
    regular_parser = RegularParagraphParser()
    mission = Mission(sql_db, "regular", regular_terms, regular_parser, scraper, use_proxy)
    mission.run(delay=0, thread_num=4)


if __name__ == "__main__":
    proxies = os.path.join(DATA_DIR, "proxy_list.txt")
    sql_db = os.path.join(DATA_DIR, "term_definitions.db")
    vocab_path = os.path.join(DATA_DIR, "vocabulary.txt")
    scraper = GoogleScraperWraper(proxies)
    dry_run = False
    if dry_run == True:
        terms = ["Objective-C", "Scala", "Swift", "Shell", "TypeScript", "go", "C#", "CSS"]
    else:
        terms = load_terms(vocab_path)

    run_pcMag_mission(sql_db, terms, scraper, use_proxy=True)
    run_stackoverflow_mission(sql_db, terms, scraper, use_proxy=True)
    run_quora_mission(sql_db, terms, scraper, use_proxy=True)
    run_regularParse_mission(sql_db, terms, scraper, use_proxy=True)
