from mission import Mission, Sqlite3Manger
from scrap_query import ScrapQuery
from scraper import GoogleScraperWraper
from common import *
import json
import wikipedia
from web_page_parser import StackOverflowParser, QuoraParser, PcMagParser, RegularParagraphParser, InnolutionParser
import logging


def load_terms(vocab_path):
    terms = []
    with open(vocab_path, encoding="utf8") as fin:
        for line in fin.readlines():
            term = line.strip("\n\t\r ")
            terms.append(term)
    return terms


def load_definitoin(definition_path):
    definition = {}
    with open(definition_path, encoding="utf8") as fin:
        for line in fin.readlines():
            line = line.strip("\n\t\r ")
            split_point = line.rfind(":")
            term = line[:split_point]
            def_doc = line[split_point + 1:]
            definition[term] = def_doc
    return definition


def run_stackoverflow_mission(sql_db, terms, scraper, use_proxy, overriding_existing=True):
    stk_querys = []
    for term in terms:
        scrap_query = ScrapQuery([term], template="what is \"{}\"", domain="stackoverflow.com")
        stk_querys.append(scrap_query)

    stk_parser = StackOverflowParser()
    mission = Mission(sql_db, "stackoverflow", stk_querys, stk_parser, scraper, use_proxy)
    mission.run(delay=0.2, process_num=4, override_existing=overriding_existing)


def run_quora_mission(sql_db, terms, scraper, use_proxy, overriding_existing=True, topic="software engineering"):
    quora_querys = []
    for term in terms:
        scrap_query = ScrapQuery([term], template="what is \"{}\" in " + topic, domain="quora.com")
        quora_querys.append(scrap_query)

    quora_parser = QuoraParser()
    mission = Mission(sql_db, "quora", quora_querys, quora_parser, scraper, use_proxy)
    mission.run(delay=0.2, process_num=4, override_existing=overriding_existing)


def run_pcMag_mission(sql_db, terms, scraper, use_proxy, overriding_existing=False):
    pcMag_queries = []
    for term in terms:
        scrap_query = ScrapQuery([term], template="definition of \"{}\"", domain="pcmag.com/encyclopedia/")
        pcMag_queries.append(scrap_query)
    pcMag_parser = PcMagParser()
    mission = Mission(sql_db, "pcMag", pcMag_queries, pcMag_parser, scraper, use_proxy)
    mission.run(delay=0, process_num=4, link_limit=1, override_existing=overriding_existing)


def run_innolution_mission(sql_db, terms, scraper, use_proxy, overriding_existing=False):
    innolution_queires = []
    for term in terms:
        scrap_query = ScrapQuery([term], template="\"{}\"", domain="innolution.com/resources/glossary")
        innolution_queires.append(scrap_query)
    innolution_parser = InnolutionParser()
    mission = Mission(sql_db, "pcMag", innolution_queires, innolution_parser, scraper, use_proxy)
    mission.run(delay=0.2, process_num=4, link_limit=1, override_existing=overriding_existing)


def run_regularParse_mission(sql_db, terms, scraper, use_proxy, overriding_existing=True, topic="software engineering"):
    regular_terms = []
    for term in terms:
        scrap_query = ScrapQuery([term], template="definition of \"{}\" in " + topic)
        regular_terms.append(scrap_query)
    regular_parser = RegularParagraphParser()
    mission = Mission(sql_db, "regular", regular_terms, regular_parser, scraper, use_proxy)
    mission.run(process_num=4, override_existing=overriding_existing)


def run_wikipedia_parse_mission(sql_db, terms, override_existing=True):
    """
    This function use wikipedia api to parse wikipedia document
    :return:
    """
    sqlite_manager = Sqlite3Manger(sql_db)
    sqlite_manager.create_table("wiki")
    wiki_dump = {}
    for term in terms:
        related_page_name = wikipedia.search(term)
        page_infos = []
        for page_name in related_page_name:
            try:
                page_info = {}
                page_obj = wikipedia.page(page_name)
                categories = page_obj.categories
                if not __check_wiki_categories(categories):
                    continue
                page_info["summary"] = page_obj.summary
                page_info["categories"] = list(set(categories))
                page_infos.append(page_info)
            except wikipedia.exceptions.DisambiguationError as e:
                first_option = e.options[0]
                print("{} has ambiguity, try first option {}".format(page_name, first_option))
                try:
                    page_info = {}
                    page_obj = wikipedia.page(first_option)
                    categories = page_obj.categories
                    if not __check_wiki_categories(categories):
                        continue
                    page_info["summary"] = page_obj.summary
                    page_info["categories"] = list(set(categories))
                    page_infos.append(page_info)
                except Exception as e2:
                    print("First option failed due to {}".format(e))
            except Exception as other_e:
                print("Exception {}".format(other_e))

        wiki_dump[term] = page_infos
    for term in wiki_dump:
        if override_existing:
            sqlite_manager.add_or_update_row("wiki", term, json.dumps(wiki_dump[term]))
        else:
            sqlite_manager.add_if_not_exist("wiki", term, json.dumps(wiki_dump[term]))
    sqlite_manager.conn.commit()


def __check_wiki_categories(categories):
    for category in categories:
        if __valid_category(category):
            return True
    return False


def __valid_category(category):
    white_list = ["software development", "software engineering", "requirement engineering", "agile development",
                  "computing", "programming"]
    for white_list_item in white_list:
        if white_list_item in category:
            return True
    return False


def run_add_definition_from_file(sql_db, definition_dict):
    logger = logging.getLogger("__name__")
    logger.info("Start Import definitions ...")
    sqlite_manager = Sqlite3Manger(sql_db)
    sqlite_manager.create_table("pcMag")
    for term in definition_dict:
        def_doc = definition_dict[term]
        db_dump = [{"term": term, "definition": def_doc}]
        sqlite_manager.add_or_update_row("pcMag", term, json.dumps(db_dump))
    sqlite_manager.conn.commit()
    logger.info("Finished importing definition")


if __name__ == "__main__":
    proxies = os.path.join(DATA_DIR, "proxy_list.txt")
    sql_db = os.path.join(DATA_DIR, "term_definitions.db")
    vocab_path = os.path.join(DATA_DIR, "expansion_on_fly.txt")
    scraper = GoogleScraperWraper(proxies)
    mode = "add_words"
    if mode == "add_definition":
        # Add extra vocabulary and definition into the database.
        glossory_data_dir = os.path.join(PROJECT_ROOT, "..", "GlossaryProcess", "data")
        for dir_name in os.listdir(glossory_data_dir):
            dir_path = os.path.join(glossory_data_dir, dir_name)
            topic = dir_name.replace("_", " ")
            if os.path.isdir(dir_path):
                for file_name in os.listdir(dir_path):
                    file_path = os.path.join(dir_path, file_name)
                    definition_dict = load_definitoin(file_path)
                    terms = list(definition_dict.keys())
                    run_add_definition_from_file(sql_db, definition_dict)
                    run_stackoverflow_mission(sql_db, terms, scraper, use_proxy=True)
                    run_quora_mission(sql_db, terms, scraper, use_proxy=True, topic=topic)
                    run_regularParse_mission(sql_db, terms, scraper, use_proxy=True, topic=topic)
                    run_wikipedia_parse_mission(sql_db, terms, scraper)
    else:
        if mode == "add_words":
            terms = load_terms(vocab_path)
        elif mode == "dry_run":
            sql_db = "Test.db"
            terms = ["Objective-C", "Scala", "Swift", "Shell", "TypeScript", "go", "C#", "CSS"]
        # Build the database from scratch by give a list of vocabulary
        #run_wikipedia_parse_mission(sql_db, terms, scraper)
        run_pcMag_mission(sql_db, terms, scraper, use_proxy=True)
        run_stackoverflow_mission(sql_db, terms, scraper, use_proxy=True)
        run_quora_mission(sql_db, terms, scraper, use_proxy=True)
        run_innolution_mission(sql_db, terms, scraper, use_proxy=True)
        # run_regularParse_mission(sql_db, terms, scraper, use_proxy=True)
    print("Finished...")
