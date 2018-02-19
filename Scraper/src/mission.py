from sql_db_manager import *
import json
from web_page_parser import StackOverflowParser
from common import *
from scraper import GoogleScraperWraper
from scrap_query import ScrapQuery
import threading
import logging


class Mission:
    """
    Specify a mission for scrapping. This class should specify:
    1. A list of queries which should be applied for scrapping
    2. How the retrieved html pages should be parsed
    3. How the parsed results are stored. Eg. Specifying the table names for each kind of information extracted.
    """

    def __init__(self, sqlite_file, mission_name, scrap_queries, html_parser, scraper, use_proxy):
        self.mission_name = mission_name
        self.sqlite_manager = Sqlite3Manger(sqlite_file)
        self.scrap_queries = scrap_queries
        self.html_parser = html_parser
        self.scraper = scraper
        self.query_scrapQuery = self.__build_queryStr_to_scrapQuery(self.scrap_queries)
        self.db_dumps = {}
        self.use_proxy = use_proxy
        self.logger = logging.getLogger(__name__)

    def __build_queryStr_to_scrapQuery(self, scrap_queries):
        """
        Build a dictionary which map query string to scraper query
        :return:
        """
        map_dict = {}
        for scrap_query in scrap_queries:
            map_dict[scrap_query.query] = scrap_query.to_db_primary_key_format()
        return map_dict

    def __fetch_content_worker(self, thread_id, query_link_dict, delay, timeout, link_limit):
        db_dump = {}
        proced_num = 0
        total_num = len(query_link_dict)
        ten_percent_num = max(1, int(total_num * 0.1))
        for query in query_link_dict:
            link_count = 0
            page_info_jsons = []
            links = query_link_dict[query]
            if proced_num % ten_percent_num == 0:  # Report the progress every 10% of total work
                self.logger.info("Thread-{} Progress:{}/{}".format(thread_id, proced_num, total_num))
            for link in links:
                link_count += 1
                if link_count > link_limit:
                    break
                link_url = link.link
                html_page = self.scraper.get_html_for_a_link(link_url, delay=delay, timeout=timeout,
                                                             use_proxy=self.use_proxy)
                page_info_jsons.append(self.html_parser.parse(html_page, query))
            # Transfer the query string into the db primary key format. Different  mission may ran different types
            # of queries for same terms, when we retrieve information for a term, we want to get them from all the
            # missions
            query_db_format = self.query_scrapQuery[query]
            db_dump[query_db_format] = json.dumps(page_info_jsons)
            proced_num += 1
        self.db_dumps[thread_id] = db_dump
        self.logger.info("Thread {} have finished work, {} entries are processed.".format(thread_id, len(db_dump)))

    def run(self, delay=0.1, timeout=10, thread_num=1, link_limit=10):
        self.sqlite_manager.create_table(self.mission_name)  # mission name as table name
        query_strs = [x.query for x in self.scrap_queries]
        query_link_dict = self.scraper.scrap_links(query_strs)
        sub_query_link_dicts = split_dict(query_link_dict, thread_num)
        all_threads = []
        for i in range(thread_num):
            sub_query_link_dict = sub_query_link_dicts[i]
            t = threading.Thread(target=self.__fetch_content_worker,
                                 args=(i, sub_query_link_dict, delay, timeout, link_limit))
            all_threads.append(t)
            t.start()

        for t in all_threads:
            t.join()

        for thread_id in self.db_dumps:
            db_dump = self.db_dumps[thread_id]
            for term in db_dump:
                self.sqlite_manager.add_or_update_row(self.mission_name, term, db_dump[term])
        self.sqlite_manager.conn.close()


if __name__ == "__main__":
    sql_db = os.path.join(DATA_DIR, "example.db")
    sp1 = ScrapQuery(["taskboard", "agile"], template="what is {} in {}", domain="stackoverflow.com")
    sp2 = ScrapQuery(["complexity", "computer science"], template="what is {} in {}", domain="stackoverflow.com")
    sp3 = ScrapQuery(["validation", "computer science"], template="what is {} in {}", domain="stackoverflow.com")
    sp4 = ScrapQuery(["assembly language", "computer science"], template="what is {} in {}", domain="stackoverflow.com")

    proxies = os.path.join(DATA_DIR, "proxy_list.txt")
    scraper = GoogleScraperWraper(proxies)
    stk_parser = StackOverflowParser()
    mission = Mission(sql_db, "test_mission", [sp1, sp2, sp3, sp4], stk_parser, scraper, use_proxy=True)
    mission.run(thread_num=4)

    sqlM = Sqlite3Manger(sql_db)
    res = sqlM.get_content_for_query(sp1.to_db_primary_key_format())
    for mis in res:
        json_str = res[mis]
        recovered_dict = json.loads(json_str)
        print(recovered_dict)
