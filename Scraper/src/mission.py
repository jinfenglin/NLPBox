import shutil
from threading import Lock

from sql_db_manager import *
import json
from web_page_parser import StackOverflowParser
from common import *
from scraper import GoogleScraperWraper
from scrap_query import ScrapQuery


class Mission:
    """
    Specify a mission for scrapping. This class should specify:
    1. A list of queries which should be applied for scrapping
    2. How the retrieved html pages should be parsed
    3. How the parsed results are stored. Eg. Specifying the table names for each kind of information extracted.
    """

    def __init__(self, sqlite_file, mission_name, scrap_queries, html_parser, scraper, use_proxy):
        self.mission_name = mission_name

        self.scrap_queries = scrap_queries
        self.html_parser = html_parser
        self.scraper = scraper
        self.query_scrapQuery = self.__build_queryStr_to_scrapQuery(self.scrap_queries)
        self.db_dumps = {}
        self.use_proxy = use_proxy
        self.query_to_dir = {}  # Translate the query into directory that data will write into
        self.file_id_cnt = 0  # IMPORTANT this variable is shared by the processes thus need to use lock
        self.file_id_lock = Lock()
        self.logger = logging.getLogger(__name__)
        pass

    def __get_file_id(self):
        """
        Get the next file's name/id
        :return:
        """
        with self.file_id_lock:
            self.file_id_cnt += 1
            file_name = str(self.file_id_cnt) + ".txt"
            return file_name

    def __build_queryStr_to_scrapQuery(self, scrap_queries):
        """
        Build a dictionary which map query string to scraper query
        :return:
        """
        map_dict = {}
        for scrap_query in scrap_queries:
            map_dict[scrap_query.query] = scrap_query.to_db_primary_key_format()
        return map_dict

    def __fetch_content_worker(self, thread_id, query_link_dict, delay, timeout):
        print("Start thread {}".format(thread_id))
        processed_num = 0
        total_num = 0
        for query in query_link_dict:
            total_num += len(query_link_dict[query])
        ten_percent_num = max(1, int(total_num * 0.1))

        for query in query_link_dict:
            links = query_link_dict[query]
            dir_path = self.query_to_dir[query]
            for link in links:
                if processed_num % ten_percent_num == 0:  # Report the progress every 10% of total work
                    self.logger.info("Thread-{} Progress:{}/{}".format(thread_id, processed_num, total_num))
                file_path = os.path.join(dir_path, self.__get_file_id())
                html_page = self.scraper.get_html_for_a_link(link, delay=delay, timeout=timeout,
                                                             use_proxy=self.use_proxy)
                with open(file_path, 'w', encoding='utf8') as fout:
                    self.logger.debug("Writing file {} from link".format(file_path, link))
                    page_info_dict = self.html_parser.parse(html_page, query)
                    content = page_info_dict['content']
                    for paragraph in content:
                        fout.write(paragraph + "\n")
                processed_num += 1
        self.logger.info("Thread-{} have finished work".format(thread_id))

    def run(self, delay=0.1, timeout=10, thread_num=1, page_num=1, overwrite=True, collect_link=True):
        """

        :param delay: The time interval between 2 requests
        :param timeout: The maximal time for each request
        :param thread_num: The number of threads
        :param link_limit: The maximal links processed for each query
        :param override_existing: Whether override the existing content in database
        :return:
        """
        query_strs = [x.query for x in self.scrap_queries]
        mission_dir_path = os.path.join(DATA_DIR, self.mission_name)
        if os.path.exists(mission_dir_path):
            if overwrite:
                shutil.rmtree(mission_dir_path)
                os.mkdir(mission_dir_path)
        else:
            os.mkdir(mission_dir_path)

        # Prepare the directories that data will write into
        for query in query_strs:
            query_dir_path = os.path.join(mission_dir_path, query)
            if not os.path.exists(query_dir_path):
                os.mkdir(query_dir_path)
            self.query_to_dir[query] = query_dir_path

        # Get links
        query_link_dict = self.scraper.scrap_links(query_strs, page_num=page_num, collect_link=collect_link)
        for query in query_link_dict:
            print("{} links collected for query: {}".format(len(query_link_dict[query]), query))
        sub_query_link_dicts = split_dict(query_link_dict, thread_num)
        print("Links dictionary are splitted into {} parts".format(thread_num))

        # Fetch content from links and write them into directory
        all_processes = []
        for i in range(thread_num):
            sub_query_link_dict = sub_query_link_dicts[i]
            t = threading.Thread(target=self.__fetch_content_worker,
                                 args=(i, sub_query_link_dict, delay, timeout))

            all_processes.append(t)
            t.start()

        for t in all_processes:
            t.join()
        print("Finished all works!")


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
