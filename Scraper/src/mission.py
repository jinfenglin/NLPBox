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

    def __init__(self, sqlite_file, mission_name, scrap_queries, html_parser, scraper):
        self.mission_name = mission_name
        self.sqlite_manager = Sqlite3Manger(sqlite_file)
        self.scrap_queries = scrap_queries
        self.html_parser = html_parser
        self.scraper = scraper
        self.query_scrapQuery = self.__build_queryStr_to_scrapQuery(self.scrap_queries)

    def __build_queryStr_to_scrapQuery(self, scrap_queries):
        """
        Build a dictionary which map query string to scraper query
        :return:
        """
        map_dict = {}
        for scrap_query in scrap_queries:
            map_dict[scrap_query.query] = scrap_query.to_db_primary_key_format()
        return map_dict

    def run(self):
        self.sqlite_manager.create_table(self.mission_name)  # mission name as table name
        query_strs = [x.query for x in self.scrap_queries]
        query_link_dict = self.scraper.scrap_links(query_strs)
        for query in query_link_dict:
            links = query_link_dict[query]
            for link in links:
                link_url = link.link
                html_page = self.scraper.get_html_for_a_link(link_url)
                json = self.html_parser.parse(html_page)
                # Transfer the query string into the db primary key format. Different  mission may ran different types
                # of queries for same terms, when we retrieve information for a term, we want to get them from all the
                # missions
                query_db_format = self.query_scrapQuery[query]
                # write result to database, the table name is mission_name, and in the table there are 2 column ['query','content']
                self.sqlite_manager.add_or_update_row(self.mission_name, query_db_format, json)

        self.sqlite_manager.conn.close()


if __name__ == "__main__":
    sql_db = os.path.join(DATA_DIR, "example.db")
    sp1 = ScrapQuery(["taskboard", "agile"], template="what is {} in {}", domain="stackoverflow.com")
    scraper = GoogleScraperWraper()
    stk_parser = StackOverflowParser()
    mission = Mission(sql_db, "test_mission", [sp1], stk_parser, scraper)
    mission.run()

    sqlM = Sqlite3Manger(sql_db)
    res = sqlM.get_content_for_query(sp1.to_db_primary_key_format())
    for mis in res:
        json_str = res[mis]
        recovered_dict = json.loads(json_str)
        print(recovered_dict)
