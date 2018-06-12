#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from GoogleScraper import scrape_with_config, GoogleSearchError

from ProxyPool import ProxyPool
from common import *
import shutil
import urllib
import time
from urllib3 import ProxyManager, make_headers, disable_warnings, exceptions
import random
import logging

from sql_db_manager import Sqlite3Manger


class GoogleScraperWraper:
    """
    A Scraper wraper, which can

    1. Scrape links for a list of queries
    2. Retrieve a html page for a link

    Current implementation is single thread due to the sqlite features.
    """

    def __init__(self, proxy_file=""):
        self.user_agent_header = "Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:1.9.0.7) Gecko/2009021910 Firefox/3.0.7"
        disable_warnings(exceptions.InsecureRequestWarning)
        self.default_scraper_db = "google_scraper.db"
        self.default_cache_dir = ".scrapecache"

        self.logger = logging.getLogger(__name__)
        self.proxy_manger = ProxyPool(proxy_file)
        self.proxy_manger.run()

    def clear(self, clear_cache=True):
        """
        Delete the sqlite database created by GoolgeScraper
        :return:
        """
        if os.path.isfile(self.default_scraper_db):
            os.remove(self.default_scraper_db)
        if clear_cache and os.path.isdir(self.default_cache_dir):
            shutil.rmtree(self.default_cache_dir)

    def scrap_links(self, query_str_list, search_engine=["bing"], page_num=12000, method="http", cache="True",
                    collect_link=True):
        """
        Scraper for a list of queries and get the links as a result. Use search engines to scrap the links.

        :param query_str_list:Queries in string format submitted to search engine
        :param search_engine: See GoogleScraper package for more information
        :param page_num: The number of page to scrap
        :param method: Use http for most case
        :param cache: Use cache
        :return: A dictionary whose key is the query string, and the value is the links
        """
        query_set = set(query_str_list)
        config = {
            'use_own_ip': 'True',
            'keywords': query_set,
            'search_engines': search_engine,
            'num_pages_for_keyword': page_num,
            'scrape_method': method,
            'do_caching': cache
        }
        res = {}
        if collect_link:
            try:
                db_session = scrape_with_config(config)
            except GoogleSearchError as e:
                self.logger.exception("Scraper Error:", e)

            print("{} serps to process...".format(db_session.serps))
            for serp in db_session.serps:
                query = serp.query
                if query not in res:
                    res[query] = set()
                for link in serp.links:
                    res[query].add(link.link)
        else:
            sql_db_manger = Sqlite3Manger("google_scraper.db")
            links = sql_db_manger.get_rows_for_table("link", ['link', 'serp_id', 'title'])
            serps = sql_db_manger.get_rows_for_table('serp', ['id', 'query'])
            serp_query_dict = {}
            for serp in serps:
                serp_id = serp[0]
                query = serp[1]
                serp_query_dict[serp_id] = query
            for link in links:
                link_url = link[0]
                serp_id = link[1]
                query = serp_query_dict[serp_id]
                if query not in res:
                    res[query] = set()
                res[query].add(link_url)
        return res

    def __request(self, link, timeout):
        headers = make_headers(user_agent=self.user_agent_header)
        request = urllib.request.Request(link, None, headers)
        with urllib.request.urlopen(request, timeout=timeout) as url:
            html_page = url.read()
            return html_page

    def __request_with_proxy(self, link, timeout, proxies):
        headers = make_headers(user_agent=self.user_agent_header)
        proxy_ip = random.sample(proxies, 1)[0]
        http = ProxyManager(proxy_ip, headers=headers)
        response = http.request("GET", link, timeout=timeout)
        return response.data

    def get_html_for_a_link(self, link, delay=0.1, timeout=10, use_proxy=False):
        """
        Retrieve the html page for a link.
        :param link:
        :param delay:
        :param timeout:
        :return:
        """
        if delay > 0:
            time.sleep(delay)
        res = ""
        try:
            proxies = self.proxy_manger.available_proxy
            if len(proxies) > 0 and use_proxy:
                try:
                    res = self.__request_with_proxy(link, timeout, proxies)
                except Exception as proxy_e:
                    res = self.__request(link, timeout)
                    self.logger.exception("Request with proxy exception:", proxy_e)
            else:
                res = self.__request(link, timeout)
        except Exception as e:
            self.logger.exception("Exceptions in Scraping link {} :{}".format(link, e))
        if not res:
            res = ""
        return res


if __name__ == "__main__":
    proxies = os.path.join(DATA_DIR, "proxy_list.txt")
    gsw = GoogleScraperWraper(proxies)
    keywords = ["apple", "google inc", "maven"]
    res_dict = gsw.scrap_links(keywords)
    for k in res_dict:
        for link in res_dict[k]:
            print(link.link)
            print(gsw.get_html_for_a_link(link.link, use_proxy=True))
            print("------------------")
