#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from GoogleScraper import scrape_with_config, GoogleSearchError
from common import *
import shutil
import urllib
import time
from urllib3 import ProxyManager, make_headers, disable_warnings, exceptions
import random
import logging


class GoogleScraperWraper:
    """
    A Scraper wraper, which can

    1. Scrape links for a list of queries
    2. Retrieve a html page for a link

    Current implementation is single thread due to the sqlite features.
    """

    def __init__(self, proxy_file=""):
        disable_warnings(exceptions.InsecureRequestWarning)
        self.default_scraper_db = "google_scraper.db"
        self.default_cache_dir = ".scrapecache"
        self.logger = logging.getLogger(__name__)
        self.proxies = []  # http://spys.one/en/https-ssl-proxy/ Available proxies
        if proxy_file != "":
            with open(proxy_file) as fin:
                for proxy_ip in fin.readlines():
                    proxy_ip = proxy_ip.strip("\n\t\r ")
                    self.proxies.append(proxy_ip)

    def clear(self, clear_cache=True):
        """
        Delete the sqlite database created by GoolgeScraper
        :return:
        """
        if os.path.isfile(self.default_scraper_db):
            os.remove(self.default_scraper_db)
        if clear_cache and os.path.isdir(self.default_cache_dir):
            shutil.rmtree(self.default_cache_dir)

    def scrap_links(self, query_str_list, search_engine=["bing"], page_num=1, method="http", cache="True"):
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
        try:
            db_session = scrape_with_config(config)
        except GoogleSearchError as e:
            self.logger.exception("Scraper Error:", e)

        res = {}
        for serp in db_session.serps:
            query = serp.query
            res[query] = serp.links
        return res

    def __request(self, link, timeout):
        headers = make_headers()
        request = urllib.request.Request(link, None, headers)
        try:
            with urllib.request.urlopen(request, timeout=timeout) as url:
                html_page = url.read()
                return html_page
        except Exception as e:
            return ""

    def __request_with_proxy(self, link, timeout):
        headers = make_headers()
        proxy_ip = random.sample(self.proxies, 1)[0]
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
            if len(self.proxies) > 0 and use_proxy:
                try:
                    res = self.__request_with_proxy(link, timeout)
                except Exception as proxy_e:
                    res = self.__request(link, timeout)
                    self.logger.exception("Request with proxy exception:", proxy_e)
            else:
                res = self.__request(link, timeout)
        except Exception as e:
            self.logger.exception(e)
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