#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from GoogleScraper import scrape_with_config, GoogleSearchError
from common import *
import shutil
import urllib
import time
from lxml import html


class GoogleScraperWraper:
    """
    A Scraper wraper, Current implementation is single thread due to the sqlite features.
    """

    def __init__(self):
        self.default_scraper_db = "google_scraper.db"
        self.default_cache_dir = ".scrapecache"
        user_agent = 'Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:1.9.0.7) Gecko/2009021910 Firefox/3.0.7'
        self.headers = {'User-Agent': user_agent, }

    def clear(self, clear_cache=True):
        """
        Delete the sqlite database created by GoolgeScraper
        :return:
        """
        if os.path.isfile(self.default_scraper_db):
            os.remove(self.default_scraper_db)
        if clear_cache and os.path.isdir(self.default_cache_dir):
            shutil.rmtree(self.default_cache_dir)

    def scrap_links(self, query_list, search_engine=["bing"], page_num=1, method="http", cache="True"):
        """
        Scraper a list of queries and get the links as a result. The query will be the key
        :param query_list:
        :param search_engine:
        :param page_num:
        :param method:
        :param cache:
        :return:
        """
        query_set = set(query_list)
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
            print("Scraper Error:", e)

        res = {}
        for serp in db_session.serps:
            query = serp.query
            res[query] = serp.links
        return res

    def __html_all_text(html_page):
        tree = html.fromstring(html_page)
        text = tree.xpath('//p/text()')
        text = " ".join(text)
        return text

    def get_content_for_a_link(self, link, html_parse_method=__html_all_text, delay=0.1, timeout=60):
        if delay > 0:
            time.sleep(delay)
        res = ""
        try:
            request = urllib.request.Request(link, None, self.headers)
            with urllib.request.urlopen(request, timeout=timeout) as url:
                html_page = url.read()
                res = html_parse_method(html_page)
        except Exception as e:
            print(e)
        return res


if __name__ == "__main__":
    gsw = GoogleScraperWraper()
    keywords = ["this is query1", "this is query1"]
    res_dict = gsw.scrap_links(keywords)
    for k in res_dict:
        for link in res_dict[k]:
            print(link.link)
            print(gsw.get_content_for_a_link(link.link))
            print("------------------")
