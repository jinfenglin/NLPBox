import os

from mission import Mission
from scrap_query import ScrapQuery
from scraper import GoogleScraperWraper, DATA_DIR
from web_page_parser import RegularParagraphParser

if __name__ == "__main__":
    proxies = os.path.join(DATA_DIR, "proxy_list.txt")
    sql_db = os.path.join(DATA_DIR, "uav_doc.db")
    scraper = GoogleScraperWraper(proxies)
    uav_querys_strs = ['Small Unmanned Aerial Systems', 'Drone Traffic Management', 'drone coordination',
                       'intelligent drone systems', 'drone collision avoidance', 'FAA small uas',
                       'UAV ground control station',
                       'mavlink', 'pixhawk', 'drone swarm', 'drone simulation', 'runtime monitoring', 'middleware',
                       'messaging system',
                       'publish subscribe', 'software architecture', 'control software', 'dronekit python']
    uav_querys = []
    for term in uav_querys_strs:
        scrap_query = ScrapQuery([term], template="{}")
        uav_querys.append(scrap_query)
    parser = RegularParagraphParser()
    mission = Mission(sql_db, "uav_article_debug", uav_querys, parser, scraper, use_proxy=True)
    mission.run(delay=1, thread_num=15, page_num=1500, collect_link=False)
