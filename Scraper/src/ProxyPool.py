import platform
import re
import threading
from time import sleep

from urllib3 import make_headers, ProxyManager

from common import DATA_DIR
import subprocess, os


class ProxyPool:
    def __init__(self, proxy_list_file):
        self.credit_record = {}
        self.waiting_round = {}
        self.proxy_list = self.__read_proxy_list(proxy_list_file)
        self.available_proxy = set()

    def __read_proxy_list(self, file_path):
        with open(file_path) as fin:
            for line in fin:
                proxy_url = line.strip("\n\t\r ")
                self.credit_record[proxy_url] = 0
                self.waiting_round[proxy_url] = 0
        return self.credit_record.keys()

    def is_alive_proxy(self, proxy):
        host = self.get_ip(proxy)

        if platform.system() == "Windows":
            command = "ping {} -n 1".format(host)
        else:
            command = "ping {} -c 1".format(host)
        proc = subprocess.Popen(command.split(), stdout=subprocess.PIPE, shell=True)
        proc.wait()
        isUpBool = False
        if proc.returncode == 0:
            if self.can_get_response("http://www.example.org", timeout=10, proxy=proxy):
                isUpBool = True
        return isUpBool

    def can_get_response(self, link, timeout, proxy):
        try:
            header = "Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:1.9.0.7) Gecko/2009021910 Firefox/3.0.7"
            headers = make_headers(user_agent=header)
            http = ProxyManager(proxy, headers=headers)
            response = http.request("GET", link, timeout=timeout)
            status_code = response.status
            if str(status_code).startswith("2"):
                return True
            else:
                return False
        except Exception as e:
            return False

    def get_ip(self, url):
        ip_pattern = r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b'
        return re.search(ip_pattern, url).group(0)

    def update_proxy_list(self, interval=10):
        while True:
            for proxy in self.proxy_list:
                penalty_degree = self.credit_record[proxy]
                remain_waiting = self.waiting_round[proxy]

                if remain_waiting > 0:
                    self.waiting_round[proxy] -= 1
                    continue

                is_live_flag = False
                try:
                    is_live_flag = self.is_alive_proxy(proxy)
                except Exception as e:
                    print(e)

                if is_live_flag:
                    if penalty_degree > 0:
                        self.credit_record[proxy] -= 1
                    self.available_proxy.add(proxy)
                else:
                    self.credit_record[proxy] += 1
                    self.waiting_round[proxy] = min(100, remain_waiting + self.credit_record[proxy])
                    if proxy in self.available_proxy:
                        self.available_proxy.remove(proxy)

            sleep(interval)

    def run(self):
        t = threading.Thread(target=self.update_proxy_list, )
        t.start()


if __name__ == "__main__":
    proxy_path = os.path.join(DATA_DIR, "proxy_list.txt")
    proxy_pool = ProxyPool(proxy_path)
    proxy_pool.run()
