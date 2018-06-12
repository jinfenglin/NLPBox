import sys

sys.path.append("../../Cleaner")
import sqlite3
import cleaner
import logging, threading


class Sqlite3Manger:
    def __init__(self, sqlite_file, commit_period=50):
        self.conn = sqlite3.connect(sqlite_file)
        self.logger = logging.getLogger(__name__)
        self.lock = threading.Lock()
        self.execute_count = 0
        self.commit_period = commit_period

    def __del__(self):
        try:
            self.conn.commit()  # commit result before quiting
            self.conn.close()
        except Exception as e:
            self.logger.exception("Close Db Exception:{}".format(e))

    def __execute(self, sql):
        """
        Synchronized function
        :param sql:
        :return:
        """
        try:
            with self.lock:
                c = self.conn.cursor()
                res = c.execute(sql)
                self.execute_count += 1
                if self.execute_count > self.commit_period:
                    self.logger.debug("Commit in database")
                    self.execute_count = 0
                    self.conn.commit()
                return res
        except Exception as e:
            self.logger.info("Error with executing sql:", sql)
            self.logger.exception(e)

    def create_table(self, table_name):
        table_name = cleaner.esapce_sql_variable_quote(table_name)
        sql = "CREATE TABLE  IF NOT EXISTS {} (query text PRIMARY KEY, content text);".format(table_name)
        self.__execute(sql)

    def drop_table(self, table_name):
        table_name = cleaner.esapce_sql_variable_quote(table_name)
        sql = "DROP TABLE IF EXISTS {};".format(table_name)
        self.__execute(sql)

    def add_or_update_row(self, table_name, query, content):
        table_name = cleaner.esapce_sql_variable_quote(table_name)
        query = cleaner.esapce_sql_variable_quote(query)
        content = cleaner.esapce_sql_variable_quote(content)
        sql = "INSERT OR REPLACE INTO {} VALUES (\'{}\',\'{}\')".format(table_name, query, content)
        self.__execute(sql)

    def add_if_not_exist(self, table_name, query, content):
        table_name = cleaner.esapce_sql_variable_quote(table_name)
        query = cleaner.esapce_sql_variable_quote(query)
        content = cleaner.esapce_sql_variable_quote(content)
        sql = "INSERT OR IGNORE INTO {} VALUES (\'{}\',\'{}\')".format(table_name, query, content)
        self.__execute(sql)

    def get_rows_for_table(self, table_name, colums):
        table_name = cleaner.esapce_sql_variable_quote(table_name)
        columns = ",".join(colums)
        sql = "SELECT {} FROM {}".format(columns,table_name)
        return self.__execute(sql)

    def get_content_for_query(self, query):
        """
        Find parsed information from all tables for a single query
        :param query:
        :return: A dictionary contains doc parsed from different source. The content is a json string
        """
        type_content = {}
        sql_get_all_tables = "SELECT name FROM sqlite_master WHERE type = 'table'"
        table_names = self.__execute(sql_get_all_tables)
        for name_row in table_names:
            table_name = cleaner.esapce_sql_variable_quote(name_row[0])
            query = cleaner.esapce_sql_variable_quote(query)
            type_content[table_name] = ""
            sql = "SELECT content FROM {} WHERE query = \'{}\'".format(table_name, query)
            for content_row in self.__execute(sql):
                content = content_row[0]
                if not content:
                    content = ""
                type_content[table_name] = content
        return type_content


if __name__ == "__main__":
    sqlM = Sqlite3Manger("google_scraper.db")
    rows = sqlM.get_rows_for_table("link", ['link', 'serp_id', 'title'])
    for row in rows:
        print(row)