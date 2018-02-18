import sqlite3
from Cleaner import corpus_cleaner


class Sqlite3Manger:
    def __init__(self, sqlite_file):
        self.conn = sqlite3.connect(sqlite_file)

    def __del__(self):
        self.conn.close()

    def __execute(self, sql):
        try:
            c = self.conn.cursor()
            res = c.execute(sql)
            self.conn.commit()
            return res
        except Exception as e:
            print("Error with executing sql:", sql)
            print(e)

    def create_table(self, table_name):
        table_name = corpus_cleaner.esapce_sql_variable_quote(table_name)
        sql = "CREATE TABLE  IF NOT EXISTS {} (query text PRIMARY KEY, content text);".format(table_name)
        self.__execute(sql)

    def drop_table(self, table_name):
        table_name = corpus_cleaner.esapce_sql_variable_quote(table_name)
        sql = "DROP TABLE IF EXISTS {};".format(table_name)
        self.__execute(sql)

    def add_or_update_row(self, table_name, query, content):
        table_name = corpus_cleaner.esapce_sql_variable_quote(table_name)
        query = corpus_cleaner.esapce_sql_variable_quote(query)
        content = corpus_cleaner.esapce_sql_variable_quote(content)
        sql = "INSERT OR REPLACE INTO {} VALUES (\'{}\',\'{}\')".format(table_name, query, content)
        self.__execute(sql)

    def get_content_for_query(self, query):
        """
        Find parsed information from all tables for a single query
        :param query:
        :return:
        """
        type_content = {}
        sql_get_all_tables = "SELECT name FROM sqlite_master WHERE type = 'table'"
        table_names = self.__execute(sql_get_all_tables)
        for name_row in table_names:
            table_name = corpus_cleaner.esapce_sql_variable_quote(name_row[0])
            query = corpus_cleaner.esapce_sql_variable_quote(query)
            type_content[table_name] = ""
            sql = "SELECT content FROM {} WHERE query = \'{}\'".format(table_name, query)
            for content_row in self.__execute(sql):
                content = content_row[0]
                if not content:
                    content = ""
                type_content[table_name] = content
        return type_content


if __name__ == "__main__":
    sqlM = Sqlite3Manger("Test.db")
    sqlM.create_table("test1")
    sqlM.create_table("test2")
    sqlM.create_table("test3")

    sqlM.add_or_update_row("test1", "q1", "this is not a json")
    sqlM.add_or_update_row("test2", "q1", "this is no a json too")

    print(sqlM.get_content_for_query("q1"))
