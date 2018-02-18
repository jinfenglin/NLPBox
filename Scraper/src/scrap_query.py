class ScrapQuery:
    def __init__(self, query_terms, template, domain=""):
        """
        Produce a query string by give a list of terms and a template. Eg. Given terms [taskboard, agile] and template
        "what is <> in <>", generate a query what is taskboard in agile.
        :param query_terms:
        :param template:
        :param domain:
        """
        self.terms = query_terms
        self.query = template.format(*query_terms)
        if domain != "":
            self.query += " site:{}".format(domain)

    def to_db_primary_key_format(self):
        return "|".join(self.terms)
