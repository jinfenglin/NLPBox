import logging


def write_csv(res, writer):
    res = [str(x) for x in res]
    content = ",".join(res)
    try:
        writer.write(content + "\n")
    except Exception as e:
        logging.getLogger(__name__).info(content)
