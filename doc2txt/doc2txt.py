#!usr/bin/env python
from pdf_adapter import PDFAdapter
import os

'''
Convert multiple format document into plain txt
'''


class Doc2Txt:
    def __init__(self):
        self.adapterMapper = dict()
        self.adapterMapper['.pdf'] = PDFAdapter()
        # self.adapterMapper['xml'] =

    def get_file_extension(self, file_path):
        """
        Get the file extension
        :param file_path: Path to the file
        :return: The extension of the file, if it is not a file or file without extension, and empty string will be returned
        """
        postfix = ""
        if os.path.isfile(file_path):
            postfix = os.path.splitext(file_path)[1]
        return postfix

    def parse_single_file(self, file_path, out_path=None):
        """
        Parse a single file into txt format

        :param file_path: Path to the file
        :param out_path: The path to write the converted file. If it is none the converted text will be returned
        :return: Return converted text if it is not written to file
        """
        extension = self.get_file_extension(file_path)
        try:
            adapter = self.adapterMapper[extension]
            text = adapter.parse(file_path)
            if out_path is None:
                return text
            else:
                self.write_to_file(text, out_path)
        except KeyError:
            print "Unable to process object {0} with extension {1}".format(file_path, extension)
            raise KeyError

    def parse_dir(self, dir_path, out_dir=None):
        """
        Parse all files in a directory

        :param dir_path: The path to the directory
        :param out_dir: The path to the directory that the converted files will be written to. If it is none, a list of converted text will be returned
        :return: A list of converted text if they are not written to file
        """
        files_in_dir = os.listdir(dir_path)
        res_list = list()
        for fileName in files_in_dir:
            print "\nProcessing file: {}".format(fileName)
            filePath = os.path.join(dir_path, fileName)
            try:
                target_path = os.path.join(out_dir, fileName + '.html')
                #If file already exist, don't parse it again
                if not os.path.exists(target_path):
                    txt = self.parse_single_file(filePath)
                else:
                    print "{} already parsed, skip this file".format(fileName)
                    continue

                if out_dir is None:
                    res_list.append(txt)
                else:
                    self.write_to_file(txt, target_path)
            except KeyError:
                continue

        return res_list

    def write_to_file(self, txt, out_path):
        """
        Write some text into a file
        :param txt: The content to be written
        :param out_path: The path of the output file
        :return:
        """
        with open(out_path, 'w') as output:
            output.write(txt)


if __name__ == '__main__':
    print Doc2Txt().parse_dir('/Users/jinfenglin/Downloads/Glossary/raw', '/Users/jinfenglin/Downloads/Glossary/txt')
