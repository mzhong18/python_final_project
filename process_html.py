"""
This code takes a url, extracts text from it, removes html tags and prints the text string to a file
This should be used to prepare text from a url to be summarized

"""

from bs4 import BeautifulSoup
from bs4.element import Comment
import urllib.request
import textwrap
import re


class Process_html:

    def __init__(self, url, filename):
        self.html = urllib.request.urlopen(url).read()
        self.filename = filename

    #this function returns true if text is part of a tag
    def tag(self, thing):
        if thing.parent.name in ['[document]', 'style', 'script', 'head', 'title', 'meta']:
            return False
        if isinstance(thing, Comment):
            return False
        return True

    #this function uses beautifulSoup parser to identify and remove html tags
    def text_from_html(self):
        html = BeautifulSoup(self.html, 'html.parser')
        texts = html.findAll(text=True)
        show = filter(self.tag, texts)
        return u" ".join(t.strip() for t in show)

    #print extracted text from html to a new file
    def create_file(self):
        text = textwrap.fill(str(self.text_from_html()), 100)
        #use re for anything that was not caught in beautiful soup just in case
        fix = re.compile(r'<\/?\w+\s*[^>]*?\/?>')
        answer = re.sub(fix, '', text)
        with open(self.filename, 'w') as f:
            print(answer, file=f)

#main method
if __name__ == '__main__':
    #Please put in an new url and NEW filename as parameters in Summarize_html to keep updating summarizer
    html_file = Process_html('https://www.wired.com/story/netflix-hulu-amazon-oscar-shortlist/', 'html_file.txt')
    html_file.create_file()
    print()
    print('NOTE: This summarizer will not work well on html as it was not trained on a html corpus')
    print('This class shows that our summarizer has the potential to work well on html once it is trained correctly')
    print('Certain symbols such as @ and other characters that have not been detected yet will cause an error')
    print()
    print()
    print('Now, please go to summarize_file.py and input your new file to summarize the new file that you have created')
