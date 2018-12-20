# python_final_project
## An extraction based automatic summarization project for linguistics 131

### Brief Description:
Core function: select the most relevant sentences as the summary of a given text. 
To do this, we make use of TFIDF and sentence length to determine sentence relevance.

### Instructions:
1.  Please run the file named original_IDF.py first - this will create an initial IDF dictionary
    that will be implemented in the summarizer. This file should only be run once.

2.  Next, go to summarize_file.py to summarize your file. We have already put some code to run one of our examples
    'file = Summarize_file("test.txt")'. In order to try our summarizer on other .txt files please change
    file = Summarize_file("test.txt") to  file = Summarize_file(**yourfilename**) and run the file. The IDF dictionary
    and its data will update and grow everytime you summarize a new file in summarize.py. Everytime you run a file, you
    can view the results in results.txt

3.  To summarize an html file, first run it through the process_html.py file 
  


