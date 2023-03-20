#Getting Data
# egrep.py
import sys, re
# sys.argv is the list of command-line arguments
# sys.argv[0] is the name of the program itself
# sys.argv[1] will be the regex specified at the command line

regex = sys.argv[1]
# for every line passed into the script

for line in sys.stdin:
# if it matches the regex, write it to stdout

if re.search(regex, line):
sys.stdout.write(line)


# line_count.py
import sys
    count = 0
for line in sys.stdin:
    count += 1
# print goes to sys.stdout
print count

type SomeFile.txt | python egrep.py "[0-9]" | python line_count.py

cat SomeFile.txt | python egrep.py "[0-9]" | python line_count.py


# most_common_words.py
import sys
from collections import Counter
# pass in number of words as first argument
    try:
        num_words = int(sys.argv[1])
        except:
            print "usage: most_common_words.py num_words"
            sys.exit(1)    # non-zero exit code indicates error





counter = Counter(word.lower()               # lowercase words
        for line in sys.stdin                #
            for word in line.strip().split() # split on spaces
                if word)                     # skip empty 'words'

for word, count in counter.most_common(num_words):
    sys.stdout.write(str(count))
    sys.stdout.write("\t")
    sys.stdout.write(word)
    sys.stdout.write("\n")




C:\DataScience>type the_bible.txt | python most_common_words.py 10
64193 the
51380 and
34753 of
13643 to
12799 that
12560 in
10263 he
9840 shall
8987 unto
8836 for







with open(filename,'r') as f:
data = function_that_gets_data_from(f)
# at this point f has already been closed, so don't try to use it
process(data)


starts_with_hash = 0
with open('input.txt','r') as f:
for line in file: # look at each line in the file
if re.match("^#",line): # use a regex to see if it starts with '#'
starts_with_hash += 1 # if it does, add 1 to the count


def get_domain(email_address):
"""split on '@' and return the last piece"""
    return email_address.lower().split("@")[-1]


with open('email_addresses.txt', 'r') as f:
    domain_counts = Counter(get_domain(line.strip())
    for line in f
    if "@" in line)


import csv
with open('tab_delimited_stock_prices.txt', 'rb') as f:
    reader = csv.reader(f, delimiter='\t')
    for row in reader:
        date = row[0]
        symbol = row[1]
        closing_price = float(row[2])
        process(date, symbol, closing_price)


with open('colon_delimited_stock_prices.txt', 'rb') as f:
reader = csv.DictReader(f, delimiter=':')
for row in reader:
    date = row["date"]
    symbol = row["symbol"]
    closing_price = float(row["closing_price"])
    process(date, symbol, closing_price)

today_prices = { 'AAPL' : 90.91, 'MSFT' : 41.68, 'FB' : 64.5 }
with open('comma_delimited_stock_prices.txt','wb') as f:
    writer = csv.writer(f, delimiter=',')
    for stock, price in today_prices.items():
        writer.writerow([stock, price])


results = [["test1", "success", "Monday"],
["test2", "success, kind of", "Tuesday"],
["test3", "failure, kind of", "Wednesday"],
["test4", "failure, utter", "Thursday"]]
# don't do this!

with open('bad_csv.txt', 'wb') as f:
    for row in results:
        f.write(",".join(map(str, row))) # might have too many commas in it!
        f.write("\n") # row might have newlines as well!


test1,success,Monday
test2,success, kind of,Tuesday
test3,failure, kind of,Wednesday
test4,failure, utter,Thursday



#Scraping the Web



<html>
<head>
<title>A web page</title>
</head>
<body>
<p id="author">Joel Grus</p>
<p id="subject">Data Science</p>
</body>
</html>


from bs4 import BeautifulSoup
import requests
html = requests.get("http://www.example.com").text
soup = BeautifulSoup(html, 'html5lib')


first_paragraph = soup.find('p') # or just soup.p


first_paragraph_text = soup.p.text
first_paragraph_words = soup.p.text.split()

first_paragraph_id = soup.p['id'] # raises KeyError if no 'id'
first_paragraph_id2 = soup.p.get('id') # returns None if no 'id'


all_paragraphs = soup.find_all('p') # or just soup('p')
paragraphs_with_ids = [p for p in soup('p') if p.get('id')]


important_paragraphs = soup('p', {'class' : 'important'})
important_paragraphs2 = soup('p', 'important')
important_paragraphs3 = [p for p in soup('p')
if 'important' in p.get('class', [])]



# warning, will return the same span multiple times
# if it sits inside multiple divs
# be more clever if that's the case


spans_inside_divs = [span
for div in soup('div') # for each <div> on the page
for span in div('span')] # find each <span> inside it



# you don't have to split the url like this unless it needs to fit in a book
url = "http://shop.oreilly.com/category/browse-subjects/" + \
"data.do?sortby=publicationDate&page=1"
soup = BeautifulSoup(requests.get(url).text, 'html5lib')



<td class="thumbtext">
<div class="thumbcontainer">
<div class="thumbdiv">
<a href="/product/9781118903407.do">
<img src="..."/>
</a>
</div>
</div>
<div class="widthchange">
<div class="thumbheader">
<a href="/product/9781118903407.do">Getting a Big Data Job For Dummies</a>
</div>
<div class="AuthorName">By Jason Williamson</div>
<span class="directorydate"> December 2014 </span>
<div style="clear:both;">
<div id="146350">
<span class="pricelabel">
Ebook:
<span class="price">&nbsp;$29.99</span>
</span>
</div>
</div>
</div>
</td>



tds = soup('td', 'thumbtext')
print len(tds)
# 30


def is_video(td):
"""it's a video if it has exactly one pricelabel, and if
the stripped text inside that pricelabel starts with 'Video'"""
    pricelabels = td('span', 'pricelabel')
    return (len(pricelabels) == 1 and 
    pricelabels[0].text.strip().startswith("Video")) 


print len([td for td in tds if not is_video(td)])
# 21 for me, might be different for you

title = td.find("div", "thumbheader").a.text

author_name = td.find('div', 'AuthorName').text
authors = [x.strip() for x in re.sub("^By ", "", author_name).split(",")]

isbn_link = td.find("div", "thumbheader").a.get("href")
# re.match captures the part of the regex in parentheses

isbn = re.match("/product/(.*)\.do", isbn_link).group(1)


date = td.find("span", "directorydate").text.strip()


def book_info(td):
    """given a BeautifulSoup <td> Tag representing a book,
    extract the book's details and return a dict"""
    title = td.find("div", "thumbheader").a.text
    by_author = td.find('div', 'AuthorName').text
    authors = [x.strip() for x in re.sub("^By ", "", by_author).split(",")]
    isbn_link = td.find("div", "thumbheader").a.get("href")
    isbn = re.match("/product/(.*)\.do", isbn_link).groups()[0]
    date = td.find("span", "directorydate").text.strip()
    return {
        "title" : title,
        "authors" : authors,
        "isbn" : isbn,
        "date" : date
        }


from bs4 import BeautifulSoup
import requests


from time import sleep
base_url = "http://shop.oreilly.com/category/browse-subjects/" + \
"data.do?sortby=publicationDate&page="
books = []
NUM_PAGES = 31 # at the time of writing, probably more by now
for page_num in range(1, NUM_PAGES + 1):
print "souping page", page_num, ",", len(books), " found so far"
url = base_url + str(page_num)
soup = BeautifulSoup(requests.get(url).text, 'html5lib')
for td in soup('td', 'thumbtext'):
if not is_video(td):
books.append(book_info(td))
# now be a good citizen and respect the robots.txt!
sleep(30)


"""book["date"] looks like 'November 2014' so we need to
split on the space and then take the second piece"""

def get_year(book):
"""book["date"] looks like 'November 2014' so we need to
split on the space and then take the second piece"""
    return int(book["date"].split()[1])
# 2014 is the last complete year of data (when I ran this)
year_counts = Counter(get_year(book) for book in books
if get_year(book) <= 2014)


import matplotlib.pyplot as plt
years = sorted(year_counts)
book_counts = [year_counts[year] for year in years]
plt.plot(years, book_counts)
plt.ylabel("# of data books")
plt.title("Data is Big!")
plt.show()



{ "title" : "Data Science Book",
"author" : "Joel Grus",
"publicationYear" : 2014,
"topics" : [ "data", "science", "data science"] }


import json
serialized = """{ "title" : "Data Science Book",
"author" : "Joel Grus",
"publicationYear" : 2014,
"topics" : [ "data", "science", "data science"] }"""
# parse the JSON to create a Python dict
deserialized = json.loads(serialized)
if "data science" in deserialized["topics"]:
    print deserialized



<Book>
<Title>Data Science Book</Title>
<Author>Joel Grus</Author>
<PublicationYear>2014</PublicationYear>
<Topics>
<Topic>data</Topic>
<Topic>science</Topic>
<Topic>data science</Topic>
</Topics>
</Book>

import requests, json
endpoint = "https://api.github.com/users/joelgrus/repos"
repos = json.loads(requests.get(endpoint).text)


u'created_at': u'2013-07-05T02:02:28Z'


pip install python-dateutil



from dateutil.parser import parse
dates = [parse(repo["created_at"]) for repo in repos]
month_counts = Counter(date.month for date in dates)
weekday_counts = Counter(date.weekday() for date in dates)



last_5_repositories = sorted(repos,
key=lambda r: r["created_at"],
reverse=True)[:5]
last_5_languages = [repo["language"]
for repo in last_5_repositories]



from twython import Twython
twitter = Twython(CONSUMER_KEY, CONSUMER_SECRET)
# search for tweets containing the phrase "data science"
for status in twitter.search(q='"data science"')["statuses"]:
user = status["user"]["screen_name"].encode('utf-8')
text = status["text"].encode('utf-8')
print user, ":", text
print




from twython import TwythonStreamer
# appending data to a global variable is pretty poor form
# but it makes the example much simpler
tweets = []
class MyStreamer(TwythonStreamer):
"""our own subclass of TwythonStreamer that specifies
how to interact with the stream"""
def on_success(self, data):
"""what do we do when twitter sends us data?
here data will be a Python dict representing a tweet"""
# only want to collect English-language tweets
if data['lang'] == 'en':
tweets.append(data)
print "received tweet #", len(tweets)
# stop when we've collected enough
if len(tweets) >= 1000:
self.disconnect()
def on_error(self, status_code, data):
print status_code, data
self.disconnect()



stream = MyStreamer(CONSUMER_KEY, CONSUMER_SECRET,
ACCESS_TOKEN, ACCESS_TOKEN_SECRET)
# starts consuming public statuses that contain the keyword 'data'
stream.statuses.filter(track='data')
# if instead we wanted to start consuming a sample of *all* public statuses
# stream.statuses.sample()


top_hashtags = Counter(hashtag['text'].lower()
for tweet in tweets
for hashtag in tweet["entities"]["hashtags"])
print top_hashtags.most_common(5)


