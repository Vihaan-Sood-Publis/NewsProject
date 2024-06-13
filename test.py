import feedparser




my_site ="https://news.google.com/rss?hl=en-GB&gl=GB&ceid=GB:en"
feed = feedparser.parse(my_site)

print(feed.entries)


for entry in feed.entries:
    print(entry.title)
    print(entry.link)
    print(entry.summary)
    print()






