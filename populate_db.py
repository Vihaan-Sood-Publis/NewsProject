from app import db,news,classification,summaries


def addto_classification():
    class1 = classification(Class = "Sports")
    class2 = classification(Class = "Politics")
    class3 = classification(Class = "General")
    db.session.add_all([class1,class2,class3])
    db.session.commit()


def addto_news(article,heading,link):
    news_object = news(Article = article,Heading = heading,Link = link)
    db.session.add(news_object)
    db.session.commit()

def addto_summaries(summary,newsID):
    summary_object = summaries(Summary = summary, NewsID = newsID)
    db.session.add(summary_object)
    db.session.commit()