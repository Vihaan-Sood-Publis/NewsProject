from transformers import BartTokenizer,BartForConditionalGeneration,pipeline
import feedparser
import morss 
import numpy as np

news_sites =[
    "https://www.independent.co.uk/news/uk/rss",
    "https://feeds.bbci.co.uk/news/uk/rss.xml",
    "https://feeds.skynews.com/feeds/rss/uk.xml"
    ]




#LLMprompt = "Following is a list of news articles in the format <Title> \n <Link>\n<Text>\n(---). Task is to summarise the given articles, give title and associated link, and to classify them. If the same news content appears in multiple articles, you are to take the most important parts from all duplicated articles, and in such a scenario, you may output any of the given links from the duplicated articles in your response. You may only classify an article into 3 categories: Sports, Politics and General. Give your response in the following manner: <Title>\n<Link>\n<Classification>\n<Summary>\n Input:"

LLMprompt = "Summarise each article:\n"

options = morss.Options(csv=True) # arguments

for site in news_sites:

    url, rss = morss.FeedFetch(site, options) # this only grabs the RSS feed
    rss = morss.FeedGather(rss, url, options) # this fills the feed and cleans it up

    output = morss.FeedFormat(rss, options, 'unicode') # formats final feed
    print(type(output))
    feed = feedparser.parse(output)



    for entry in feed.entries:
    

            title = entry.title
            link = entry.link
            content =  str(entry.summary)

            LLMprompt += title+" "+content+"\n"

LLMprompt += "Output:"

# LLMprompt = LLMprompt.replace("\\n","")
LLMprompt = re.sub("<a>","",LLMprompt)
# LLMprompt = re.sub("<\\a>","",LLMprompt)
# LLMprompt = re.sub("<a><\\a>","",LLMprompt)


print (LLMprompt)
# print(len(LLMprompt))


# model = "facebook/bart-large"
# bart_model = BartForConditionalGeneration.from_pretrained(model,max_length = 1024)
# newmodel = pipeline()
# tokenizer = BartTokenizer.from_pretrained(model)

# inputs = tokenizer.encode(LLMprompt, return_tensors="pt", max_length=1024, truncation=True)
# summary_ids = bart_model.generate(inputs, max_length=150, min_length=50, length_penalty=2.0, num_beams=4, early_stopping=True)

# generated_text = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
   
# print("Generated Text:")
# print(generated_text)

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
print(summarizer(LLMprompt, max_length=600, min_length=30, do_sample=True))