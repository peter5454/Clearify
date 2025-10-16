from newspaper import Article
import nltk

#Ensuring the punkt is available
nltk.download('punkt', quiet=True)

#Integrating Newspaper3k 

def scrape_article(url: str) -> dict:
    """
    Download, parse, and summarize an article using Newspaper3k.
    Returns a dictionary with title, authors, publish date, text, summary, and keywords.
    """
    article = Article(url)
    article.download()
    article.parse()
    article.nlp()

def fetch_data():
    text = scrape_article()
    return [{"text": text}]

    return {
        "title": article.title,
        "authors": article.authors,
        "publish_date": article.publish_date,
        "text": article.text,
        "summary": article.summary,
        "keywords": article.keywords,
    }
