from newspaper import Article
import nltk

#Ensuring the punkt is available
nltk.download('punkt', quiet=True)

#Integrating Newspaper3k 

def scrape_article(url):
    try:
        article = Article(url)
        article.download()
        article.parse()
        return article.text
    except Exception as e:
        print(f"Scraper error: {e}")
        return None

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
