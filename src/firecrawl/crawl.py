from firecrawl import FirecrawlApp
from dotenv import load_dotenv
import os

load_dotenv()

app = FirecrawlApp(api_key=os.getenv('FIRECRAWL_API_KEY'))

def crawl(url: str):
    crawl_status = app.async_crawl_url(
      url, 
      params={
        'limit': 100, 
        'scrapeOptions': {'formats': ['markdown']}
      }
    )
    print(crawl_status)
    return crawl_status

crawl('https://docs.useparagon.com/')
crawl('https://www.socalgas.com/')
crawl('https://www.ycombinator.com/companies')
