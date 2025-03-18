from firecrawl import FirecrawlApp
from dotenv import load_dotenv
import os

load_dotenv()

app = FirecrawlApp(api_key=os.getenv('FIRECRAWL_API_KEY'))

def crawl(url: str):
    crawl_status = app.async_crawl_url(
      url, 
      params={
        'limit': 1, 
        'scrapeOptions': {'formats': ['markdown']}
      }
    )
    print(crawl_status)
    return crawl_status

def create_directory_struct():
    if not os.path.exists('./knowledge-base/'):
        os.makedirs('./knowledge-base/')

def create_mds(fc_id: str) -> None:
    crawl_status = app.check_crawl_status(fc_id)
    for md in crawl_status['data']:
        with open(f'./knowledge-base/{md['metadata']['url'].replace('/', '').replace('.', '').replace(':', '')}.md', 'w') as file:
            file.write(md['markdown'])

def crawl_to_md(urls: list) -> None:
    create_directory_struct()
    for url in urls:
        crawl_async = crawl(url)
        # crawl('https://docs.useparagon.com')
        # crawl('https://www.socalgas.com/')
        # crawl('https://www.ycombinator.com/companies')
        create_mds(crawl_async['id'])
