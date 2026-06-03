import json
import time
import requests
from xdk import Client
import os
from dotenv import load_dotenv
load_dotenv()
client = Client(bearer_token=os.getenv('BEARER_TOKEN'))
fields_to_include = ["article",
                     "attachments",
                     "author_id",
                     "conversation_id",
                     "created_at",
                     "entities",
                     "geo",
                     "id",
                     "lang",
                     "note_tweet",
                     "text"]
# Search with automatic pagination
count = len([name for name in os.listdir('.') if os.path.isfile(name)])
save_file = 'Markuszower-Part2.json'
all_posts = []
def scrape(id, date):
    try:
        for page in client.posts.search_all(
            query='from:GidiMarkuszower -is:retweet',
            max_results=100,  # Per page
            tweet_fields=fields_to_include,  # Optional expansions
            expansions=['attachments.media_keys'],
            media_fields=['url', 'variants'],
            start_time='2016-06-01T00:00:00.000Z',
            end_time=date
        ):
            all_posts.extend(page.data)
            print(f"Fetched {len(page.data)} Posts (total: {len(all_posts)})")
            last_post = page.data[-1]
            last_post_date = last_post.created_at if hasattr(last_post, 'created_at') else last_post.get('created_at', '')
            if hasattr(page, 'meta') and page.meta:
                if hasattr(page.meta, 'oldest_id'):
                    oldest_id = page.meta.oldest_id
    except requests.exceptions.HTTPError as e:
        print(e)
        print(f"Failed to fetch all posts, still dumping scraped tweets, if any were scraped")
        if all_posts:
            with open(str(save_file).replace('.json', f'-{str(count)}.json'), "w") as outfile:
                json.dump(all_posts, outfile, indent=4, ensure_ascii=False)
        print(f"Total tweets: {len(all_posts)}")
        time.sleep((60 * 3))
        scrape(oldest_id, last_post_date)



if __name__ == "__main__":
    scrape(1715728574643122658, '2020-01-01T10:30:42.000Z')
# with open("search_all_success.json", "w") as outfile:
#     json.dump(all_posts, outfile, indent=4, ensure_ascii=False)
