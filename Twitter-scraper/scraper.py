from xdk import Client
import json
import os
from dotenv import load_dotenv
import argparse

load_dotenv()
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
client = Client(bearer_token=os.getenv('BEARER_TOKEN'))

parser = argparse.ArgumentParser(description="Scraper CLI")
parser.add_argument("--userid", type=int, required=True)
parser.add_argument("--n", type=int, default=10)
parser.add_argument("--file", type=str)
parser.add_argument("--username", type=str, default="")
parser.add_argument("--since", type=str, default="")
parser.add_argument("--until", type=str, default="")

args = parser.parse_args()
all_posts = []

def scrape(userID, nPosts, file, username, since, until):
    """Default scrape function"""
    if username:
        for uname in client.users.get_by_username(username=username, user_fields=["id"]):
            if uname[1] and len(uname[1]) > 0:
                userID = int(uname[1]['id'])
    for post in client.users.get_posts(id=userID, max_results=nPosts, tweet_fields=fields_to_include,
                                       exclude=['retweets'], expansions=['attachments.media_keys'],
                                       media_fields=['url', 'variants'], until_id=1730117810238464368):
        if post.data and len(post.data) > 0:
            # page_data = getattr(post, 'data', []) or []
            all_posts.extend(post.data)
            with open(file, 'w', encoding="UTF-8") as dump:
                json.dump(post.model_dump(mode='json'), dump, indent=4, ensure_ascii=False)
            first_post = post.data[0]
            post_text = first_post.text if hasattr(first_post, 'text') else first_post.get('text', '')
            print(f"Latest Post: {post_text}")
            break
        else:
            print("Something went wrong, no posts were found")
            break
    with open("test.json", 'w', encoding="UTF-8") as dump:
        json.dump(all_posts, dump, indent=4, ensure_ascii=False)


if __name__ == '__main__':
    scrape(args.userid, args.n, args.file, args.username, args.since, args.until)
