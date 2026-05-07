from xdk import Client
import json
import os
from dotenv import load_dotenv
load_dotenv()
import argparse
fields_to_include = ["article",
  "attachments",
  "author_id",
  "card_uri",
  "community_id",
  "context_annotations",
  "conversation_id",
  "created_at",
  "display_text_range",
  "edit_history_tweet_ids",
  "entities",
  "geo",
  "id",
  "in_reply_to_user_id",
  "lang",
  "note_tweet",
  "possibly_sensitive",
  "public_metrics",
  "referenced_tweets",
  "reply_settings",
  "scopes",
  "source",
  "suggested_source_links",
  "suggested_source_links_with_counts",
  "text",
  "withheld"]
client = Client(bearer_token=os.getenv('BEARER_TOKEN'))

parser = argparse.ArgumentParser(description="Scraper CLI")
parser.add_argument("--userid", type=int, required=True)
parser.add_argument("--n", type=int, default=10)
parser.add_argument("--file", type=str)
parser.add_argument("--username", type=str, default="")

args = parser.parse_args()

def scrape(userID, nPosts, file, username):
    """Default scrape function"""
    if username:
        for uname in client.users.get_by_username(username=username, user_fields=["id"]):
            if uname[1] and len(uname[1]) > 0:
                userID = int(uname[1]['id'])
    for post in client.users.get_posts(id=userID, max_results=nPosts, tweet_fields=fields_to_include, exclude=['retweets'], expansions=['attachments.media_keys'], media_fields=['url', 'variants']):
        if post.data and len(post.data) > 0:
            with open (file, 'w', encoding="UTF-8") as dump:
                json.dump(post.model_dump(mode='json'), dump, indent=4, ensure_ascii=False)
            first_post = post.data[0]
            post_text = first_post.text if hasattr(first_post, 'text') else first_post.get('text', '')
            print(f"Latest Post: {post_text}")
            break
        else:
            print("Something went wrong, no posts were found")
            break

if __name__ == '__main__':
    scrape(args.userid, args.n, args.file, args.username)