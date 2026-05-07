# GEERT WILDERS ID = 41778159
# Import the client
from xdk import Client
import json
import os
from dotenv import load_dotenv
load_dotenv()
# import click
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
# Replace with your actual Bearer Token
client = Client(bearer_token=os.getenv('BEARER_TOKEN'))
# Fetch recent Posts mentioning "api"
# search_recent returns an Iterator, so iterate over it
# @click.command()
# @click.option('--userID', type=int, default=41778159, prompt='Provide the ID of the user', help='UserID to scrape tweets of')
# @click.option('--nPosts', type=int, default=10, prompt="Enter the amount of tweets you want to scrape", help="n Amount of posts to scrape from UserID")
# @click.option('--file', type=str, prompt="Enter filename to save dump to")
# @click.option('--username', type=str, default="", help="Username to scrape if UserID is unknown")

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
        for post in client.users.get_posts(id=userID, max_results=nPosts, tweet_fields=fields_to_include, exclude=['retweets'], expansions=['attachments.media_keys'], media_fields=['url']):
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
    else:
                                                                            #tweet_fields=fields_to_include
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



# for post in client.users.get_posts(id=41778159):
#     if post.data and len(post.data) > 0:
#         # y = json.dumps(post.data)
#         with open('posts.json', 'w', encoding="UTF-8") as file:
#             json.dump(post.data, file, indent=4, ensure_ascii=False)
#         # print(f"Post data: {post.data}")
#         first_post = post.data[0]
#         post_text = first_post.text if hasattr(first_post, 'text') else first_post.get('text', '')
#         print(f"Latest Post: {post_text}")
#         break
#     else:
#         print("No Posts found.")
#         break        
# for page in client.posts.search_recent(query="api", max_results=10):
#     if page.data and len(page.data) > 0:
#         # Access first Post - Pydantic models support both attribute and dict access
#         first_post = page.data[0]
#         post_text = first_post.text if hasattr(first_post, 'text') else first_post.get('text', '')
#         print(f"Latest Post: {post_text}")
#         break
#     else:
#         print("No Posts found.")
#         break