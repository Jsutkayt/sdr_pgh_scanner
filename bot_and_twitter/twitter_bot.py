import tweepy
import os
from dotenv import load_dotenv

# Load your .env file
load_dotenv("/Users/jacksonsutkaytis/Documents/Pgh_Scanner_Project/sdr_pgh_scanner/.env")

# Grab your keys (same variables)
consumer_key = os.getenv("XCONSUMER_KEY")
consumer_secret = os.getenv("XCONSUMER_SECRET")
access_token = os.getenv("XACCESS_TOKEN")
access_token_secret = os.getenv("XACCESS_SECRET")

# Create the v2 Client (this switches to API v2 endpoints)
client = tweepy.Client(
    consumer_key=consumer_key,
    consumer_secret=consumer_secret,
    access_token=access_token,
    access_token_secret=access_token_secret,
    wait_on_rate_limit=True  # Optional: auto-waits if you hit limits
)

tweet_text = "Can Anyone See This? (now via v2 API!)"

try:
    # This calls the v2 endpoint: POST /2/tweets
    response = client.create_tweet(text=tweet_text)
    print("Tweet posted successfully!")
    print("Tweet ID:", response.data['id'])
    print("Link: https://x.com/pgh_dispatch/status/" + response.data['id'])
except tweepy.TweepyException as e:
    print("Error posting tweet:", e)
    # For more details on errors (like 403 or rate limits), print the full response
    if hasattr(e, 'response'):
        print("Full error details:", e.response.text)
