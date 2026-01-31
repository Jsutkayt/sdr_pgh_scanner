from dotenv import load_dotenv
import os

load_dotenv("/Users/jacksonsutkaytis/Documents/Pgh_Scanner_Project/sdr_pgh_scanner/.env")

print("XCONSUMER_KEY =", os.getenv("XCONSUMER_KEY"))
print("XCONSUMER_KEY_SECRET =", os.getenv("XCONSUMER_SECRET"))
print("XACCESS_KEY =", os.getenv("XACCESS_TOKEN"))
print("XACCESS_TOKEN_SECRET =", os.getenv("XACCESS_SECRET"))
