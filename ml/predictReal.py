import re
import joblib
import requests
import pandas as pd
from urllib.parse import urlparse
from bs4 import BeautifulSoup

model = joblib.load("phishing_model_full.pkl")   # your original 67-feature model

TOP_TLDS = ["com","org","net","app","uk","co","io","de","ru","au",
            "top","dev","jp","it","edu","fr","br","nl","ca","info"]

SOCIAL_DOMAINS = ["facebook.com","twitter.com","x.com","instagram.com",
                   "linkedin.com","youtube.com"]

def safe_get(url, timeout=8):     
    headers = {"User-Agent": "Mozilla/5.0"}     
    return requests.get(url, headers=headers, timeout=timeout, allow_redirects=True)   
