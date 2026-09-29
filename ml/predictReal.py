import re
import joblib
import requests
import pandas as pd
from urllib.parse import urlparse
from bs4 import BeautifulSoup

model = joblib.load("phishing_model_full.pkl")   # original 67-feature model
# NOTE: In this model's training data, label = 0 means PHISHING, label = 1 means LEGITIMATE

TOP_20_TLDS = ["com","org","net","app","uk","co","io","de","ru","au",
               "top","dev","jp","it","edu","fr","br","nl","ca","info"]

TLD_DUMMY_COLS = ['au','br','ca','co','com','de','dev','edu','fr','info',
                   'io','it','jp','net','nl','org','other','ru','top','uk']

SOCIAL_DOMAINS = ["facebook.com","twitter.com","x.com","instagram.com",
                   "linkedin.com","youtube.com","tiktok.com"]

def safe_get(url, timeout=8):
    headers = {"User-Agent": "Mozilla/5.0"}
    return requests.get(url, headers=headers, timeout=timeout, allow_redirects=True)

def get_tld_dummies(tld):
    tld_group = tld if tld in TOP_20_TLDS else "other"
    dummies = {f"TLD_{t}": 0 for t in TLD_DUMMY_COLS}
    if tld_group != "app":   # "app" was dropped as the reference category during training
        key = f"TLD_{tld_group}"
        if key in dummies:
            dummies[key] = 1
    return dummies

def extract_features(url, resp):
    """Build the feature row using the URL plus an already-fetched response."""
    parsed = urlparse(url)
    domain = parsed.netloc.replace("www.", "")
    tld = domain.split(".")[-1] if "." in domain else ""

    # ---- URL-string features ----
    letters = sum(c.isalpha() for c in url)
    digits = sum(c.isdigit() for c in url)
    specials = len(url) - letters - digits
    is_ip = bool(re.match(r"^\d{1,3}(\.\d{1,3}){3}$", domain))

    obf_chars = len(re.findall(r"%[0-9A-Fa-f]{2}", url))

    max_run, run = 1, 1
    for i in range(1, len(url)):
        run = run + 1 if url[i] == url[i-1] else 1
        max_run = max(max_run, run)

    # ---- Parse the already-fetched page ----
    html = resp.text
    redirects = len(resp.history)
    final_domain = urlparse(resp.url).netloc
    soup = BeautifulSoup(html, "html.parser")

    lines = html.splitlines() or [""]
    title_tag = soup.find("title")
    title_text = title_tag.text.strip() if title_tag else ""

    forms = soup.find_all("form")
    external_form = any(
        f.get("action") and urlparse(f.get("action")).netloc not in ("", final_domain)
        for f in forms
    )

    links = soup.find_all("a", href=True)
    self_ref = sum(1 for a in links if final_domain in a["href"] or a["href"].startswith("/"))
    empty_ref = sum(1 for a in links if a["href"].strip() in ("", "#", "javascript:void(0)"))
    external_ref = len(links) - self_ref - empty_ref

    page_text = soup.get_text().lower()

    features = {
        "URLLength": len(url),
        "DomainLength": len(domain),
        "IsDomainIP": int(is_ip),
        "CharContinuationRate": max_run / max(len(url), 1),
        "TLDLegitimateProb": 0.5,          # approximation
        "URLCharProb": 0.05,               # approximation
        "TLDLength": len(tld),
        "NoOfSubDomain": max(domain.count(".") - 1, 0),
        "HasObfuscation": int(obf_chars > 0),
        "NoOfObfuscatedChar": obf_chars,
        "ObfuscationRatio": obf_chars / max(len(url), 1),
        "LetterRatioInURL": letters / max(len(url), 1),
        "NoOfDegitsInURL": digits,
        "DegitRatioInURL": digits / max(len(url), 1),
        "NoOfEqualsInURL": url.count("="),
        "NoOfQMarkInURL": url.count("?"),
        "NoOfAmpersandInURL": url.count("&"),
        "NoOfOtherSpecialCharsInURL": specials,
        "SpacialCharRatioInURL": specials / max(len(url), 1),
        "IsHTTPS": int(parsed.scheme == "https"),
        "LineOfCode": len(lines),
        "LargestLineLength": max((len(l) for l in lines), default=0),
        "HasTitle": int(bool(title_text)),
        "DomainTitleMatchScore": 100.0 if domain.split(".")[0] in title_text.lower() else 0.0,
        "HasFavicon": int(bool(soup.find("link", rel=lambda x: x and "icon" in x.lower()))),
        "Robots": 0,
        "IsResponsive": int(bool(soup.find("meta", attrs={"name": "viewport"}))),
        "NoOfURLRedirect": redirects,
        "NoOfSelfRedirect": redirects,
        "HasDescription": int(bool(soup.find("meta", attrs={"name": "description"}))),
        "NoOfPopup": html.lower().count("window.open("),
        "NoOfiFrame": len(soup.find_all("iframe")),
        "HasExternalFormSubmit": int(external_form),
        "HasSocialNet": int(any(s in html for s in SOCIAL_DOMAINS)),
        "HasSubmitButton": int(bool(soup.find("input", {"type": "submit"}) or soup.find("button"))),
        "HasHiddenFields": int(bool(soup.find("input", {"type": "hidden"}))),
        "HasPasswordField": int(bool(soup.find("input", {"type": "password"}))),
        "Bank": int("bank" in page_text),
        "Pay": int("pay" in page_text),
        "Crypto": int("crypto" in page_text or "bitcoin" in page_text),
        "HasCopyrightInfo": int("©" in page_text or "copyright" in page_text),
        "NoOfImage": len(soup.find_all("img")),
        "NoOfCSS": len(soup.find_all("link", rel="stylesheet")) + len(soup.find_all("style")),
        "NoOfJS": len(soup.find_all("script")),
        "NoOfSelfRef": self_ref,
        "NoOfEmptyRef": empty_ref,
        "NoOfExternalRef": external_ref,
    }

    features.update(get_tld_dummies(tld))
    return pd.DataFrame([features])

def predict_url(url):
    print(f"\nURL: {url}")

    try:
        resp = safe_get(url)
    except Exception as e:
        print(f"Could not analyze — page unreachable ({e})")
        print("Note: an unreachable page could itself be suspicious (dead link, blocked, taken down) — treat with caution rather than assuming safe.")
        return "UNKNOWN"

    features = extract_features(url, resp)
    features = features.reindex(columns=model.feature_names_in_, fill_value=0)

    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0]

    # IMPORTANT: in this model, label 0 = phishing, label 1 = legitimate
    label = "PHISHING" if prediction == 0 else "LEGITIMATE"
    confidence = probability[prediction] * 100

    print(f"Prediction: {label}")
    print(f"Confidence: {confidence:.2f}%")
    return label

if __name__ == "__main__":
    predict_url("https://www.google.com")
    predict_url("https://www.wikipedia.org")
    predict_url("https://github.com")
    predict_url("http://localhost:8000/fake_phish_test.html")