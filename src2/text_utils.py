import re

_UNICODE_TABLE = str.maketrans({
    "\u2018": "'",
    "\u2019": "'",
    "\u201c": '"',
    "\u201d": '"',
    "\u2026": "...",
    "\u2011": "-",
    "\u2013": "-",
    "\u2014": " ",
    "\u2080": "0",
    "\u2081": "1",
    "\u2082": "2",
    "\u2083": "3",
    "\u2084": "4",
    "\u2085": "5",
    "\u2086": "6",
    "\u2087": "7",
    "\u2088": "8",
    "\u2089": "9",
    "\u2066": "",
    "\u2067": "",
    "\u2068": "",
    "\u2069": "",
    "\u200b": "",
    "\u200c": "",
    "\u200d": "",
    "\ufeff": "",
})

_EMOJI_RE = re.compile(
    r"[\U0001F300-\U0001F9FF"
    r"\U0001FA00-\U0001FAFF"
    r"\u2600-\u27BF"
    r"\uFE00-\uFE0F]+"
)
_AT_USER_RE = re.compile(r"@user\b", re.IGNORECASE)
_AT_NAMED_RE = re.compile(r"@(\w+)")
_HASHTAG_RE = re.compile(r"#(\w+)")
_THREAD_RE = re.compile(r"\b\d+/\s*")
_QUOTE_RE = re.compile(r'["\u201c\u201d](.*?)["\u201c\u201d]')
_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_LEADING_MENTION_CHAIN_RE = re.compile(r"^\s*(?:@\w+[\s,;:!.\-]*){2,}")
_CAMEL_SPLIT_1_RE = re.compile(r"([a-z])([A-Z])")
_CAMEL_SPLIT_2_RE = re.compile(r"([A-Z]+)([A-Z][a-z])")
_TRAILING_SOURCE_RE = re.compile(
    r"\s+(?:quelle|source|via|mehr dazu|en savoir plus|lire ici)\s*[:\-].*$",
    re.IGNORECASE,
)


def _split_camel_case(token: str) -> str:
    token = _CAMEL_SPLIT_2_RE.sub(r"\1 \2", token)
    token = _CAMEL_SPLIT_1_RE.sub(r"\1 \2", token)
    return token


def _expand_hashtag_token(match: re.Match[str]) -> str:
    token = match.group(1).replace("_", " ")
    token = _split_camel_case(token)
    return token


def preprocess_de_query_raw(text: str) -> str:
    text = _URL_RE.sub(" ", text)
    text = _LEADING_MENTION_CHAIN_RE.sub(" ", text)
    text = _HASHTAG_RE.sub(_expand_hashtag_token, text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_de_query(text: str) -> str:
    text = _split_camel_case(text)

    for prefix in (
        "Pub alert!",
        "Pub alert:",
        "Hier noch ein Beitrag",
        "Hier ein Beitrag",
        "Frischer Beitrag",
        "Neuer Artikel",
        "Neue Studie",
    ):
        if text.lower().startswith(prefix.lower()):
            text = text[len(prefix):].lstrip(" ,:;-")

    lower = text.lower()
    for marker in (" quelle:", " source:", " lesen sie", " im bericht", " mehr dazu"):
        pos = lower.find(marker)
        if pos >= 40:
            text = text[:pos].rstrip(" ,:;-")
            break

    return re.sub(r"\s+", " ", text).strip()


def preprocess_translation_input(text: str, lang: str) -> str:
    text = text.translate(_UNICODE_TABLE)
    text = _URL_RE.sub(" ", text)
    text = _LEADING_MENTION_CHAIN_RE.sub(" ", text)
    text = _HASHTAG_RE.sub(_expand_hashtag_token, text)

    if lang == "de":
        text = normalize_de_query(text)
    elif lang == "fr":
        text = _TRAILING_SOURCE_RE.sub("", text)

    return re.sub(r"\s+", " ", text).strip()


def clean_tweet(text: str) -> str:
    text = text.translate(_UNICODE_TABLE)
    text = _URL_RE.sub(" ", text)
    text = _EMOJI_RE.sub(" ", text)
    text = _AT_USER_RE.sub("", text)
    text = _AT_NAMED_RE.sub(r"\1", text)
    text = _HASHTAG_RE.sub(r"\1", text)
    text = _THREAD_RE.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def extract_quoted_boost(text: str) -> str:
    quotes = [q.strip() for q in _QUOTE_RE.findall(text) if len(q.strip()) > 5]
    if not quotes:
        return text
    return " ".join(quotes) + " " + text
