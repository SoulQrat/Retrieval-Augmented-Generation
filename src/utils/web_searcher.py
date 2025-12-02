import requests
from bs4 import BeautifulSoup
import re

class WebSearcher:
    def __init__(self, timeout=10):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/605.1.15 (KHTML, like Gecko) "
                "Version/17.2 Safari/605.1.15"
            )
        })
        self.timeout = timeout

    def _search(self, query: str):
        r = self.session.get("https://html.duckduckgo.com/html/", params={"q": query}, timeout=self.timeout)
        soup = BeautifulSoup(r.text, "html.parser")

        links = soup.find_all("a", href=True)
        for link in links:
            href = link["href"]
            if "duckduckgo.com" in href:
                continue
            if href.startswith("http"):
                return href
        return None

    def _get_page_text(self, url: str):
        resp = self.session.get(url, timeout=self.timeout)
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "noscript", "header", "footer", "svg"]):
            tag.decompose()
        text = " ".join(soup.stripped_strings)
        text = re.sub(r"\s+", " ", text)
        return text

    def search(self, query: str):
        url = self._search(query)

        if url is None:
            return {'text': None, 'source': None}
        return {'text': self._get_page_text(url), 'source': url}