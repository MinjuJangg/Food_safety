from bs4 import BeautifulSoup, Comment

NON_CONTENT_TAGS = [
  "script",
  "noscript",
  "style",
  "nav",
  "header",
  "footer",
  "img",
  "svg",
  "video",
  "audio",
  "form",
  "label",
  "input",
  "select",
  "option",
  "button",
  "object",
  "embed",
  "iframe",
  "canvas",
  "map",
  "area",
  "picture",
  "source",
  "track",
  "wbr",
  "slot",
  "template",
  "datalist",
]


def stripNonContentTags(soup):
    for tag in NON_CONTENT_TAGS:
        for el in soup.find_all(tag):
            el.decompose() # content로 간주하지 않는 태그를 soup에서 제거
            
    for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
        comment.extract() # html의 주석 제거