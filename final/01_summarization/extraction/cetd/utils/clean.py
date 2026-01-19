from bs4 import BeautifulSoup, Tag, NavigableString
from typing import Union

CETD_MARK = 'cetd_mark'

def cleanTreeByMark(element: Union[Tag, NavigableString]):
    if isinstance(element, Tag):
        mark = int(element[CETD_MARK])
        
        if mark == 0:
            element.decompose()
        elif mark == 1:
            return
        else:
            for child in list(element.children):
                cleanTreeByMark(child)
                

def extractText(element: Union[Tag, NavigableString]):
    all_content_tags = element.find_all(attrs={CETD_MARK: '1'})
    
    leaf_tags = []
    
    for tag in all_content_tags:
        has_marked_child = any(
            isinstance(child, Tag) and child[CETD_MARK] == '1'
            for child in tag.children
        )
        
        if not has_marked_child:
            leaf_tags.append(tag)
    
    text_list = [tag.get_text(strip=True) for tag in leaf_tags if tag.get_text(strip=True) != ""]
    
    return text_list


def extractBlock(element: Union[Tag, NavigableString]):
    all_content_tags = element.find_all(attrs={CETD_MARK: '1'})
    
    outermost_tags = []
    
    for tag in all_content_tags:
        is_nested = False
        
        for parent in tag.parents:
            if parent is element:
                break
            
            if parent[CETD_MARK] == '1':
                is_nested = True
                break
            
        if not is_nested:
            outermost_tags.append(tag)
            
    return outermost_tags