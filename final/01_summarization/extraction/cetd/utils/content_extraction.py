from bs4 import BeautifulSoup, Tag, NavigableString
from typing import Union
import math
import sys

eps = sys.float_info.epsilon

def countChar(element: Union[Tag, NavigableString]):
    if isinstance(element, Tag):
        char_num = len(element.text)
        element['cetd_char_num'] = str(char_num)
        
        for child in element.children:
            countChar(child)
           

def countTag(element: Union[Tag, NavigableString]):
    if isinstance(element, Tag):
        tag_num = 0
        
        for child in element.children:
            if isinstance(child, Tag):
                countTag(child)
                tag_num += int(child['cetd_tag_num'])+1
                
        element['cetd_tag_num'] = str(tag_num)
    
    
def updateLinkChar(element: Union[Tag, NavigableString]):
    if isinstance(element, Tag):
        for child in element.children:
            if isinstance(child, Tag):
                child['cetd_linkchar_num'] = child['cetd_char_num']
                updateLinkChar(child)
    

# call this function after countChar
def countLinkChar(element: Union[Tag, NavigableString]):
    if isinstance(element, Tag):
        linkchar_num = 0
        tag_name = element.name
    
        for child in element.children:
            countLinkChar(child)
        
        # deal with hyperlink and sth like that
        if tag_name in ['a', 'button', 'select']:
            linkchar_num = int(element['cetd_char_num'])
            updateLinkChar(element)
        else:
            for child in element.children:
                if isinstance(child, Tag):
                    linkchar_num += int(child['cetd_linkchar_num'])
            
        element['cetd_linkchar_num'] = str(linkchar_num)
    
    
def updateLinkTag(element: Union[Tag, NavigableString]):
    if isinstance(element, Tag):
        for child in element.children:
            if isinstance(child, Tag):
                child['cetd_linktag_num'] = child['cetd_tag_num']
                updateLinkTag(child)
    

# call this function after countChar, countLinkChar
def countLinkTag(element: Union[Tag, NavigableString]):
    if isinstance(element, Tag):
        linktag_num = 0
        tag_name = element.name
        
        for child in element.children:
            countLinkTag(child)
            
        # deal with hyperlink and sth like that
        if tag_name in ['a', 'button', 'select']:
            linktag_num = int(element['cetd_tag_num'])
            updateLinkChar(element) #########################################################################
        else:
            for child in element.children:
                if isinstance(child, Tag):
                    linktag_num += int(child['cetd_linktag_num'])
                    tag_name = child.name
                    
                    # if a tag is <a> or sth plays similar role in web pages, then anchor number add 1
                    if tag_name in ['a', 'button', 'select']:
                        linktag_num += 1
                    else:
                        child_linktag_num = int(child['cetd_linktag_num'])
                        child_tag_num = int(child['cetd_tag_num'])
                        child_char_num = int(child['cetd_char_num'])
                        child_linkchar_num = int(child['cetd_linkchar_num'])
                        
                        # child_linktag_num != 0: there are some anchor under this child
                        if child_linktag_num == child_tag_num and child_char_num == child_linkchar_num and child_linktag_num != 0:
                            linktag_num += 1
                            
        element['cetd_linktag_num'] = str(linktag_num)        
    

# call this function after countChar, countTag, countLinkChar, countLinkTag
def computeTextDensity(element: Union[Tag, NavigableString], ratio: float):
    if isinstance(element, Tag):
        char_num = int(element['cetd_char_num'])
        tag_num = int(element['cetd_tag_num'])
        linkchar_num = int(element['cetd_linkchar_num'])
        linktag_num = int(element['cetd_linktag_num'])
        
        text_density = 0.0
        if char_num == 0:
            text_density = 0
        else:
            un_linkchar_num = char_num - linkchar_num
            
            if tag_num == 0:
                tag_num = 1
            if linkchar_num == 0:
                linkchar_num = 1
            if linktag_num == 0:
                linktag_num = 1
            if un_linkchar_num == 0:
                un_linkchar_num = 1
                
            text_density = (1.0 * char_num / tag_num) * math.log((1.0 * char_num * tag_num) / (1.0 * linkchar_num * linktag_num)) / math.log(math.log(1.0 * char_num * linkchar_num / un_linkchar_num + ratio * char_num + math.exp(1.0)))
            # text_density = 1.0 * char_num / tag_num
            
        element['cetd_text_density'] = str(text_density)
        
        for child in element.children:
            computeTextDensity(child, ratio)
    
    
def computeDensitySum(element: Union[Tag, NavigableString], ratio: float):
    if isinstance(element, Tag):
        density_sum = 0.0
        char_num_sum = 0
        content = element.text
        
        from_idx = 0
        index = 0
        length = 0
        
        if not any(isinstance(c, Tag) for c in element.children):
            density_sum = float(element['cetd_text_density'])
        else:
            for child in element.children:
                computeDensitySum(child, ratio)
            for child in element.children:
                if isinstance(child, Tag):
                    density_sum += float(child['cetd_text_density'])
                    char_num_sum += int(child['cetd_char_num'])
                    
                    # text before tag
                    child_content = child.text
                    index = content.find(child_content, from_idx)
                    
                    if index > -1:
                        length = index - from_idx
                        if length > 0:
                            density_sum += length * math.log(1.0 * length) / math.log(math.log(ratio * length + math.exp(1.0)))
                        from_idx = index + len(child_content)
        
            # text after tag
            length = len(element.text) - from_idx
            if length > 0:
                density_sum += length * math.log(1.0 * length) / math.log(math.log(ratio * length + math.exp(1.0)))

        element['cetd_density_sum'] = str(density_sum)
        
    
def findMaxDensitySum(element: Union[Tag, NavigableString]):
    if isinstance(element, Tag):
        max_density_sum = float(element['cetd_density_sum'])
        temp_max_density_sum = 0.0
        
        for child in element.children:
            if isinstance(child, Tag):
                temp_max_density_sum = findMaxDensitySum(child)
                if temp_max_density_sum - max_density_sum > eps:
                    max_density_sum = temp_max_density_sum
            else:
                continue
                
        # record the max_density_sum under the element
        element['cetd_max_density_sum'] = str(max_density_sum)
        return max_density_sum
    

def searchTag(element: Union[Tag, NavigableString], attribute: str, value: float):
    if isinstance(element, Tag):
        target = element
    
        attr_value = float(element[attribute])
        if abs(attr_value - value) < eps:
            return target
    
        for child in element.find_all(attrs={attribute: str(value)}):
            return child
        return None
    
    
def getThreshold(element: Union[Tag, NavigableString], max_density_sum: float):
    if isinstance(element, Tag):
        threshold = -1.0
        
        # search the max densitysum element
        target = searchTag(element, 'cetd_density_sum', max_density_sum)
        threshold = float(target['cetd_text_density'])
        setMark(target, 1)
        
        parent = target.parent
        while True:
            if isinstance(parent, Tag):
                if parent.name != 'html':
                    text_density = float(parent['cetd_text_density'])
                    if threshold - text_density > -1 * eps:
                        threshold = text_density
                    parent['cetd_mark'] = "2"
                    parent = parent.parent
                else:
                    break
            else:
                break
            
        return threshold
    
    
def setMark(element: Union[Tag, NavigableString], mark: int):
    if isinstance(element, Tag):
        element['cetd_mark'] = str(mark)
        
        for child in element.children:
            setMark(child, mark)        
    
    
def findMaxDensitySumTag(element: Union[Tag, NavigableString], max_density_sum: float):
    if isinstance(element, Tag):
        target = searchTag(element, 'cetd_density_sum', max_density_sum)
        
        mark = int(target['cetd_mark'])
        if mark == 1:
            return
        
        setMark(target, 1)
        
        parent = target.parent
        while True:
            if isinstance(parent, Tag):
                if parent.name != 'html':
                    parent['cetd_mark'] = "2"
                    parent = parent.parent
                else:
                    break
            else:
                break
    
    
def markContent(element: Union[Tag, NavigableString], threshold: float):
    if isinstance(element, Tag):
        text_density = float(element['cetd_text_density'])
        max_density_sum = float(element['cetd_max_density_sum'])
        mark = int(element['cetd_mark'])
    
        if mark != 1 and (text_density - threshold > -1 * eps):
            findMaxDensitySumTag(element, max_density_sum)
            for child in element.children:
                markContent(child, threshold)
    

# another method: collapse the character count of link anchors to 0
def updateLinkCharVariant(element: Union[Tag, NavigableString]):
    if isinstance(element, Tag):
        for child in element.children:
            if isinstance(child, Tag):
                child['cetd_char_num'] = "0"
                updateLinkCharVariant(child)
    
    
def countCharVariant(element: Union[Tag, NavigableString]):
    if isinstance(element, Tag):
        char_num = 0
        plain_text_length = len(element.text)
        child_plain_text_length = 0
        tag_name = element.name
        
        if tag_name in ['a', 'button', 'select']:
            element['cetd_char_num'] = "0"
        else:
            if not any(isinstance(c, Tag) for c in element.children):
                element['cetd_char_num'] = str(plain_text_length)
            else:
                for child in element.children:
                    if isinstance(child, Tag):
                        countCharVariant(child)
                for child in element.children:
                    if isinstance(child, Tag):
                        char_num += int(child['cetd_char_num'])
                        child_plain_text_length = len(child.text)
                        plain_text_length =+ child_plain_text_length
                char_num = char_num + plain_text_length
                element['cetd_char_num'] = str(char_num)
    
    
def computeTextDensityVariant(element: Union[Tag, NavigableString]):
    if isinstance(element, Tag):
        char_num = int(element['cetd_char_num'])
        tag_num = int(element['cetd_tag_num'])
        
        text_density = 0.0
        
        if char_num == 0:
            text_density = 0
        else:
            if tag_num == 0:
                tag_num = 1
            text_density = 1.0 * char_num / tag_num
            
        element['cetd_text_density'] = str(text_density)
        
        for child in element.children:
            computeTextDensityVariant(child)
    
    
def computeDensitySumVariant(element: Union[Tag, NavigableString]):
    if isinstance(element, Tag):
        density_sum = 0.0
        char_num_sum = 0
        char_num = 0
        
        if not any(isinstance(c, Tag) for c in element.children):
            density_sum = float(element['cetd_text_density'])
        else:
            for child in element.children:
                computeDensitySumVariant(child)
            for child in element.children:
                if isinstance(child, Tag):
                    density_sum += float(child['cetd_text_density'])
                    char_num_sum += int(child['cetd_char_num'])
            char_num = int(element['cetd_char_num'])
            if char_num > char_num_sum:
                char_num -= char_num_sum
                density_sum += char_num
                
        element['cetd_density_sum'] = str(density_sum)