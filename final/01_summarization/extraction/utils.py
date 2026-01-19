import re
import itertools
from bs4 import BeautifulSoup, Tag

import spacy
nlp = spacy.load("en_core_web_sm")

CETD_MARK = 'cetd_mark'

def get_raw_text(html_text):
    soup = BeautifulSoup(html_text, 'html.parser')
    body = soup.body
    if body is None:
        return ""
    return body.get_text(separator=' ', strip=True)
    
def table2text(html):
    if html in ["Fail: no body tag found", "Fail: something is wrong"]:
        return html
    
    soup = BeautifulSoup(html, 'html.parser')
    table_list = soup.find_all('table')
    
    for i, table in enumerate(table_list):
        table_str = table.get_text(separator=' | ', strip=True)
        table_tag_str = f"The contents of the table are as follows: {table_str}."
        description_tag = soup.new_tag('p')
        description_tag.attrs = table.attrs
        description_tag.attrs["cetd_mark"] = "1"
        description_tag.string = table_tag_str
        
        table.replace_with(description_tag)
        
    return str(soup)

def extract_text(html):
    if html in ["Fail: no body tag found", "Fail: something is wrong"]:
        return []
    
    element = BeautifulSoup(html, 'html.parser')
    text_list = element.get_text(separator="[SSL_SEP]", strip=True).split("[SSL_SEP]")
    
    return text_list

def extract_block(html):
    if html in ["Fail: no body tag found", "Fail: something is wrong"]:
        return []
    
    element = BeautifulSoup(html, 'html.parser')
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

def extract_text_for_block(element):
    text_list = element.get_text(separator="[SSL_SEP]", strip=True).split("[SSL_SEP]")
    return text_list

def extract_text_for_blocks(blocks):
    if blocks == []:
        return []
    
    text_list = []
    
    for block in blocks:
        text_list.append(extract_text_for_block(block))
    
    return text_list

def calculate_percentage(cetd_block_content, neuscraper_content):
    denominator = len(cetd_block_content)
    if denominator == 0:
        return 0.0
    intersection = []
    for leaf in cetd_block_content:
        if leaf in neuscraper_content:
            intersection.append(leaf)
    numerator = len(intersection)
    
    return round(numerator / denominator, 2)

def parallel_cross_validation(cetd_blocks, cetd_blocks_contents, neuscraper_content):
    percentage = [calculate_percentage(block_content, neuscraper_content) for block_content in cetd_blocks_contents]
    select_idx = [idx for idx, value in enumerate(percentage) if value != 0]
    selected_blocks = [cetd_blocks[idx] for idx in select_idx]
    return selected_blocks

def table2text_for_block(soup):
    soup = BeautifulSoup(str(soup), 'html.parser')
    table_list = soup.find_all('table')
    
    for i, table in enumerate(table_list):
        table_str = table.get_text(separator=' | ', strip=True)
        table_tag_str = f"The contents of the table are as follows: {table_str}."
        description_tag = soup.new_tag('p')
        description_tag.attrs = table.attrs
        description_tag.attrs["cetd_mark"] = "1"
        description_tag.string = table_tag_str
        
        table.replace_with(description_tag)
        
    return soup

def table2text_for_blocks(blocks):
    if blocks == []:
        return []
    
    new_blocks = []
    
    for block in blocks:
        new_blocks.append(table2text_for_block(block))
        
    return new_blocks

def leaf_text_preprocessing(leaf_nodes):
    preprocessed_texts = []
    
    for leaf in leaf_nodes:
        leaf = leaf.encode('utf-8').decode('unicode_escape')
        new_leaf = leaf.replace('\n', ' ').replace('\t', ' ') # \n, \t 제거
        new_leaf = re.sub(r'\s+', ' ', new_leaf) # 다중 공백을 단일 공백으로 변환
        new_leaf = [sent.text.strip() for sent in nlp(new_leaf).sents] # 문장 단위로 분리
        preprocessed_texts.append(new_leaf)
        
    preprocessed_texts = list(itertools.chain.from_iterable(preprocessed_texts)) # 중첩 리스트 -> 1차원 리스트
    preprocessed_leaves = [text for text in preprocessed_texts if text != ''] # 빈 문자열 제거
    
    return preprocessed_leaves

def text_preprocessing(raw_texts):
    raw_texts = raw_texts.encode('utf-8').decode('unicode_escape')
    new_texts = raw_texts.replace('\n', ' ').replace('\t', ' ') # \n, \t 제거
    new_texts = re.sub(r'\s+', ' ', new_texts) # 다중 공백을 단일 공백으로 변환
    
    if len(new_texts) > nlp.max_length:
        pre_split_texts = re.split(r'(?<=[.!?])\s+', new_texts) # 정규표현식을 사용해 마침표, 물음표, 느낌표 뒤의 공백을 기준으로 1차 분리
        final_sents = []
        for segment in pre_split_texts:
            if segment: # 빈 문자열이 아닐 경우에만 처리
                doc = nlp(segment)
                final_sents.extend([sent.text.strip() for sent in doc.sents])
        return [text for text in final_sents if text] # 빈 문자열 제거
    
    else:
        doc = nlp(new_texts)
        return [sent.text.strip() for sent in doc.sents if sent.text.strip()] # 빈 문자열 제거