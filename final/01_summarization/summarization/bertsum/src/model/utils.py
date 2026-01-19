from typing import Optional, List

def get_ngrams(
        tokens: List[str],
        n: int,
):
    """
        주어진 토큰 리스트에사 n-gram 집합을 생성
    """
    ngrams = set()
    num = len(tokens) - n
    for i in range(num + 1):
        ngrams.add(tuple(tokens[i:i + n]))
    return ngrams


def ngram_blocking(
        sent: str, # 새로 고려 중인 문장
        can_sum: List[str], # 현재까지 선택된 요약 문장 리스트
        ngram: int, # 중복을 차단할 n-gram 크기
):  
    """
        주어진 문장이 현재까지 선택된 문장들과 n-gram이 중복되는지 확인
    """
    sent_tri = get_ngrams(sent.split(), ngram)
    for can_sent in can_sum:
        can_tri = get_ngrams(can_sent.split(), ngram)
        if len(sent_tri.intersection(can_tri)) > 0:
            return True # 중복이 있으면 True 반환
    # 중복이 없으면 False 반환
    return False


def tri_blocking(sent, can_sum):
    return ngram_blocking(sent, can_sum, 3)


def quad_blocking(sent, can_sum):
    return ngram_blocking(sent, can_sum, 4)


def get_candidate_sum(
        text: str, # 전체 텍스트
        prediction: List[int], # 중요도 높은 문장의 인덱스(중요도 순 정렬)
        sum_size: Optional[int] = None, # 선택할 요약 문장 수 제한
        n_block: int = 3 # n-gram 차단 크기
):  
    """
        주어진 텍스트와 예측된 문장 인덱스 리스트를 기반으로 요약 후보 문장 생성
        중복되는 n-gram이 있는 문장은 제외
    """
    can_sum = []
    for i, sent_id in enumerate(prediction):
        sent = text[sent_id]
        # n-gram 중복이 없을 경우에만 요약 문장으로 추가
        if not ngram_blocking(sent, can_sum, n_block):
            can_sum.append(sent)

        # 최대 문장 수 제한 도달 시 중단
        if sum_size and (len(can_sum) == sum_size):
            break

    return can_sum

