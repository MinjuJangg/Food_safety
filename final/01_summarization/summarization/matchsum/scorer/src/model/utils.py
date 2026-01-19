from typing import Optional, List

def get_candidate_sum(
        text: str, # 전체 텍스트
        prediction: List[int], # 중요도 높은 문장의 인덱스(중요도 순 정렬)
        sum_size: Optional[int] = None, # 선택할 요약 문장 수 제한
):  
    """
        주어진 텍스트와 예측된 문장 인덱스 리스트를 기반으로 요약 후보 문장 생성
    """
    can_sum = []
    for i, sent_id in enumerate(prediction):
        sent = text[sent_id]
        can_sum.append(sent)

        # 최대 문장 수 제한 도달 시 중단
        if sum_size and (len(can_sum) == sum_size):
            break

    return can_sum

