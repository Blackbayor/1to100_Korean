import random
from typing import List, Dict, Any

# --- Type Aliases for Clarity ---
Component = Dict[str, Any]
LogicalUnit = List[Component]

def shuffle_logical_units(logical_units: List[LogicalUnit]) -> List[LogicalUnit]:
    """
    논리적 단위 리스트와 그 내부의 문제들을 셔플합니다.
    - '문제 세트' 단위로 1차 셔플합니다.
    - ★★★ 각 문제 세트 내부의 'question_block'들도 순서를 2차 셔플합니다. ★★★
    - 'footer'는 결과에서 제외됩니다.
    """
    shufflable_units: List[LogicalUnit] = []
    
    # 셔플할 유닛만 필터링
    for unit in logical_units:
        if not unit: continue
        
        first_label = unit[0]['label']
        if first_label == 'footer':
            continue # footer는 최종 결과에서 제외
        
        if first_label in ['header', 'passage', 'question_block']:
            
            # --- ★★★ 내부 셔플 로직 시작 ★★★ ---
            fixed_prefix: List[Component] = []
            questions_to_shuffle: List[Component] = []
            
            # 유닛 내 컴포넌트를 '고정' 부분과 '셔플 대상' 부분으로 분리
            for component in unit:
                if component['label'] in ['header', 'passage']:
                    fixed_prefix.append(component)
                elif component['label'] == 'question_block':
                    questions_to_shuffle.append(component)
            
            # question_block들만 순서를 섞음
            random.shuffle(questions_to_shuffle)
            
            # 고정 부분과 셔플된 문제들을 다시 합쳐서 새로운 유닛을 생성
            new_shuffled_unit = fixed_prefix + questions_to_shuffle
            shufflable_units.append(new_shuffled_unit)
            # --- ★★★ 내부 셔플 로직 끝 ★★★ ---

    # 전체 문제 세트의 순서를 셔플 (1차 셔플)
    random.shuffle(shufflable_units)
    
    return shufflable_units