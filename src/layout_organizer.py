import random
from typing import List, Dict, Any

# --- Type Aliases for Clarity ---
Component = Dict[str, Any]
LogicalUnit = List[Component]


def group_components(predictions: List[Component]) -> List[LogicalUnit]:
    """
    탐지된 객체들을 의미있는 '논리적 단위'(예: 지문-문제 세트)로 그룹핑합니다.

    - 객체들을 y-축 기준으로 정렬합니다.
    - 'passage'가 나오면 새 세트를 시작합니다.
    - 'passage' 다음에 나오는 'question_block'들은 해당 세트에 포함시킵니다.
    - 'header', 'figure', 'footer' 등은 독립적인 단위로 처리합니다.

    Args:
        predictions (List[Component]): 한 페이지에 대한 모델의 추론 결과 리스트.

    Returns:
        List[LogicalUnit]: 그룹핑된 논리적 단위들의 리스트.
    """
    if not predictions:
        return []

    # y-좌표(bbox의 y1)를 기준으로 객체들을 위에서 아래로 정렬
    # 'coordinates'는 [x1, y1, x2, y2] 형식이라고 가정
    sorted_predictions = sorted(predictions, key=lambda p: p['coordinates'][1])

    logical_units: List[LogicalUnit] = []
    current_unit: LogicalUnit = []
    
    # 직전 객체가 passage였거나, 혹은 question_block 세트의 일부였는지 추적
    is_part_of_question_set = False

    for component in sorted_predictions:
        label = component['class_name']

        if label in ['header', 'figure', 'footer']:
            if current_unit:
                logical_units.append(current_unit)
            logical_units.append([component])
            current_unit = []
            is_part_of_question_set = False

        elif label == 'passage':
            if current_unit:
                logical_units.append(current_unit)
            current_unit = [component]
            is_part_of_question_set = True # passage는 문제 세트의 시작

        elif label == 'question_block':
            if is_part_of_question_set:
                current_unit.append(component)
            else: # 독립적인 문제(standalone question)
                if current_unit:
                    logical_units.append(current_unit)
                current_unit = [component]
                is_part_of_question_set = True # 독립 문제도 하나의 세트
        
        # 'question_number'는 현재 로직에서 별도 유닛으로 취급하지 않음
        elif label == 'question_number':
            pass # 일단 무시

    # 마지막으로 처리중이던 유닛이 있다면 추가
    if current_unit:
        logical_units.append(current_unit)

    return logical_units


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
        
        # 유닛의 첫번째 컴포넌트 라벨을 기준으로 footer 제외 (버그 수정)
        has_footer = any(comp.get('class_name') == 'footer' for comp in unit)
        if has_footer:
            continue

        # --- ★★★ 내부 셔플 로직 시작 ★★★ ---
        fixed_prefix: List[Component] = []
        questions_to_shuffle: List[Component] = []
        
        # 유닛 내 컴포넌트를 '고정' 부분과 '셔플 대상' 부분으로 분리 (버그 수정)
        for component in unit:
            if component['class_name'] in ['header', 'passage']:
                fixed_prefix.append(component)
            elif component['class_name'] == 'question_block':
                questions_to_shuffle.append(component)
        
        # question_block들만 순서를 섞음
        random.shuffle(questions_to_shuffle)
        
        # 고정 부분과 셔플된 문제들을 다시 합쳐서 새로운 유닛을 생성
        new_shuffled_unit = fixed_prefix + questions_to_shuffle
        if new_shuffled_unit:
            shufflable_units.append(new_shuffled_unit)
        # --- ★★★ 내부 셔플 로직 끝 ★★★ ---

    # 전체 문제 세트의 순서를 셔플 (1차 셔플)
    random.shuffle(shufflable_units)
    
    return shufflable_units