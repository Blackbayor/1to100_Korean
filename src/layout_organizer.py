import random
from typing import List, Dict, Any

Component = Dict[str, Any]
LogicalUnit = List[Component]

def shuffle_logical_units(logical_units: List[LogicalUnit], seed: int | None = None) -> List[LogicalUnit]:
    """
    '문제 세트' 단위로 1차 셔플하고, 각 세트 내부의 'question_block'만 2차 셔플합니다.
    header/passage는 세트 선두 고정, attachments는 블록과 동반 이동합니다.
    """
    rng = random.Random(seed)
    shufflable_units: List[LogicalUnit] = []

    for unit in logical_units:
        if not unit:
            continue
        first_label = unit[0].get('label')
        if first_label == 'footer':
            continue

        if first_label in ['header', 'passage', 'question_block']:
            fixed_prefix: List[Component] = []
            questions_to_shuffle: List[Component] = []

            for component in unit:
                if component.get('label') in ['header', 'passage']:
                    fixed_prefix.append(component)
                elif component.get('label') == 'question_block':
                    questions_to_shuffle.append(component)
                else:
                    # unknown types follow prefix (rare)
                    fixed_prefix.append(component)

            rng.shuffle(questions_to_shuffle)

            new_unit = fixed_prefix + questions_to_shuffle
            shufflable_units.append(new_unit)

    rng.shuffle(shufflable_units)
    return shufflable_units