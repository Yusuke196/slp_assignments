import pandas as pd


def preprocess(sents: list[list], feat_type: str = 'all') -> list[pd.DataFrame]:
    # これはtrain dataの前処理のための関数だが、以下の関数の中にはtest dataの処理にも使うものがある
    steps = []
    for sent in sents:
        steps += _sent_to_steps(sent)

    X, y = _encode(steps, feat_type)
    return X, y


def _sent_to_steps(sent: list[dict]) -> list[list]:
    stack = [{'id': 0, 'token': 'ROOT', 'pos': 'ROOT', 'head': -1}] * 2
    buffer = sent
    steps = []  # stackの最後の2つとbufferの先頭、それからactionをここに入れていく

    while buffer != [] or len(stack) > 2:
        if _is_child(stack[-2:]) == 'left':
            feats = extract_feats(stack[-2:], get_buffer_head(buffer)).values()
            steps.append(list(feats) + ['Reduce L'])
            stack.pop(-2)
        elif _is_child(stack[-2:]) == 'right' and not _is_dependent_remained(stack[-1], buffer):
            feats = extract_feats(stack[-2:], get_buffer_head(buffer)).values()
            steps.append(list(feats) + ['Reduce R'])
            stack.pop(-1)
        else:
            feats = extract_feats(stack[-2:], buffer[0]).values()
            steps.append(list(feats) + ['Shift'])
            stack.append(buffer.pop(0))

    return steps


def _is_child(tokens: list[dict]) -> str:
    left = tokens[0]
    right = tokens[1]
    if left['head'] == right['id']:
        return 'left'
    if right['head'] == left['id']:
        return 'right'
    else:
        return 'none'


def extract_feats(
    st_tail_two: list[dict], buf_fir: list[str], feat_type: str = 'all_feat'
) -> dict:
    # raw fea†uresを返す
    st_sec = st_tail_two[0]
    st_last = st_tail_two[1]
    if feat_type == 'all_feat':
        feats = {
            'st_sec': st_sec['token'],
            'st_sec_pos': st_sec['pos'],
            'st_last': st_last['token'],
            'st_last_pos': st_last['pos'],
            'buf_fir': buf_fir['token'],
            'buf_fir_pos': buf_fir['pos'],
        }
    elif feat_type == 'all_pos':
        feats = {
            'st_sec_pos': st_sec['pos'],
            'st_last_pos': st_last['pos'],
            'buf_fir_pos': buf_fir['pos'],
        }
    elif feat_type == 'st_last_pos':
        feats = {'st_last_pos': st_last['pos']}

    return feats


def get_buffer_head(buffer: list[dict]) -> dict:
    if len(buffer) > 0:
        return buffer[0]
    else:
        return {'token': None, 'pos': None}


def _is_dependent_remained(stack_tail: dict, buffer: list[dict]) -> bool:
    for t in buffer:
        if t['head'] == stack_tail['id']:
            return True
    return False


def _encode(steps: list[list], use_cols: str) -> tuple[pd.DataFrame]:
    df = pd.DataFrame(
        steps,
        columns=[
            'st_sec',
            'st_sec_pos',
            'st_last',
            'st_last_pos',
            'buf_fir',
            'buf_fir_pos',
            'action',
        ],
    )
    if use_cols == 'st_last_pos':
        df = df[['st_last_pos', 'action']]
    elif use_cols == 'all_pos':
        df = df[['st_sec_pos', 'st_last_pos', 'buf_fir_pos', 'action']]

    X = pd.get_dummies(df.iloc[:, :-1])
    y = df.iloc[:, -1]
    return X, y
