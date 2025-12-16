import numpy as np

class FactorBuilder:
    """
    Manages the construction of a factor expression tree.
    State: [CurrentDepth, SlotType(OneHot), LastActionEmbedding...]
    """
    TYPE_OP = 0
    TYPE_FEATURE = 1
    TYPE_INT = 2

    def __init__(self, max_depth=3, features=None):
        self.max_depth = max_depth
        self.features = features or []
        
        # --- Expanded Operator Set ---
        self.ops_binary = ["Add", "Sub", "Mul", "Div"]
        
        # Unary Ops needing a Window (Int)
        # Added: Slope, EMA, WMA, ROC (Rate of Change), Rank
        self.ops_unary_win = ["Mean", "Std", "Max", "Min", "Slope", "EMA", "WMA", "Rank"] 
        
        # Unary Ops needing an Offset (Int)
        # Added: Delta
        self.ops_unary_off = ["Ref", "Delta"] 
        
        # Unary Ops needing NO extra parameter (Simple)
        # Added: Abs, Log, Sign
        self.ops_unary_simple = ["Abs", "Log", "Sign"]

        self.windows = [5, 10, 20, 60]
        self.offsets = [1, 5, 10]
        
        # Flatten Action Space
        self.action_map = []
        self.action_map.extend([(op, 'BINARY') for op in self.ops_binary])
        self.action_map.extend([(op, 'UNARY_WIN') for op in self.ops_unary_win])
        self.action_map.extend([(op, 'UNARY_OFF') for op in self.ops_unary_off])
        self.action_map.extend([(op, 'UNARY_SIMPLE') for op in self.ops_unary_simple])
        self.action_map.extend([(f, 'FEATURE') for f in self.features])
        self.action_map.extend([(i, 'INT') for i in set(self.windows + self.offsets)])

        self.max_seq_len = 20 # 定义一个最大序列长度，比如20
        self.action_history = [] # 用于记录历史动作
        
        self.reset()

    def reset(self):
        # Stack contains: (required_type, current_depth)
        self.stack = [(self.TYPE_OP, 0)] 
        self.expression_parts = [] # To store the constructed tree structure
        self.done = False
        self.action_history = [] # 重置历史
        return self.get_state()

    def get_state(self):
        """
        返回 历史动作序列 (定长，不足补0)
        """
        # 预留 0 作为 padding，真实动作索引整体 +1，避免与 padding 冲突
        seq = [a + 1 for a in self.action_history]

        # 截断或填充到固定长度 (Padding)
        if len(seq) < self.max_seq_len:
            seq = seq + [0] * (self.max_seq_len - len(seq))
        else:
            seq = seq[-self.max_seq_len:] # 只取最后 N 步
            
        return np.array(seq, dtype=np.int64) # 必须是整数类型!

    def get_valid_actions(self):
        """
        Returns a mask of valid action indices.
        """
        if not self.stack:
            return []

        req_type, depth = self.stack[-1]
        valid_indices = []

        for idx, (val, kind) in enumerate(self.action_map):
            if req_type == self.TYPE_OP:
                # If we reached max depth, we CANNOT pick operators anymore, must pick features
                if depth >= self.max_depth:
                    if kind == 'FEATURE':
                        valid_indices.append(idx)
                else:
                    # Can pick Operators OR Features
                    if kind in ['BINARY', 'UNARY_WIN', 'UNARY_OFF', 'UNARY_SIMPLE', 'FEATURE']:
                        valid_indices.append(idx)
            
            elif req_type == self.TYPE_FEATURE:
                # Must pick feature
                if kind == 'FEATURE':
                    valid_indices.append(idx)
            
            elif req_type == self.TYPE_INT:
                # Must pick int
                if kind == 'INT':
                    valid_indices.append(idx)

        return valid_indices

    def step(self, action_idx):
        if self.done:
            raise Exception("Episode is done")
        
        # 记录动作
        self.action_history.append(action_idx)

        val, kind = self.action_map[action_idx]
        req_type, depth = self.stack.pop()
        
        # Store the choice
        self.expression_parts.append(val)

        # Push new requirements to stack based on action
        if kind == 'BINARY':
            # Needs 2 arguments. Push right then left (so left is popped first)
            self.stack.append((self.TYPE_OP, depth + 1))
            self.stack.append((self.TYPE_OP, depth + 1))
        elif kind in ['UNARY_WIN', 'UNARY_OFF']:
            # Needs 1 argument + 1 int
            self.stack.append((self.TYPE_INT, depth))
            self.stack.append((self.TYPE_OP, depth + 1))
        elif kind == 'UNARY_SIMPLE':
            # Needs 1 argument only
            self.stack.append((self.TYPE_OP, depth + 1))
        elif kind == 'FEATURE':
            # Terminal
            pass
        elif kind == 'INT':
            # Terminal
            pass

        if not self.stack:
            self.done = True
        
        return self.get_state(), self.done

    def build_expression(self):
        """
        Reconstructs the string expression from the sequence of choices (Pre-order traversal).
        [Modified] Applies Canonicalization (Normalization) for commutative operators (Add, Mul).
        """
        if not self.expression_parts: return ""
        
        iterator = iter(self.expression_parts)
        
        def _recurse():
            try:
                token = next(iterator)
            except StopIteration:
                return "Error"

            # Check type of token
            if isinstance(token, str):
                if token in self.ops_binary:
                    # Binary Op: recursively build left and right children
                    left = _recurse()
                    right = _recurse()
                    
                    # 对于满足交换律的算子 (Add, Mul)，按字符串字典序重排参数
                    # 这确保了 Add(A, B) 和 Add(B, A) 生成完全相同的字符串
                    if token in ["Add", "Mul"]:
                        if left > right: # 字符串比较：如果左边大于右边，则交换
                            left, right = right, left

                    return f"{token}({left}, {right})"
                
                elif token in self.ops_unary_win + self.ops_unary_off:
                    arg = _recurse()
                    param = _recurse() # The INT
                    return f"{token}({arg}, {param})"
                
                elif token in self.ops_unary_simple:
                    arg = _recurse()
                    return f"{token}({arg})"
                
                else:
                    # Feature
                    return token
            else:
                # Int
                return str(token)

        try:
            return _recurse()
        except:
            return "Error"
