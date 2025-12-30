import numpy as np

class FactorBuilder:
    """
    Builds factor expressions using Reverse Polish Notation (RPN).
    Each action selects a token (feature/int/operator), and RPN is
    converted back into a Qlib expression for validation/evaluation.
    """
    STACK_EXPR = "expr"
    STACK_INT = "int"

    def __init__(self, max_seq_len=20, features=None):
        self.features = features or []
        
        # --- Expanded Operator Set ---
        self.ops_binary = ["Add", "Sub", "Mul", "Div"]
        
        # Unary Ops needing a Window (Int)
        # Added: Slope, EMA, WMA, ROC (Rate of Change), Rank
        self.ops_unary_win = ["Mean", "Std", "Max", "Min", "Slope", "EMA", "WMA", "Rank", "Ts_Rank"] 
        
        # Unary Ops needing an Offset (Int)
        # Added: Delta (Temp Removed)
        self.ops_unary_off = ["Ref"] 
        
        # Unary Ops needing NO extra parameter (Simple)
        # Added: Abs, Log, Sign(Temp removed)
        self.ops_unary_simple = ["Abs", "Log"]

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
        self.action_map.append(("END", "END"))

        self.max_seq_len = max_seq_len # 定义一个最大序列长度
        self.action_history = [] # 用于记录历史动作
        
        self.reset()

    def reset(self):
        self.stack = []  # Stack of (type, depth) for RPN parsing
        self.rpn_tokens = [] # RPN token sequence
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
        if self.done:
            return []
        valid_indices = []
        remaining = self.max_seq_len - len(self.rpn_tokens) - 1
        if remaining < 0:
            return []
        expr_count, int_count = self._count_stack()
        stack_top = self.stack[-1][0] if self.stack else None

        # 获取上一个算子（忽略 INT 和 FEATURE），用于检查重复连续调用
        last_op = None
        for token in reversed(self.rpn_tokens):
            if isinstance(token, str) and token not in ["END"] and not token.startswith('$'):
                last_op = token
                break

        for idx, (val, kind) in enumerate(self.action_map):
            if kind == "FEATURE":
                if stack_top == self.STACK_INT:
                    continue
                if self._can_finish_after_action(expr_count + 1, int_count, remaining):
                    valid_indices.append(idx)
            elif kind == "INT":
                # Int tokens are only useful after an expr exists (for Ref/Mean/etc.)
                if stack_top == self.STACK_EXPR:
                    if self._can_finish_after_action(expr_count, int_count + 1, remaining):
                        valid_indices.append(idx)
            elif kind == "UNARY_SIMPLE":
                if stack_top == self.STACK_INT:
                    continue
                # 阻止连续调用相同的算子 (如 Abs, Abs)
                if last_op == val:
                    continue
                if self._can_apply_unary():
                    if self._can_finish_after_action(expr_count, int_count, remaining):
                        valid_indices.append(idx)
            elif kind == "UNARY_WIN" or kind == "UNARY_OFF":
                # 阻止连续调用相同的算子 (如 Mean, Mean)
                if last_op == val:
                    continue
                if self._can_apply_unary_with_int():
                    if self._can_finish_after_action(expr_count, int_count - 1, remaining):
                        valid_indices.append(idx)
            elif kind == "BINARY":
                if stack_top == self.STACK_INT:
                    continue
                # 阻止连续调用相同的算子 (如 Add, Add)
                if last_op == val:
                    continue
                if self._can_apply_binary():
                    if self._can_finish_after_action(expr_count - 1, int_count, remaining):
                        valid_indices.append(idx)
            elif kind == "END":
                if self._can_end():
                    valid_indices.append(idx)

        return valid_indices

    def step(self, action_idx):
        if self.done:
            raise Exception("Episode is done")
        
        # 记录动作
        self.action_history.append(action_idx)

        val, kind = self.action_map[action_idx]
        
        # Store token
        self.rpn_tokens.append(val)

        # Apply RPN stack transition
        if kind == "FEATURE":
            self.stack.append((self.STACK_EXPR, 1))
        elif kind == "INT":
            self.stack.append((self.STACK_INT, 1))
        elif kind == "UNARY_SIMPLE":
            self._apply_unary()
        elif kind == "UNARY_WIN" or kind == "UNARY_OFF":
            self._apply_unary_with_int()
        elif kind == "BINARY":
            self._apply_binary()
        elif kind == "END":
            self.done = True

        if len(self.rpn_tokens) >= self.max_seq_len:
            self.done = True
        
        return self.get_state(), self.done

    def build_expression(self):
        """
        Converts RPN tokens into a Qlib expression.
        """
        if not self.rpn_tokens:
            return ""

        stack = []
        for token in self.rpn_tokens:
            if token == "END":
                break
            if isinstance(token, str) and token in self.ops_binary:
                if len(stack) < 2:
                    return "Error"
                right = stack.pop()
                left = stack.pop()
                if token in ["Add", "Mul"] and left > right:
                    left, right = right, left
                stack.append(f"{token}({left}, {right})")
            elif isinstance(token, str) and token in self.ops_unary_win + self.ops_unary_off:
                if len(stack) < 2:
                    return "Error"
                param = stack.pop()
                arg = stack.pop()
                stack.append(f"{token}({arg}, {param})")
            elif isinstance(token, str) and token in self.ops_unary_simple:
                if len(stack) < 1:
                    return "Error"
                arg = stack.pop()
                stack.append(f"{token}({arg})")
            elif isinstance(token, str):
                # Feature
                stack.append(token)
            else:
                # Int
                stack.append(str(token))

        if len(stack) != 1:
            return "Error"
        return stack[0]

    def _can_apply_unary(self):
        if len(self.stack) < 1:
            return False
        t, _ = self.stack[-1]
        return t == self.STACK_EXPR

    def _can_apply_unary_with_int(self):
        if len(self.stack) < 2:
            return False
        t_int, _ = self.stack[-1]
        t_expr, _ = self.stack[-2]
        return t_expr == self.STACK_EXPR and t_int == self.STACK_INT

    def _can_apply_binary(self):
        if len(self.stack) < 2:
            return False
        t2, _ = self.stack[-1]
        t1, _ = self.stack[-2]
        return t1 == self.STACK_EXPR and t2 == self.STACK_EXPR

    def _can_end(self):
        return len(self.stack) == 1 and self.stack[0][0] == self.STACK_EXPR

    def _apply_unary(self):
        t, depth = self.stack.pop()
        new_depth = depth + 1
        self.stack.append((self.STACK_EXPR, new_depth))

    def _apply_unary_with_int(self):
        t_int, depth_int = self.stack.pop()
        t_expr, depth_expr = self.stack.pop()
        new_depth = max(depth_expr, depth_int) + 1
        self.stack.append((self.STACK_EXPR, new_depth))

    def _apply_binary(self):
        t2, depth2 = self.stack.pop()
        t1, depth1 = self.stack.pop()
        new_depth = max(depth1, depth2) + 1
        self.stack.append((self.STACK_EXPR, new_depth))

    def _count_stack(self):
        expr_count = 0
        int_count = 0
        for t, _ in self.stack:
            if t == self.STACK_EXPR:
                expr_count += 1
            elif t == self.STACK_INT:
                int_count += 1
        return expr_count, int_count

    def _min_steps_to_reduce(self, expr_count, int_count):
        if expr_count <= 0:
            return float("inf")
        return int_count + max(0, expr_count - 1)

    def _can_finish_after_action(self, expr_count, int_count, remaining_steps):
        if expr_count <= 0 or int_count < 0:
            return False
        min_steps = self._min_steps_to_reduce(expr_count, int_count)
        return min_steps <= remaining_steps
