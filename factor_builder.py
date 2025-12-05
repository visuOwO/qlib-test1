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
        # Added: Slope, EMA, WMA, ROC (Rate of Change)
        self.ops_unary_win = ["Mean", "Std", "Max", "Min", "Slope", "EMA", "WMA", "ROC"] 
        
        # Unary Ops needing an Offset (Int)
        # Added: Delta
        self.ops_unary_off = ["Ref", "Delta"] 
        
        self.windows = [5, 10, 20, 60]
        self.offsets = [1, 5, 10]
        
        # Flatten Action Space
        self.action_map = []
        self.action_map.extend([(op, 'BINARY') for op in self.ops_binary])
        self.action_map.extend([(op, 'UNARY_WIN') for op in self.ops_unary_win])
        self.action_map.extend([(op, 'UNARY_OFF') for op in self.ops_unary_off])
        self.action_map.extend([(f, 'FEATURE') for f in self.features])
        self.action_map.extend([(i, 'INT') for i in set(self.windows + self.offsets)])
        
        self.reset()

    def reset(self):
        # Stack contains: (required_type, current_depth)
        self.stack = [(self.TYPE_OP, 0)] 
        self.expression_parts = [] # To store the constructed tree structure
        self.done = False
        return self.get_state()

    def get_state(self):
        """
        Returns a vector representation of the current state.
        Vector Size: 3 (Depth, Type, IsStackEmpty)
        """
        if not self.stack:
            return np.array([0, -1, 1], dtype=np.float32)
        
        req_type, depth = self.stack[-1]
        # Normalize depth
        return np.array([depth / self.max_depth, req_type, 0], dtype=np.float32)

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
                    if kind in ['BINARY', 'UNARY_WIN', 'UNARY_OFF', 'FEATURE']:
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
                    left = _recurse()
                    right = _recurse()
                    return f"{token}({left}, {right})"
                elif token in self.ops_unary_win + self.ops_unary_off:
                    arg = _recurse()
                    param = _recurse() # The INT
                    return f"{token}({arg}, {param})"
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
