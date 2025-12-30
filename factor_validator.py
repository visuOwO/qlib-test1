import ast
from enum import Enum

class FactorType(Enum):
    PRICE_ABS = 1   # 绝对价格 (恒正): open, close, high, low, vwap, mean(price)
    PRICE_REL = 2   # 相对价差 (有正负): price - price, slope(price), delta(price)
    
    VOLUME_ABS = 3  # 绝对量 (恒正): volume, amount
    VOLUME_REL = 4  # 相对量差 (有正负): vol - vol, slope(vol)
    
    # === [拆分 RATIO] ===
    RATIO_PCT = 5   # 小数值 (Percent): turnover_rate, dv_ttm (0.0x ~ 1.0)
    RATIO_MUL = 6   # 倍数值 (Multiplier): pe, pb, volume_ratio (1.0 ~ 100.0)

    TIME = 7        # 时间窗口
    UNKNOWN = 99
    ERROR = -1

class FactorValidator:
    def __init__(self, max_feature_repeat=None):
        # 定义同一特征的最大出现次数，None 表示不限制
        self.max_feature_repeat = max_feature_repeat
        self.feature_counts = {}  # 特征出现次数计数

        # 定义原子特征的类型
        self.atom_types = {
            '$open': FactorType.PRICE_ABS,
            '$high': FactorType.PRICE_ABS,
            '$low':  FactorType.PRICE_ABS,
            '$close': FactorType.PRICE_ABS,
            '$vwap': FactorType.PRICE_ABS,
            '$volume': FactorType.VOLUME_ABS,
            '$amount': FactorType.VOLUME_ABS,

            '$turnover_rate':   FactorType.RATIO_PCT,
            '$turnover_rate_f': FactorType.RATIO_PCT,
            '$dv_ttm':          FactorType.RATIO_PCT,

            '$pe_ttm':       FactorType.RATIO_MUL,
            '$pb':           FactorType.RATIO_MUL,
            '$ps_ttm':       FactorType.RATIO_MUL,
            '$volume_ratio': FactorType.RATIO_MUL,
            
            # 新增：市值类 (视为 VOLUME_ABS，因为是巨大的金额)
            '$total_mv': FactorType.VOLUME_ABS,
            '$circ_mv':  FactorType.VOLUME_ABS,
        }
        # 细分无量纲比率的语义，防止“股息率-换手率”这类跨语义相减
        self.ratio_semantics = {
            '$turnover_rate': 'liquidity',
            '$turnover_rate_f': 'liquidity',
            '$dv_ttm': 'dividend',
        }

    def validate(self, expression: str) -> bool:
        """
        验证表达式是否符合物理量纲规则。
        返回: True (合法) / False (非法)
        """
        try:
            # 1. 重置特征计数器
            self.feature_counts = {}

            # 2. 预处理：Qlib 的 $ 符号在 Python ast 中是非法的，替换为合法字符
            # 例如: $close -> V_close
            clean_expr = expression.replace('$', 'V_')

            # 3. 解析为 AST 语法树
            tree = ast.parse(clean_expr, mode='eval')

            # 4. 拒绝顶层就是 Sign(...) 的二值因子（信息量过低）
            if isinstance(tree.body, ast.Call) and getattr(tree.body.func, "id", None) == "Sign":
                return False

            # 5. 递归推断类型
            result_type = self._infer_type(tree.body)

            # 6. 检查是否有中间错误
            if result_type == FactorType.ERROR:
                return False

            # 7. 强制根节点必须是 RATIO (无量纲)
            # 这意味着公式的最终输出不能是 价格、成交量 或 时间
            if (result_type != FactorType.RATIO_MUL) and (result_type != FactorType.RATIO_PCT):
                # 可选：打印日志方便调试
                # print(f"Rejected: Type is {result_type}, expected RATIO. Expr: {expression}")
                return False

            return True

        except Exception as e:
            # print(f"Validation Parse Error for expression: {e}")
            return False

    def _infer_type(self, node):
        # === 1. 基础节点 ===
        if isinstance(node, ast.Name):
            origin_name = node.id.replace('V_', '$')

            # 检查特征出现次数
            if self.max_feature_repeat is not None:
                self.feature_counts[origin_name] = self.feature_counts.get(origin_name, 0) + 1
                if self.feature_counts[origin_name] > self.max_feature_repeat:
                    return FactorType.ERROR

            return self.atom_types.get(origin_name, FactorType.UNKNOWN) # 建议默认为 UNKNOWN 或在 init 里定义全
            
        if isinstance(node, ast.Constant):
            if isinstance(node.value, int):
                return FactorType.TIME
            return FactorType.RATIO_PCT # 浮点常数视为无量纲系数(小数)
        
        # 辅助函数：判断是否为某种 Ratio
        def is_ratio(t):
            return t in [FactorType.RATIO_PCT, FactorType.RATIO_MUL]

        # === 2. 函数调用 ===
        if isinstance(node, ast.Call):
            func_name = node.func.id
            args = node.args
            
            # 递归获取参数类型
            arg_types = [self._infer_type(arg) for arg in args]
            
            # 错误传递
            if FactorType.ERROR in arg_types:
                return FactorType.ERROR

            # --- 全局结构检查 ---
            
            # 1. 禁止自指运算 (Sub(A, A), Div(A, A))
            if func_name in ['Sub', 'Div']:
                try:
                    # ast.unparse 需要 Python 3.9+
                    left_str = ast.unparse(args[0])
                    right_str = ast.unparse(args[1])
                    if left_str == right_str:
                        return FactorType.ERROR
                except:
                    pass # 低版本 Python 跳过此检查

            # --- 2. 窗口/参数检查 (分情况讨论) ---
            
            # 组 A: 聚合类函数 (Window=1 是废话)
            # Mean(x, 1) == x, Max(x, 1) == x
            if func_name in ['Mean', 'EMA', 'WMA', 'Max', 'Min', 'Sum', 'Ts_Rank']:
                if len(args) > 1 and isinstance(args[1], ast.Constant) and args[1].value == 1:
                    return FactorType.ERROR

            # 组 B: 统计类函数 (Window < 2 无法计算)
            # Std(x, 1) == 0, Slope(x, 1) 无意义
            if func_name in ['Std', 'Var', 'Skew', 'Kurt', 'Slope', 'Correlation']:
                if len(args) > 1 and isinstance(args[1], ast.Constant) and args[1].value < 2:
                    return FactorType.ERROR

            # 组 C: 位移/差分类函数 (Window=1 是合法的！)
            # Ref(x, 1) 是昨日值，Delta(x, 1) 是日收益
            # 这类函数不需要检查 value == 1，只需要检查 value > 0 即可 (通常默认都是正整数)
            if func_name in ['Ref', 'Delta']:
                if len(args) > 1 and isinstance(args[1], ast.Constant) and args[1].value < 1:
                    return FactorType.ERROR
                # 排除 Delta(Sign(x), n) 这类无信息的“方向跳变”因子
                if func_name == 'Delta' and isinstance(args[0], ast.Call) and getattr(args[0].func, 'id', None) == 'Sign':
                    return FactorType.ERROR

                # --- 核心算子逻辑 ---
            
            # A. 加减法 (生成 REL 或 保持类型)
            if func_name in ['Add', 'Sub']:
                t1, t2 = arg_types[0], arg_types[1]

                # 严格禁止 PCT(小数) 与 MUL(大数) 相加减
                if {t1, t2} == {FactorType.RATIO_PCT, FactorType.RATIO_MUL}:
                    return FactorType.ERROR

                # 禁止价格/成交量类与任意 Ratio 混加减（防止 $turnover_rate - $close 这类跨量纲）
                non_ratio_types = {FactorType.PRICE_ABS, FactorType.PRICE_REL, FactorType.VOLUME_ABS, FactorType.VOLUME_REL}
                if (is_ratio(t1) and t2 in non_ratio_types) or (is_ratio(t2) and t1 in non_ratio_types):
                    return FactorType.ERROR

                # 无量纲比例之间的相加减需要语义一致，否则拒绝
                if t1 == t2 == FactorType.RATIO_PCT:
                    s1 = self._get_ratio_semantic(args[0])
                    s2 = self._get_ratio_semantic(args[1])
                    # 两个原子比率语义不同，直接判为非法
                    if s1 and s2 and s1 != s2:
                        return FactorType.ERROR

                # 同类型处理
                if t1 == t2:
                    if func_name == 'Sub' and t1 in [FactorType.PRICE_ABS, FactorType.VOLUME_ABS]:
                        # 绝对值相减 -> 相对值
                        return FactorType.PRICE_REL if t1 == FactorType.PRICE_ABS else FactorType.VOLUME_REL
                    return t1 # Add 保持原样, Sub(Rel, Rel) 保持 Rel
                
                # 混合类型: ABS + REL -> ABS
                if {t1, t2} == {FactorType.PRICE_ABS, FactorType.PRICE_REL}:
                    return FactorType.PRICE_ABS
                if {t1, t2} == {FactorType.VOLUME_ABS, FactorType.VOLUME_REL}:
                    return FactorType.VOLUME_ABS
                
                # 允许 Price +/- Ratio (平移)
                if (t1 == FactorType.PRICE_ABS and is_ratio(t2)) or (t2 == FactorType.PRICE_ABS and is_ratio(t1)):
                    return FactorType.PRICE_ABS

                return FactorType.ERROR

            # B. 乘法 (Mul)
            if func_name == 'Mul':
                t1, t2 = arg_types[0], arg_types[1]

                # Ratio * Ratio -> MUL (倾向于变大)
                if is_ratio(t1) and is_ratio(t2):
                    return FactorType.RATIO_MUL

                # Price * Volume -> Volume (Amount)
                if {t1, t2} == {FactorType.PRICE_ABS, FactorType.VOLUME_ABS}:
                    return FactorType.VOLUME_ABS
                
                # Any * Ratio -> Keep Type (Scaling)
                if is_ratio(t1): return t2
                if is_ratio(t2): return t1
                
                # 禁止 Price * Price, Vol * Vol
                return FactorType.ERROR

            # C. 除法 (Div)
            if func_name == 'Div':
                t1, t2 = arg_types[0], arg_types[1]

                # 1. 拦截: Ratio / Price 或 Ratio / Volume (小/大 = 0)
                if is_ratio(t1) and t2 in [FactorType.PRICE_ABS, FactorType.VOLUME_ABS, FactorType.PRICE_REL, FactorType.VOLUME_REL]:
                    return FactorType.ERROR
                
                # 2. 拦截: 绝对价格 / 相对价差 = 时间 (无意义)
                # 例如 Div($high, Slope(...))
                if t1 == FactorType.PRICE_ABS and t2 == FactorType.PRICE_REL:
                    return FactorType.ERROR

                # 3. Ratio / Ratio -> PCT (倾向于变小)
                if is_ratio(t1) and is_ratio(t2):
                    return FactorType.RATIO_PCT

                # 4. 同大类相除 -> Ratio_MUL (如 PE = Price/EPS)
                # Price / Price
                if t1 in [FactorType.PRICE_ABS, FactorType.PRICE_REL] and t2 in [FactorType.PRICE_ABS, FactorType.PRICE_REL]:
                    return FactorType.RATIO_MUL
                # Vol / Vol
                if t1 in [FactorType.VOLUME_ABS, FactorType.VOLUME_REL] and t2 in [FactorType.VOLUME_ABS, FactorType.VOLUME_REL]:
                    return FactorType.RATIO_MUL
                
                # 5. Vol / Price -> Vol (Shares)
                if t1 in [FactorType.VOLUME_ABS] and t2 in [FactorType.PRICE_ABS]:
                    return FactorType.VOLUME_ABS

                # 6. Any / Ratio -> Keep Type (Scaling)
                if is_ratio(t2):
                    return t1

                return FactorType.ERROR
            # D. 比较 (Max, Min) - 禁止与时间比较
            if func_name in ['Max', 'Min', 'Gt', 'Lt']:
                t1, t2 = arg_types[0], arg_types[1]
                if t1 == FactorType.TIME or t2 == FactorType.TIME:
                    return FactorType.ERROR
                if t1 != t2:
                    return FactorType.ERROR
                # Gt/Lt 返回 Ratio (0/1), Max/Min 返回原类型
                return FactorType.RATIO_PCT if func_name in ['Gt', 'Lt'] else t1

            # 规则 E: 时序函数与窗口检查 (分拆逻辑)
            
            # 1. 聚合类函数: Window=1 是废话 (等于自身)
            if func_name in ['Mean', 'EMA', 'WMA', 'Max', 'Min', 'Sum', 'Rank', 'Ts_Rank']:
                window_node = args[1]
                if isinstance(window_node, ast.Constant) and window_node.value == 1:
                    return FactorType.ERROR
                # Rank($close, 10) -> RATIO (百分比)
                if func_name in ['Rank', 'Ts_Rank']:
                    return FactorType.RATIO_PCT
                return arg_types[0] # 保持输入类型

            # 2. 统计类函数: Window < 2 无法计算 (方差/斜率至少要2个点)
            if func_name in ['Std', 'Var', 'Skew', 'Kurt', 'Slope', 'Correlation']:
                window_node = args[1]
                if isinstance(window_node, ast.Constant) and window_node.value < 2:
                    return FactorType.ERROR
                # Std/Slope 会改变类型 (ABS -> REL)，这里已经在前面处理过了，
                # 如果代码逻辑走到这里没返回，说明前面漏了，这里做一个兜底：
                t = arg_types[0]
                if t == FactorType.PRICE_ABS: return FactorType.PRICE_REL
                if t == FactorType.VOLUME_ABS: return FactorType.VOLUME_REL
                return t

            # 3. 位移/差分类函数: Window=1 是完全合法的！
            # Ref(x, 1) = 昨天, Delta(x, 1) = 今天-昨天
            if func_name in ['Ref', 'Delta']:
                # 这里不需要检查 value == 1
                if func_name == 'Delta':
                    # Delta 会把绝对值变成相对值
                    t = arg_types[0]
                    if t == FactorType.PRICE_ABS: return FactorType.PRICE_REL
                    if t == FactorType.VOLUME_ABS: return FactorType.VOLUME_REL
                    return t
                
                # Ref 保持原类型
                return arg_types[0]
            
            # F. 变异函数 (Std, Delta, Slope)
            if func_name in ['Std', 'Delta', 'Slope']:
                t = arg_types[0]
                # 绝对值变异后成为相对值
                if t == FactorType.PRICE_ABS: return FactorType.PRICE_REL
                if t == FactorType.VOLUME_ABS: return FactorType.VOLUME_REL
                return t

            # G. 去量纲函数
            if func_name in ['Correlation', 'Sign']:
                # Sign 拦截绝对值
                if func_name == 'Sign' and arg_types[0] in [FactorType.PRICE_ABS, FactorType.VOLUME_ABS]:
                    return FactorType.ERROR
                return FactorType.RATIO_PCT

            if func_name == 'Abs':
                arg_node = args[0] if args else None
                # 拦截嵌套 Abs，避免 Abs(Abs(x)) 这类冗余表达式
                if isinstance(arg_node, ast.Call) and getattr(arg_node.func, "id", "") == "Abs":
                    return FactorType.ERROR
                t = arg_types[0]
                if t in [FactorType.UNKNOWN, FactorType.ERROR]:
                    return FactorType.ERROR
                return t
            
            
            if func_name == 'Log':
                arg_node = args[0] if args else None
                # 拦截嵌套 Log，避免 Log(Log(x)) 这类极易产生 -inf 的表达式
                if isinstance(arg_node, ast.Call) and getattr(arg_node.func, "id", "") == "Log":
                    return FactorType.ERROR
                # 常数或表达式若无法保证恒正，则不允许取 Log
                if not self._is_definitely_positive(arg_node):
                    return FactorType.ERROR
                t = arg_types[0]
                if t == FactorType.RATIO_MUL: return FactorType.RATIO_MUL
                if t == FactorType.PRICE_ABS: return FactorType.RATIO_MUL # Log(Price) 视为无量纲数值
                if t == FactorType.VOLUME_ABS: return FactorType.RATIO_MUL
                if t == FactorType.RATIO_PCT: return FactorType.RATIO_MUL
                return FactorType.ERROR

            return FactorType.UNKNOWN

        return FactorType.UNKNOWN

    def _get_ratio_semantic(self, node):
        """
        仅在原子比率之间做相加减时使用，用于拦截跨语义的组合。
        """
        if isinstance(node, ast.Name):
            origin_name = node.id.replace('V_', '$')
            return self.ratio_semantics.get(origin_name)
        return None

    def _is_definitely_positive(self, node) -> bool:
        """
        粗略判定表达式是否恒正（>0），用于保护 Log 的输入。
        保守策略：不确定即判为 False。
        """
        # 原子：特征
        if isinstance(node, ast.Name):
            origin = node.id.replace('V_', '$')
            t = self.atom_types.get(origin, FactorType.UNKNOWN)
            return t in [FactorType.PRICE_ABS, FactorType.VOLUME_ABS, FactorType.RATIO_PCT, FactorType.RATIO_MUL]

        # 原子：常数
        if isinstance(node, ast.Constant):
            try:
                return float(node.value) > 0
            except Exception:
                return False

        # 调用表达式
        if isinstance(node, ast.Call):
            func = getattr(node.func, "id", None)
            args = node.args

            if func in ['Ref', 'Mean', 'EMA', 'WMA', 'Sum']:
                return len(args) >= 1 and self._is_definitely_positive(args[0])
            if func in ['Max', 'Min']:
                return len(args) >= 2 and self._is_definitely_positive(args[0]) and self._is_definitely_positive(args[1])
            if func in ['Mul', 'Div']:
                return len(args) >= 2 and self._is_definitely_positive(args[0]) and self._is_definitely_positive(args[1])
            if func == 'Add':
                return len(args) >= 2 and self._is_definitely_positive(args[0]) and self._is_definitely_positive(args[1])
            # Abs / Rank / Gt / Lt / Sign / Delta / Slope / Std 等均不保证恒正
            return False

        return False

# 简单测试
if __name__ == "__main__":
    v = FactorValidator()
    # 案例 1: 错误 (价+量)
    expr1 = "Sub($high, Mul(Max($open, 20), Add($volume, $open)))"
    print(f"Check {expr1}: {v.validate(expr1)}") # 应为 False

    expr1_5 = "Add(Add($open, $close), $volume)"
    print(f"Check {expr1_5}: {v.validate(expr1_5)}") # 应为 True

    # 案例 2: 正确 (价/价)
    expr2 = "Div($close, Ref($close, 1))"
    print(f"Check {expr2}: {v.validate(expr2)}") # 应为 True

    # 案例 3: 正确 (价+价)
    expr3 = "Rank($pe, 2)"
    print(f"Check {expr3}: {v.validate(expr3)}") # 应为 True

    expr4 = "$ps_ttm"
    print(f"Check {expr4}: {v.validate(expr4)}") # 应为 True

    expr5 = "Ref(Log(Sub($turnover_rate, $close)), 5)"
    print(f"Check {expr5}: {v.validate(expr5)}") # 应为 False，跨语义比率相减应被拦截

    # 案例 6: 嵌套 Abs 测试 (重复计算，应为非法)
    expr6 = "Abs(Abs($close))"
    print(f"Check {expr6}: {v.validate(expr6)}") # 应为 False

    # 案例 7: 正常 Abs 测试 (应为合法)
    expr7 = "Abs(Sub($pb, $pe_ttm))"
    print(f"Check {expr7}: {v.validate(expr7)}") # 应为 True

    # 案例 8: 三重嵌套 Abs 测试 (应为非法)
    expr8 = "Abs(Abs(Abs($close)))"
    print(f"Check {expr8}: {v.validate(expr8)}") # 应为 False

    # 案例 9: 嵌套 Log 测试 (应为非法)
    expr9 = "Log(Log($close))"
    print(f"Check {expr9}: {v.validate(expr9)}") # 应为 False

    # === 特征重复次数测试 ===
    print("\n=== 特征重复次数测试 (max_feature_repeat=2) ===")
    v2 = FactorValidator(max_feature_repeat=2)

    # 同一特征出现 2 次 (合法)
    expr10 = "Add($dv_ttm, $dv_ttm)"
    print(f"Check {expr10}: {v2.validate(expr10)}") # 应为 True

    # 同一特征出现 3 次 (非法)
    expr11 = "Add(Add($dv_ttm, $dv_ttm), $dv_ttm)"
    print(f"Check {expr11}: {v2.validate(expr11)}") # 应为 False

    # 不同特征各出现 1 次 (合法)
    expr12 = "Mul($dv_ttm, Std(Mul($dv_ttm, $dv_ttm), 20))"
    print(f"Check {expr12}: {v2.validate(expr12)}") # 应为 True

    # Ref($close, 1) + $close (出现 2 次，合法)
    expr13 = "Add($dv_ttm, Ref($dv_ttm, 1))"
    print(f"Check {expr13}: {v2.validate(expr13)}") # 应为 True

    print("\n=== 无限制测试 (max_feature_repeat=None) ===")
    v3 = FactorValidator()  # 默认不限制

    expr14 = "Add(Add(Add($dv_ttm, $dv_ttm), $dv_ttm), $dv_ttm)"
    print(f"Check {expr14}: {v3.validate(expr14)}") # 应为 True
