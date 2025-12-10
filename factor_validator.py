import ast
from enum import Enum

class FactorType(Enum):
    PRICE_ABS = 1   # 绝对价格 (恒正): open, close, high, low, vwap, mean(price)
    PRICE_REL = 2   # 相对价差 (有正负): price - price, slope(price), delta(price)
    
    VOLUME_ABS = 3  # 绝对量 (恒正): volume, amount
    VOLUME_REL = 4  # 相对量差 (有正负): vol - vol, slope(vol)
    
    RATIO = 5       # 比率 (无量纲)
    TIME = 6        # 时间窗口
    UNKNOWN = 99
    ERROR = -1

class FactorValidator:
    def __init__(self):
        # 定义原子特征的类型
        self.atom_types = {
            '$open': FactorType.PRICE_ABS,
            '$high': FactorType.PRICE_ABS,
            '$low':  FactorType.PRICE_ABS,
            '$close': FactorType.PRICE_ABS,
            '$vwap': FactorType.PRICE_ABS,
            '$volume': FactorType.VOLUME_ABS,
            '$amount': FactorType.VOLUME_ABS,
            '$turnover_rate': FactorType.RATIO, # 换手率通常视为 Ratio
        }

    def validate(self, expression: str) -> bool:
        """
        验证表达式是否符合物理量纲规则。
        返回: True (合法) / False (非法)
        """
        try:
            # 1. 预处理：Qlib 的 $ 符号在 Python ast 中是非法的，替换为合法字符
            # 例如: $close -> V_close
            clean_expr = expression.replace('$', 'V_')
            
            # 2. 解析为 AST 语法树
            tree = ast.parse(clean_expr, mode='eval')
            
            # 3. 递归推断类型
            result_type = self._infer_type(tree.body)
            
            # 4. 检查是否有中间错误
            if result_type == FactorType.ERROR:
                return False
            
            # 5. 强制根节点必须是 RATIO (无量纲)
            # 这意味着公式的最终输出不能是 价格、成交量 或 时间
            if result_type != FactorType.RATIO:
                # 可选：打印日志方便调试
                # print(f"Rejected: Type is {result_type}, expected RATIO. Expr: {expression}")
                return False
            
            return True
            
        except Exception as e:
            # print(f"Validation Parse Error for expression: {e}")
            return False

    def _infer_type(self, node):
        # === 1. 基础节点 ===
        
        # 变量名 (如 V_close)
        if isinstance(node, ast.Name):
            # 还原回 Qlib 格式查表
            origin_name = node.id.replace('V_', '$')
            return self.atom_types.get(origin_name, FactorType.RATIO)
            
        # 数字 (如 5, 20)
        if isinstance(node, ast.Constant): # Python 3.8+
            if isinstance(node.value, int):
                return FactorType.TIME
            return FactorType.RATIO # 浮点数视为无量纲系数
            
        # === 2. 函数调用 (即算子) ===
        if isinstance(node, ast.Call):
            func_name = node.func.id
            args = node.args
            
            # 递归获取参数类型
            arg_types = [self._infer_type(arg) for arg in args]
            
            # 如果任何子节点已有错误，直接向上传递错误
            if FactorType.ERROR in arg_types:
                return FactorType.ERROR
            
            if func_name in ['Sub', 'Div']:
                # 将 AST 节点转回字符串进行比较
                left_str = ast.unparse(args[0])
                right_str = ast.unparse(args[1])
                
                if left_str == right_str:
                    # Sub(A, A) = 0, Div(A, A) = 1
                    # 这种恒等式没有意义，浪费算力
                    return FactorType.ERROR
                
            # --- 核心物理规则校验 ---
            
            # 规则: 加减法 (生成 REL 类型)
            if func_name in ['Add', 'Sub']:
                t1, t2 = arg_types[0], arg_types[1]
                # ABS +/- ABS = ?
                if t1 == t2:
                    if func_name == 'Sub' and t1 in [FactorType.PRICE_ABS, FactorType.VOLUME_ABS]:
                        # 绝对值相减 -> 相对值 (如 close - open)
                        return FactorType.PRICE_REL if t1 == FactorType.PRICE_ABS else FactorType.VOLUME_REL
                    return t1 # 其他保持原样 (如 Add)
                
                # ABS + REL = ABS (如 close + delta)
                if {t1, t2} == {FactorType.PRICE_ABS, FactorType.PRICE_REL}:
                    return FactorType.PRICE_ABS
                if {t1, t2} == {FactorType.VOLUME_ABS, FactorType.VOLUME_REL}:
                    return FactorType.VOLUME_ABS

                # 其他情况 (如 ABS + Vol) -> Error
                return FactorType.ERROR
                

            # 规则 B: 乘法 (Mul)
            if func_name == 'Mul':
                t1, t2 = arg_types[0], arg_types[1]
                # Price * Volume = Amount (视为 Volume)
                if {t1, t2} == {FactorType.PRICE, FactorType.VOLUME}:
                    return FactorType.VOLUME
                # Price * Ratio = Price
                if FactorType.RATIO in [t1, t2]:
                    return t1 if t2 == FactorType.RATIO else t2
                return FactorType.ERROR # Price * Price 视为非法

            # 规则: 除法 (Div) -> 产生 Ratio
            if func_name == 'Div':
                t1, t2 = arg_types[0], arg_types[1]
                # 同大类相除 (ABS/ABS, REL/REL, ABS/REL) -> Ratio
                # 简单处理：只要不跨界 (Price/Vol)，都视为 Ratio
                is_price_1 = t1 in [FactorType.PRICE_ABS, FactorType.PRICE_REL]
                is_price_2 = t2 in [FactorType.PRICE_ABS, FactorType.PRICE_REL]
                if is_price_1 and is_price_2: return FactorType.RATIO

                is_vol_1 = t1 in [FactorType.VOLUME_ABS, FactorType.VOLUME_REL]
                is_vol_2 = t2 in [FactorType.VOLUME_ABS, FactorType.VOLUME_REL]
                if is_vol_1 and is_vol_2: return FactorType.RATIO
                
                if t1 == t2: return FactorType.RATIO

                # 拦截 Ratio / ABS
                if t1 == FactorType.RATIO and t2 in [FactorType.PRICE_ABS, FactorType.VOLUME_ABS]:
                    return FactorType.ERROR
                
                # 拦截: Price / Ratio -> 依然是 Price，不是 Ratio
                # 这会拦截 Div(Delta($open), Log($open))
                if t1 in [FactorType.PRICE_ABS, FactorType.PRICE_REL] and t2 == FactorType.RATIO:
                    # 除非您允许输出 Price 类型，否则对于“根节点必须是 RATIO”的要求，这里应该返回 PRICE
                    # 从而在 validate 主函数中被根节点检查拦截
                    return t1

                return FactorType.RATIO # 默认放行其他除法

            # 规则 D: 比较 (Max, Min) 必须同类型
            if func_name in ['Max', 'Min']:
                t1, t2 = arg_types[0], arg_types[1]
                
                # 1. 左右必须同类型 (Ratio vs Ratio, Price vs Price)
                if t1 != t2:
                    return FactorType.ERROR
                
                # 2. 【关键】禁止与 TIME (整数常数) 进行比较
                #    防止出现 Min(..., 10) 这种把天数当阈值的情况
                if t1 == FactorType.TIME or t2 == FactorType.TIME:
                    return FactorType.ERROR
                
                return t1 # 返回原类型

            # 规则 E: 时序函数
            if func_name in ['Mean', 'Std', 'Ref', 'Delta', 'EMA', 'WMA', 'Slope', 'Max', 'Min']:
                # 获取窗口参数节点
                window_node = args[1] 
                
                # 检查是否为常数 1
                if isinstance(window_node, ast.Constant) and window_node.value == 1:
                    # Window=1 的操作毫无意义 (Mean, EMA, Max, Min) 或者是恒为0 (Delta, Std)
                    # 直接判死刑
                    return FactorType.ERROR
                
                return arg_types[0]

            # 规则 F: 去量纲函数 (Rank, Sign, Correlation)
            if func_name in ['Rank', 'Correlation']:
                return FactorType.RATIO
            
            # 规则: Sign (核心拦截逻辑)
            # 只允许对“相对值”或“比率”取符号，禁止对“绝对值”取符号
            if func_name == 'Sign':
                t = arg_types[0]
                if t in [FactorType.PRICE_ABS, FactorType.VOLUME_ABS]:
                    return FactorType.ERROR  # 拦截 Sign($close), Sign($volume)
                return FactorType.RATIO
            
            # 规则: 趋势函数 (Slope, Delta) -> 产出 REL 类型
            if func_name in ['Slope', 'Delta']:
                t = arg_types[0]
                if t == FactorType.PRICE_ABS: return FactorType.PRICE_REL
                if t == FactorType.VOLUME_ABS: return FactorType.VOLUME_REL
                return t # Ratio 或 REL 保持不变
            
            # 规则 H: 逻辑判断 (Gt, Lt) -> RATIO (0/1 信号)
            if func_name in ['Gt', 'Lt']:
                t1, t2 = arg_types[0], arg_types[1]
                if t1 != t2:
                    return FactorType.ERROR
                return FactorType.RATIO
            
            # 规则: Sign(X)
            if func_name == 'Sign':
                arg_type = arg_types[0]
                # 拦截: 对“绝对价格”取符号 (因为绝对价格永远为正，Sign后全是1，无意义)
                if arg_type == FactorType.PRICE_ABS:
                    return FactorType.ERROR
                # 允许: 对“价差”或“比率”取符号 (有正有负，有意义)
                if arg_type in [FactorType.PRICE_REL, FactorType.RATIO]:
                    return FactorType.RATIO
                
            # 规则: Div(A, B)
            if func_name == 'Div':
                t1, t2 = arg_types[0], arg_types[1]
                
                # ... (原有的同类型相除逻辑) ...

                # 拦截: 绝对价格 / 相对价差 = 时间 (非 Ratio)
                # 例如: $high / Slope(...) 
                

            # 默认
            return FactorType.UNKNOWN

        return FactorType.UNKNOWN

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
    expr3 = "Add($close, $open)"
    print(f"Check {expr3}: {v.validate(expr3)}") # 应为 True