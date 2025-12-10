import ast
from enum import Enum

class FactorType(Enum):
    PRICE = 1   # 价格类 (open, close, vwap...)
    VOLUME = 2  # 量类 (volume, amount)
    RATIO = 3   # 比率/无量纲 (1.0, Rank输出, Price/Price)
    TIME = 4    # 时间窗口 (整数 5, 10, 20)
    UNKNOWN = 99
    ERROR = -1  # 类型冲突

class FactorValidator:
    def __init__(self):
        # 定义原子特征的类型
        self.atom_types = {
            '$open': FactorType.PRICE,
            '$high': FactorType.PRICE,
            '$low': FactorType.PRICE,
            '$close': FactorType.PRICE,
            '$vwap': FactorType.PRICE,
            '$volume': FactorType.VOLUME,
            '$amount': FactorType.VOLUME, # 金额视为量的一种，或者单独一类，这里简化合并
            '$turnover_rate': FactorType.RATIO,
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
            print(f"Validation Parse Error: {e}")
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
                
            # --- 核心物理规则校验 ---
            
            # 规则 A: 加减法 (Add, Sub) 必须同类型
            if func_name in ['Add', 'Sub']:
                t1, t2 = arg_types[0], arg_types[1]
                if t1 == t2:
                    return t1 # Price + Price = Price
                # 允许 Price + Ratio (例如 close + 1.0 这种平移虽然物理上怪，但在量化里常见)
                # 但严格禁止 Price + Volume
                if {t1, t2} == {FactorType.PRICE, FactorType.VOLUME}:
                    return FactorType.ERROR 
                return t1 # 宽松处理其他情况

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

            # 规则 C: 除法 (Div)
            if func_name == 'Div':
                t1, t2 = arg_types[0], arg_types[1]
                # 同类型相除 = Ratio (Price/Price, Vol/Vol)
                if t1 == t2:
                    return FactorType.RATIO
                # Vol / Price (Shares) -> 允许
                return FactorType.RATIO 

            # 规则 D: 比较 (Max, Min) 必须同类型
            if func_name in ['Max', 'Min']:
                t1, t2 = arg_types[0], arg_types[1]
                if t1 != t2:
                    # 特例：Max($close, 5) 第二个参数是 Time/Int，这是非法语法，应该是 Max($close, 5) 这里的5代表窗口
                    # 但 Qlib 的 Max 是 Max(Field, Window)，所以这里要看参数定义
                    # 如果是 Max(Field, Field) 比较，必须同类型
                    # 您的 builder 里 Max 是 UNARY_WIN，即 Max(Data, Int)
                    if arg_types[1] == FactorType.TIME:
                        return arg_types[0] # Max($close, 20) -> Price
                    else:
                        # 如果是两个序列比较 Max($close, $open)
                        return t1 if t1 == t2 else FactorType.ERROR
                return t1

            # 规则 E: 时序函数 (Mean, Std, Ref, Delta...)
            if func_name in ['Mean', 'Std', 'Ref', 'Delta', 'EMA', 'WMA']:
                # 保持输入类型: Mean($close) -> Price
                return arg_types[0]

            # 规则 F: 去量纲函数 (Rank, Sign, Correlation)
            if func_name in ['Rank', 'Sign', 'Correlation']:
                return FactorType.RATIO
            
            # 规则 G: 斜率 (Slope)
            if func_name == 'Slope':
                # Slope(Price) = Price / Time = Price (近似，虽有量纲变化但在股票里仍视为有单位)
                # 只有 Slope(Ratio) 才是 Ratio
                return arg_types[0]
            
            # 规则 H: 逻辑判断 (Gt, Lt) -> RATIO (0/1 信号)
            if func_name in ['Gt', 'Lt']:
                t1, t2 = arg_types[0], arg_types[1]
                if t1 != t2:
                    return FactorType.ERROR
                return FactorType.RATIO

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