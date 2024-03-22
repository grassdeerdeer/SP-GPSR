import regex
from sympy import *


def convert_power_sin(string):
    # 如果字符串中包含"**"，则进行转换
    pattern = r"sin\((?:[^()]+|(?R))*\)\*{2}\d+"
    while "**" in string:
        # 用正则表达式匹配所有的幂运算，得到一个匹配对象的列表
        matches = regex.finditer(pattern,string)
        # 创建一个空列表，用于存放转换后的幂运算
        new_powers = []
        # 遍历匹配对象的列表
        for match in matches:
            # 获取匹配的字符串
            power = match.group()
            # 用"**"分割字符串，得到底数和指数
            base, exponent = power.rsplit("**",1)
            # 将指数转换为整数
            exponent = int(exponent)
            # 用"*"重复底数指数次，得到新的幂运算
            new_power = "*".join([base] * exponent)
            # 将新的幂运算添加到列表中
            new_powers.append(new_power)
        # 用正则表达式替换所有的幂运算为新的幂运算，得到新的字符串
        string = regex.sub(pattern, lambda m: new_powers.pop(0), string)
        # 返回新的字符串

    return string

s = "sin(0.650969002738136*x1*x1)"
print(convert_power_sin(s))
# pattern = r"sin\((?:[^()]+|(?R))*\)\*{2}\d+"
