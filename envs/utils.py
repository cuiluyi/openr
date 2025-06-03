import math
import numpy as np  # 使用numpy处理数值稳定性

def entropy_from_logprobs(log_probs, base=math.e):
    """
    从对数概率计算概率分布和熵
    
    参数:
    log_probs -- 包含对数概率的列表（取值范围 ≤0）
    base -- 对数底数（默认自然对数e）
    
    返回:
    dict -- 包含以下键的字典:
        'probability_distribution': 归一化后的概率分布列表
        'entropy': 计算得到的熵 (以比特为单位)
        'max_entropy': 最大可能熵 (基于列表长度)
        'relative_entropy': 相对熵 (熵/最大熵)
        'message': 处理过程的描述信息
    """
    n = len(log_probs)
    
    # 处理空列表
    if n == 0:
        return {
            'probability_distribution': [],
            'entropy': 0.0,
            'max_entropy': 0.0,
            'relative_entropy': 0.0,
            'message': '空列表无法计算概率分布'
        }
    
    # 转换为NumPy数组提高计算效率
    log_probs = np.array(log_probs)
    
    # 数值稳定处理：减去最大值避免指数下溢
    max_log = np.max(log_probs)
    shifted = log_probs - max_log
    
    # 计算未归一化概率和归一化常数
    unnormalized_probs = np.exp(shifted)
    total = np.sum(unnormalized_probs)
    
    # 处理全零情况（概率总和为0）
    if total == 0:
        prob_dist = np.ones(n) / n
        message = "警告：所有概率为零，使用均匀分布"
    else:
        prob_dist = unnormalized_probs / total
        message = f"成功从{base}为底的对数概率归一化"
    
    # 计算熵（使用对数概率避免数值问题）
    entropy_val = 0.0
    for i, p in enumerate(prob_dist):
        if p > 0:
            # 使用原始对数概率计算，避免浮点精度损失
            log_p = log_probs[i] - np.log(total) - max_log
            entropy_val -= p * (log_p / math.log(2))  # 转换为log2
    
    # 计算最大熵和相对熵
    max_entropy = math.log2(n) if n > 1 else 0.0
    relative_entropy = entropy_val / max_entropy if max_entropy > 0 else 0.0
    
    # 保留4位小数
    return {
        'probability_distribution': [round(p, 4) for p in prob_dist],
        'entropy': round(entropy_val, 4),
        'max_entropy': round(max_entropy, 4) if n > 0 else 0.0,
        'relative_entropy': round(relative_entropy, 4),
        'message': message
    }

# 示例用法
if __name__ == "__main__":
    # 测试对数概率输入
    test_cases = [
        ([-1.0, -2.0, -3.0], math.e),     # 自然对数概率
        ([-0.5, -0.5, -0.5], 2),          # 以2为底的对数概率
        ([-10, -10, -10], math.e),        # 小概率情况
        ([0, -1000, -1000], math.e),      # 包含零和对数概率
    ]
    
    for log_probs, base in test_cases:
        result = entropy_from_logprobs(log_probs, base)
        print(f"\n输入对数概率 (底数={base}): {log_probs}")
        print(f"概率分布: {result['probability_distribution']}")
        print(f"熵: {result['entropy']}")
        print(f"最大熵: {result['max_entropy']}")
        print(f"相对熵: {result['relative_entropy']}")
        print(f"信息: {result['message']}")