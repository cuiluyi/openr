import torch
import torch.nn.functional as F

device = "cuda"

def pairwise_rank_loss(labels: torch.Tensor, values: list[float]) -> torch.Tensor:
    # 将labels转换为布尔张量（0=False，1=True）
    labels_tensor = torch.tensor(labels, dtype=torch.bool, device=device)
    values_tensor = torch.tensor(values, dtype=torch.float32, requires_grad=True, device=device)

    # 分割正负样本
    pos_values = values_tensor[labels_tensor]  # 正样本（标签为1）
    neg_values = values_tensor[~labels_tensor]  # 负样本（标签为0）

    n, m = len(pos_values), len(neg_values)

    # 处理无正负样本的情况
    if n == 0 or m == 0:
        return torch.tensor(0.0, requires_grad=True, device=device)

    # 计算所有正负样本对的得分差异
    diffs = pos_values.view(-1, 1) - neg_values.view(1, -1)  # 形状 (n, m)

    # 使用数值稳定的对数sigmoid损失
    loss = -F.logsigmoid(diffs).mean()  # 等价于 -log(σ(diffs)).mean()
    return loss


if __name__ == "__main__":
    # 测试极端值情况
    labels = [1, 1, 0, 0]
    values = [0.7, 0.65, 0.4, 0.2]  # 极大差异
    loss = pairwise_rank_loss(labels, values)  # 正常输出约 0.0

    print(loss)
