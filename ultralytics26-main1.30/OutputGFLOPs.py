import torch
from ultralytics import YOLO
import warnings
warnings.filterwarnings('ignore')

def calculate_model_flops(model_path):
    # 1. 将模型加载到 CPU 上，防止随机输入触发 CUDA 强行报错崩溃
    device = torch.device('cpu')
    print("正在加载模型...")
    model = YOLO(model_path).model.to(device)
    model.eval()

    # 2. 创建标准的 640x640 虚拟输入 (Batch size = 1)
    dummy_input = torch.randn(1, 3, 640, 640).to(device)
    
    print("\n======= 开始计算 GFLOPs =======")

    # ---------------------------------------------------------
    # 方法一：使用 thop (Ultralytics 自带依赖，最常用)
    # ---------------------------------------------------------
    try:
        from thop import profile
        # thop 会自动递归遍历子模块。你的 SRC_LDConv 内部的标准卷积会被精确计算，
        # 而类似 gather/切片等不消耗浮点计算的操作会计为 0 FLOPs（这非常符合逻辑，因为它们是 I/O 密集型）
        flops, params = profile(model, inputs=(dummy_input,), verbose=False)
        print(f"[thop 测算结果]")
        print(f"  - 模型参数量 (Params): {params:,}")
        print(f"  - 理论计算量 (GFLOPs): {flops / 1e9:.2f} GFLOPs")
    except Exception as e:
        print(f"thop 测算失败: {e}")

    print("-" * 40)


if __name__ == '__main__':
    # 指向你训练好的最佳权重
    calculate_model_flops('runs/detect/train15/weights/best.pt')