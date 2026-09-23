import torch
import torchvision.models as models
from torchvision.models import ViT_L_16_Weights
from torchvision import transforms
from PIL import Image
import time

def run_large_model_demo():
    # ==========================================
    # 1. 设备准备 (Device Setup)
    # ==========================================
    # 必须使用 GPU，因为 ViT-L 这种模型在 CPU 上运行会非常缓慢
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- 正在检测设备 ---")
    print(f"当前使用的设备: {device}")
    if device.type == 'cuda':
        print(f"GPU 型号: {torch.cuda.get_device_name(0)}")
        print(f"剩余显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("⚠️ 警告: 未检测到 GPU，ViT-L 模型在 CPU 上可能需要几分钟才能完成一次推理！")
    print("-" * 30)

    # ==========================================
    # 2. 加载预训练模型 (Loading ViT-L/16)
    # ==========================================
    print("\n[步骤 1: 正在下载/加载 ViT-L/16 模型...]")
    print("注意：该模型权重文件较大，初次运行需要时间下载。")

    try:
        # 使用最新的 weights 参数方式加载
        # ViT_L_16_Weights.IMAGENET1K_V1 是目前 torchvision 提供的权重
        weights = ViT_L_16_Weights.IMAGENET1K_V1
        model = models.vit_l_16(weights=weights)

        # 将模型移动到 GPU
        model = model.to(device)
        model.eval() # 设置为评估模式 (关闭 Dropout 等)
        print("✅ 模型加载成功并已移动到 GPU。")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return

    # 计算并打印参数量 (以亿为单位)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型总参数量: {total_params / 1e6:.2f} M (约 {total_params / 1e8:.2f} 亿参数)")

    # ==========================================
    # 3. 准备预处理流程 (Preprocessing)
    # ==========================================
    # Transformer 模型对输入图像非常敏感，必须使用官方指定的预处理方式
    print("\n[步骤 2: 准备数据预处理流程]")
    preprocess = weights.transforms()
    # 这会自动包含：Resize(224), CenterCrop(224), ToTensor, Normalize 等

    # ==========================================
    # 4. 模拟输入数据并进行推理 (Inference)
    # ==========================================
    print("\n[步骤 3: 开始推理测试]")

    # 创建一个随机的模拟图像 (Batch_size=1, Channels=3, Height=224, Width=224)
    # 在实际应用中，你会用 Image.open("path.jpg") 加载真实图片
    dummy_input = torch.randn(1, 3, 224, 224).to(device)

    # 开始计时
    start_time = time.time()

    try:
        with torch.no_grad(): # 必须使用 no_grad 以节省显存
            # 执行推理
            output = model(dummy_input)

        end_time = time.time()

        print(f"✅ 推理完成！")
        print(f"推理耗时: {end_time - start_time:.4f} 秒")
        print(f"输出张量形状: {output.shape}")

        # 获取预测概率最高的类别
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        top_prob, top_catid = torch.max(probabilities, 0)
        print(f"最高概率类别 ID: {top_catid.item()}, 置信度: {top_prob.item():.4f}")

    except RuntimeError as e:
        if "out of memory" in str(e):
            print("❌ 错误: 显存溢出 (OOM)! ViT-L 模型非常吃显存。请尝试关闭其他程序或使用更小的模型。")
        else:
            print(f"❌ 推理时发生错误: {e}")

if __name__ == "__main__":
    run_large_model_demo()

