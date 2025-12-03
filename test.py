import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import normalize
import pickle
import os
from .train import MultiModalPredictor

# ============ 1. 设置随机种子 ============
def set_seed(seed=20251203):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(20251203)

# ============ 2. 配置参数 ============
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
HIDDEN_DIM = 512  # 与训练时一致
MLP_HIDDEN_DIM = 512
DROPOUT_RATE = 0.2
N_FOLDS = 10

print("=" * 80)
print("改进模型测试 - 10折模型集成预测")
print("=" * 80)
print(f"使用设备: {DEVICE}")

# ============ 3. 主预测流程 ============
def main():
    # -------- 3.1 加载测试数据 --------
    print("\n[1/4] 加载测试数据...")
    img_test = np.load("test/image_features.npy")
    txt_test = np.load("test/text_features.npy")
    vid_test = np.load("test/video_features.npy")
    
    print(f"  图像特征: {img_test.shape}")
    print(f"  文本特征: {txt_test.shape}")
    print(f"  视频特征: {vid_test.shape}")
    
    # -------- 3.2 加载预处理器并处理测试数据 --------
    print("\n[2/4] 应用预处理...")
    
    # 加载预处理器
    print("  - 加载预处理器...")
    with open("models/scalers/vid_transformer.pkl", "rb") as f:
        vid_transformer = pickle.load(f)
    with open("models/scalers/x_scaler.pkl", "rb") as f:
        x_scaler = pickle.load(f)
    with open("models/scalers/y_scaler.pkl", "rb") as f:
        y_scaler = pickle.load(f)
    
    # 应用相同的预处理
    print("  - RankGauss转换视频特征...")
    vid_test = vid_transformer.transform(vid_test)
    
    print("  - L2归一化所有模态...")
    img_test = normalize(img_test, norm='l2', axis=1)
    txt_test = normalize(txt_test, norm='l2', axis=1)
    vid_test = normalize(vid_test, norm='l2', axis=1)
    
    # 拼接特征
    X_test = np.concatenate([img_test, txt_test, vid_test], axis=1).astype(np.float32)
    
    # StandardScaler
    print("  - 标准化特征...")
    X_test_scaled = x_scaler.transform(X_test)
    
    print(f"  ✓ 测试样本数: {len(X_test)}")
    
    # -------- 3.3 加载10个模型进行集成预测 --------
    print(f"\n[3/4] 加载{N_FOLDS}个模型并进行集成预测...")
    print("-" * 80)
    
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(DEVICE)
    all_predictions = []
    
    for fold in range(N_FOLDS):
        print(f"  加载 Fold {fold+1}/{N_FOLDS}...", end=" ")
        
        # 创建模型
        model = MultiModalPredictor(
            image_dim=1024,
            video_dim=1024,
            text_dim=768,
            hidden_dim=HIDDEN_DIM
        ).to(DEVICE)
        
        # 加载权重
        model_path = f"models/improved_10fold/fold_{fold}.pth"
        if not os.path.exists(model_path):
            print(f"\n  ❌ 错误: 模型文件 {model_path} 不存在!")
            print("  请先运行 train_improved.py 训练模型")
            return
        
        model.load_state_dict(torch.load(
            model_path,
            map_location=DEVICE,
            weights_only=True
        ))
        model.eval()
        
        # 预测
        with torch.no_grad():
            img_t = X_test_tensor[:, :1024]
            txt_t = X_test_tensor[:, 1024:2048]
            vid_t = X_test_tensor[:, 2048:]
            
            pred_scaled, attn = model(img_t, txt_t, vid_t)
            all_predictions.append(pred_scaled.cpu().numpy())
        
        print("✓")
    
    # -------- 3.4 集成并反归一化 --------
    print("\n[4/4] 集成预测结果并保存...")
    
    # 平均10个模型的预测
    avg_pred_scaled = np.mean(all_predictions, axis=0)
    
    # 反归一化
    final_predictions = y_scaler.inverse_transform(avg_pred_scaled)
    
    # 保存预测结果
    output_file = "predictions_improved.npy"
    np.save(output_file, final_predictions)
    
    print(f"\n✓ 预测完成!")
    print(f"  结果已保存到: {output_file}")
    print(f"  形状: {final_predictions.shape}")
    print("=" * 80)
    
    return final_predictions


if __name__ == '__main__':
    main()
