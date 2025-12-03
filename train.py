import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler, QuantileTransformer, normalize
import pickle
import os

#  1. 设置随机种子
def set_seed(seed=20251203):
    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)

    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False

    torch.backends.cudnn.deterministic = True

set_seed(20251203)

# 2. 配置参数
BATCH_SIZE = 128
LEARNING_RATE = 5e-4  # 适中学习率
EPOCHS = 300
PATIENCE = 20  # 适度早停
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
HIDDEN_DIM = 512  # 适中隐藏层维度
DROPOUT_RATE = 0.2  # 降低Dropout避免过度正则化
N_FOLDS = 10  # K-Fold 交叉验证
VALIDATION_SIZE = 0.1  # 10% 验证集

print("=" * 80)
print("视频流行度预测模型训练")
print("策略: K-Fold 交叉验证 + 单模型多输出 + 轻量级多模态注意力 + 模型集成")
print("=" * 80)
print(f"使用设备: {DEVICE}")


# 3. 辅助函数: 计算NMSE评估指标
def calc_nmse(y_true, y_pred):
    """计算Normalized Mean Square Error"""
    numerator = np.sum((y_true - y_pred) ** 2)
    denominator = np.sum((y_true - np.mean(y_true, axis=0)) ** 2)
    return numerator / (denominator + 1e-9)


# 4. 多模态架构
class MultiModalPredictor(nn.Module):
    """
    多模态预测器：
    - 分模态编码
    - 使用轻量级注意力机制
    - 单模型预测3个目标
    """
    def __init__(self, image_dim=1024, video_dim=768, text_dim=1024, hidden_dim=512):
        super(MultiModalPredictor, self).__init__()
        
        #  阶段1: 分模态编码器 
        self.image_encoder = nn.Sequential(
            nn.Linear(image_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(DROPOUT_RATE)
        )
        
        self.video_encoder = nn.Sequential(
            nn.Linear(video_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(DROPOUT_RATE)
        )
        
        self.text_encoder = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(DROPOUT_RATE)
        )
        
        #  阶段2: 轻量级模态注意力 
        self.modal_attention = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Linear(128, 1)   # 为每个模态生成一个重要性分数
        )
        
        #  阶段3: 特征融合 
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.25),
            
            nn.Linear(hidden_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(DROPOUT_RATE)
        )
        
        #  阶段4: 多目标预测头 
        self.predictor = nn.Sequential(
            nn.Linear(384, 256),
            nn.GELU(),
            nn.Dropout(DROPOUT_RATE),
            
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(128, 3)  # 输出3个目标
        )
    
    def forward(self, image, video, text):
        # 阶段1: 分模态编码
        img_feat = self.image_encoder(image)
        vid_feat = self.video_encoder(video)
        txt_feat = self.text_encoder(text)
        
        # 阶段2: 计算模态注意力权重
        modal_feats = torch.stack([img_feat, vid_feat, txt_feat], dim=1)
        
        attn_scores = torch.cat([
            self.modal_attention(img_feat),
            self.modal_attention(vid_feat),
            self.modal_attention(txt_feat)
        ], dim=1)
        
        attn_weights = torch.softmax(attn_scores, dim=1).unsqueeze(-1)
        
        # 加权模态特征
        weighted_feats = modal_feats * attn_weights
        
        # 阶段3: 拼接融合
        concat_feat = weighted_feats.view(weighted_feats.size(0), -1)
        fused_feat = self.fusion(concat_feat)
        
        # 阶段4: 预测
        output = self.predictor(fused_feat)
        
        return output, attn_weights.squeeze(-1)


#  5. 主训练流程：K-Fold 交叉验证 
def main():
    # -------- 5.1 数据加载 --------
    print("\n[1/5] 加载数据...")
    img = np.load("Data/train/image_features.npy")
    txt = np.load("Data/train/text_features.npy")
    vid = np.load("Data/train/video_features.npy")
    pop = np.load("Data/train/popularity_counts.npy")
    
    print(f"  图像特征: {img.shape}")
    print(f"  文本特征: {txt.shape}")
    print(f"  视频特征: {vid.shape}")
    print(f"  流行度标签: {pop.shape}")
    
    # -------- 5.2 数据预处理 --------
    print("\n[2/5] 对数据进行预处理...")
    # RankGauss转换视频特征
    print("  - 对视频特征应用RankGauss转换（高斯化）")
    vid_transformer = QuantileTransformer(
        output_distribution='normal',
        n_quantiles=1000,
        random_state=20251203
    )
    vid = vid_transformer.fit_transform(vid)
    
    # L2 Normalize
    print("  - L2归一化所有模态")
    img = normalize(img, norm='l2', axis=1)
    txt = normalize(txt, norm='l2', axis=1)
    vid = normalize(vid, norm='l2', axis=1)
    
    # 拼接特征
    X_all = np.concatenate([img, txt, vid], axis=1).astype(np.float32)
    y_all = pop.astype(np.float32)
    
    # 划分验证集
    print(f"  - 划分验证集 ({VALIDATION_SIZE*100:.0f}%)...")
    X_dev, X_validation, y_dev, y_validation = train_test_split(
        X_all, y_all, test_size=VALIDATION_SIZE, random_state=20251203
    )
    
    # 数据标准化
    print("  - 标准化特征和目标...")
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    
    X_dev_scaled = x_scaler.fit_transform(X_dev)
    y_dev_scaled = y_scaler.fit_transform(y_dev)
    X_validation_scaled = x_scaler.transform(X_validation)
    
    # 保存标准化参数（用于test.py）
    os.makedirs("models/scalers", exist_ok=True)
    os.makedirs("models/improved_10fold", exist_ok=True)
    
    with open("models/scalers/x_scaler.pkl", "wb") as f:
        pickle.dump(x_scaler, f)
    with open("models/scalers/y_scaler.pkl", "wb") as f:
        pickle.dump(y_scaler, f)
    with open("models/scalers/vid_transformer.pkl", "wb") as f:
        pickle.dump(vid_transformer, f)
    
    print(f"  ✓ 训练集: {len(X_dev)} 样本")
    print(f"  ✓ 验证集: {len(X_validation)} 样本")
    print(f"  ✓ 预处理器已保存")
    
    # -------- 5.3 K-Fold 交叉验证训练 --------
    print(f"\n[3/5] 开始{N_FOLDS}折交叉验证训练...")
    print("-" * 80)
    
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=20251203)
    X_dev_tensor = torch.tensor(X_dev_scaled, dtype=torch.float32)
    y_dev_tensor = torch.tensor(y_dev_scaled, dtype=torch.float32)
    
    fold_results = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_dev_tensor)):
        print(f"\n>>> Fold {fold + 1}/{N_FOLDS} <<<")
        
        # 准备数据
        X_train = X_dev_tensor[train_idx]
        y_train = y_dev_tensor[train_idx]
        X_val = X_dev_tensor[val_idx]
        y_val = y_dev_tensor[val_idx]
        y_val_raw = y_dev[val_idx]
        
        train_loader = DataLoader(
            TensorDataset(X_train, y_train),
            batch_size=BATCH_SIZE,
            shuffle=True
        )
        
        # 创建模型
        model = MultiModalPredictor(
            image_dim=1024,
            video_dim=1024,
            text_dim=768,
            hidden_dim=HIDDEN_DIM
        ).to(DEVICE)
        
        if fold == 0:
            total_params = sum(p.numel() for p in model.parameters())
            print(f"  模型参数量: {total_params:,}")
        
        # 优化器和调度器
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-3)
        criterion = nn.MSELoss()
        scheduler = ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        
        best_nmse = float('inf')
        no_improve = 0
        
        # 训练循环
        for epoch in range(EPOCHS):
            # 训练
            model.train()
            for xb, yb in train_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                
                # 分离模态特征
                img_b = xb[:, :1024]
                txt_b = xb[:, 1024:2048]
                vid_b = xb[:, 2048:]
                
                pred, _ = model(img_b, txt_b, vid_b)
                loss = criterion(pred, yb)
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            
            # 验证
            model.eval()
            with torch.no_grad():
                X_val_gpu = X_val.to(DEVICE)
                img_val = X_val_gpu[:, :1024]
                txt_val = X_val_gpu[:, 1024:2048]
                vid_val = X_val_gpu[:, 2048:]
                
                pred_val_scaled, attn = model(img_val, txt_val, vid_val)
                val_loss = criterion(pred_val_scaled.cpu(), y_val).item()
                
                # 反归一化计算NMSE
                pred_val_raw = y_scaler.inverse_transform(pred_val_scaled.cpu().numpy())
                nmse = calc_nmse(y_val_raw, pred_val_raw)
            
            scheduler.step(val_loss)
            
            # 保存最佳模型
            if nmse < best_nmse:
                best_nmse = nmse
                no_improve = 0
                torch.save(model.state_dict(),
                          f"models/improved_{N_FOLDS}fold/fold_{fold}.pth")
                if (epoch + 1) % 20 == 0:
                    avg_attn = attn.mean(dim=0).cpu().numpy()
                    print(f"  Epoch {epoch+1:3d}: NMSE={nmse:.6f} (Best) | "
                          f"Attn[Img:{avg_attn[0]:.3f}, Txt:{avg_attn[1]:.3f}, Vid:{avg_attn[2]:.3f}]")
            else:
                no_improve += 1
            
            if no_improve >= PATIENCE:
                print(f"  Early stop at epoch {epoch+1}")
                break
        
        fold_results.append(best_nmse)
        print(f"  ✓ Fold {fold+1} 最佳NMSE: {best_nmse:.6f}")
    
    # -------- 5.4 验证集集成评估 --------
    print(f"\n[4/5] 验证集集成评估...")
    print("-" * 80)
    
    X_validation_tensor = torch.tensor(X_validation_scaled, dtype=torch.float32).to(DEVICE)
    validation_preds = []
    
    for fold in range(N_FOLDS):
        model = MultiModalPredictor(
            image_dim=1024,
            video_dim=1024,
            text_dim=768,
            hidden_dim=HIDDEN_DIM
        ).to(DEVICE)
        
        model.load_state_dict(torch.load(
            f"models/improved_10fold/fold_{fold}.pth",
            map_location=DEVICE,
            weights_only=True
        ))
        model.eval()
        
        with torch.no_grad():
            img_v = X_validation_tensor[:, :1024]
            txt_v = X_validation_tensor[:, 1024:2048]
            vid_v = X_validation_tensor[:, 2048:]
            
            pred, attn = model(img_v, txt_v, vid_v)
            validation_preds.append(pred.cpu().numpy())
    
    # 平均K个模型的预测
    avg_pred_scaled = np.mean(validation_preds, axis=0)
    avg_pred_raw = y_scaler.inverse_transform(avg_pred_scaled)
    
    final_nmse = calc_nmse(y_validation, avg_pred_raw)
    
    # 计算每个目标的NMSE
    nmse_per_target = []
    for i in range(3):
        nmse_i = calc_nmse(y_validation[:, i], avg_pred_raw[:, i])
        nmse_per_target.append(nmse_i)
    
    # -------- 5.5 结果汇总 --------
    print(f"\n[5/5] 训练完成！")
    print("=" * 80)
    print("交叉验证结果:")
    print(f"  平均CV NMSE: {np.mean(fold_results):.6f} ± {np.std(fold_results):.6f}")
    print(f"  各折NMSE: {', '.join([f'{x:.6f}' for x in fold_results])}")
    
    print(f"\n验证集集成结果:")
    print(f"  总体NMSE: {final_nmse:.6f}")
    print(f"  播放量NMSE: {nmse_per_target[0]:.6f}")
    print(f"  点赞量NMSE: {nmse_per_target[1]:.6f}")
    print(f"  评论量NMSE: {nmse_per_target[2]:.6f}")
    
    print(f"\n保存的模型:")
    print(f"  - models/improved_{N_FOLDS}fold/fold_{{0-9}}.pth")
    print(f"  - models/scalers/*.pkl")
    
    print("\n✓ 训练完成！")
    print("=" * 80)
    
    return final_nmse


if __name__ == '__main__':
    main()
