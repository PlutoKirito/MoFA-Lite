import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 80)
print("特征数据统计分析")
print("=" * 80)

# 加载原始训练数据
print("\n加载原始训练数据...")
video_features = np.load('Data/train/video_features.npy')
text_features = np.load('Data/train/text_features.npy')
image_features = np.load('Data/train/image_features.npy')
popularity_counts = np.load('Data/train/popularity_counts.npy')

print(f"video_features 形状: {video_features.shape}")
print(f"text_features 形状: {text_features.shape}")
print(f"image_features 形状: {image_features.shape}")
print(f"popularity_counts 形状: {popularity_counts.shape}")

# 创建结果保存目录
os.makedirs('feature_analysis', exist_ok=True)

def compute_statistics(data, name):
    """计算特征数据的统计特征"""
    print(f"\n{'='*60}")
    print(f"{name} 统计分析")
    print(f"{'='*60}")
    
    # 基本统计量
    print(f"数据形状: {data.shape}")
    print(f"数据类型: {data.dtype}")
    print(f"\n全局统计:")
    print(f"均值: {np.mean(data):.6f}")
    print(f"中位数: {np.median(data):.6f}")
    print(f"标准差: {np.std(data):.6f}")
    print(f"方差: {np.var(data):.6f}")
    print(f"最小值: {np.min(data):.6f}")
    print(f"最大值: {np.max(data):.6f}")
    print(f"范围: {np.ptp(data):.6f}")
    
    # 检查异常值
    nan_count = np.sum(np.isnan(data))
    inf_count = np.sum(np.isinf(data))
    print(f"\n数据质量:")
    print(f"NaN 数量: {nan_count} ({nan_count/data.size*100:.4f}%)")
    print(f"Inf 数量: {inf_count} ({inf_count/data.size*100:.4f}%)")
    
    # 按样本统计（每行）
    if data.ndim == 2:
        sample_means = np.mean(data, axis=1)
        sample_stds = np.std(data, axis=1)
        sample_norms = np.linalg.norm(data, axis=1)
        
        print(f"\n按样本统计 (每行):")
        print(f"样本均值的均值: {np.mean(sample_means):.6f}")
        print(f"样本均值的标准差: {np.std(sample_means):.6f}")
        print(f"样本标准差的均值: {np.mean(sample_stds):.6f}")
        print(f"样本标准差的标准差: {np.std(sample_stds):.6f}")
        print(f"样本范数的均值: {np.mean(sample_norms):.6f}")
        print(f"样本范数的标准差: {np.std(sample_norms):.6f}")
        
        # 按特征统计（每列）
        feature_means = np.mean(data, axis=0)
        feature_stds = np.std(data, axis=0)
        
        print(f"\n按特征统计 (每列):")
        print(f"特征均值的均值: {np.mean(feature_means):.6f}")
        print(f"特征均值的标准差: {np.std(feature_means):.6f}")
        print(f"特征标准差的均值: {np.mean(feature_stds):.6f}")
        print(f"特征标准差的标准差: {np.std(feature_stds):.6f}")
        
        # 分位数统计
        print(f"\n分位数统计:")
        for q in [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]:
            print(f"{q*100:.0f}%: {np.percentile(data, q*100):.6f}")
        
        return {
            'name': name,
            'shape': data.shape,
            'mean': np.mean(data),
            'median': np.median(data),
            'std': np.std(data),
            'min': np.min(data),
            'max': np.max(data),
            'sample_means': sample_means,
            'sample_stds': sample_stds,
            'sample_norms': sample_norms,
            'feature_means': feature_means,
            'feature_stds': feature_stds
        }
    else:
        return {
            'name': name,
            'shape': data.shape,
            'mean': np.mean(data),
            'median': np.median(data),
            'std': np.std(data),
            'min': np.min(data),
            'max': np.max(data)
        }

# 计算原始数据统计
print("\n" + "="*80)
print("原始数据统计分析")
print("="*80)

video_stats = compute_statistics(video_features, "Video Features (原始)")
text_stats = compute_statistics(text_features, "Text Features (原始)")
image_stats = compute_statistics(image_features, "Image Features (原始)")

# 分析流行度特征
print("\n" + "="*80)
print("流行度特征统计分析")
print("="*80)

# 提取三个流行度指标
plays = popularity_counts[:, 0]
likes = popularity_counts[:, 1]
comments = popularity_counts[:, 2]

def analyze_popularity(data, name):
    """分析流行度数据的统计特征"""
    print(f"\n{'='*60}")
    print(f"{name} 统计分析")
    print(f"{'='*60}")
    
    print(f"数据形状: {data.shape}")
    print(f"\n基本统计:")
    print(f"均值: {np.mean(data):.6f}")
    print(f"中位数: {np.median(data):.6f}")
    print(f"标准差: {np.std(data):.6f}")
    print(f"方差: {np.var(data):.6f}")
    print(f"最小值: {np.min(data):.6f}")
    print(f"最大值: {np.max(data):.6f}")
    print(f"范围: {np.ptp(data):.6f}")
    
    # 分位数
    print(f"\n分位数:")
    for q in [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]:
        print(f"{q*100:.0f}%: {np.percentile(data, q*100):.6f}")
    
    # 偏度和峰度
    skewness = stats.skew(data)
    kurtosis = stats.kurtosis(data)
    print(f"\n分布特征:")
    print(f"偏度 (Skewness): {skewness:.4f}")
    print(f"  |偏度| < 0.5: 近似对称")
    print(f"  0.5 < |偏度| < 1: 中度偏态")
    print(f"  |偏度| > 1: 严重偏态")
    print(f"峰度 (Kurtosis): {kurtosis:.4f}")
    print(f"  峰度 = 0: 正态分布")
    print(f"  峰度 > 0: 厚尾分布")
    print(f"  峰度 < 0: 薄尾分布")
    
    # 检查异常值
    zero_count = np.sum(data == 0)
    negative_count = np.sum(data < 0)
    print(f"\n数据质量:")
    print(f"零值数量: {zero_count} ({zero_count/len(data)*100:.2f}%)")
    print(f"负值数量: {negative_count} ({negative_count/len(data)*100:.2f}%)")
    
    return {
        'name': name,
        'data': data,
        'mean': np.mean(data),
        'median': np.median(data),
        'std': np.std(data),
        'min': np.min(data),
        'max': np.max(data),
        'skewness': skewness,
        'kurtosis': kurtosis,
        'q25': np.percentile(data, 25),
        'q75': np.percentile(data, 75)
    }

plays_stats = analyze_popularity(plays, "播放量 (对数化后)")
likes_stats = analyze_popularity(likes, "点赞量 (对数化后)")
comments_stats = analyze_popularity(comments, "评论量 (对数化后)")

# 计算对数比值
like_play_ratio = likes - plays
comment_play_ratio = comments - plays

print("\n" + "="*60)
print("对数比值分析")
print("="*60)

like_ratio_stats = analyze_popularity(like_play_ratio, "点赞/播放 对数比值")
comment_ratio_stats = analyze_popularity(comment_play_ratio, "评论/播放 对数比值")

# 绘制统计可视化
# 绘制流行度特征可视化
print("\n" + "="*80)
print("生成流行度特征可视化图表...")
print("="*80)

def plot_popularity_analysis(pop_stats_list, save_dir='feature_analysis'):
    """绘制流行度特征分析图"""
    
    # 1. 流行度分布对比
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('流行度特征分布分析', fontsize=16, fontweight='bold')
    
    # 播放量、点赞量、评论量的直方图
    for idx, stats in enumerate(pop_stats_list[:3]):
        ax = axes[0, idx]
        ax.hist(stats['data'], bins=50, alpha=0.7, edgecolor='black', color=['blue', 'orange', 'green'][idx])
        ax.set_xlabel('对数值')
        ax.set_ylabel('频数')
        ax.set_title(f"{stats['name']}\n均值:{stats['mean']:.3f}, 标准差:{stats['std']:.3f}")
        ax.grid(axis='y', alpha=0.3)
        ax.axvline(stats['mean'], color='red', linestyle='--', linewidth=2, label='均值')
        ax.axvline(stats['median'], color='purple', linestyle='--', linewidth=2, label='中位数')
        ax.legend()
    
    # 箱线图对比
    ax = axes[1, 0]
    box_data = [s['data'] for s in pop_stats_list[:3]]
    box_labels = [s['name'].split('(')[0].strip() for s in pop_stats_list[:3]]
    bp = ax.boxplot(box_data, labels=box_labels, patch_artist=True, showmeans=True)
    for patch, color in zip(bp['boxes'], ['lightblue', 'lightyellow', 'lightgreen']):
        patch.set_facecolor(color)
    ax.set_ylabel('对数值')
    ax.set_title('流行度指标箱线图对比')
    ax.grid(axis='y', alpha=0.3)
    
    # 小提琴图对比
    ax = axes[1, 1]
    parts = ax.violinplot(box_data, showmeans=True, showmedians=True)
    ax.set_xticks(range(1, len(box_labels) + 1))
    ax.set_xticklabels(box_labels, rotation=15)
    ax.set_ylabel('对数值')
    ax.set_title('流行度指标小提琴图对比')
    ax.grid(axis='y', alpha=0.3)
    
    # 统计量对比条形图
    ax = axes[1, 2]
    x = np.arange(len(box_labels))
    width = 0.35
    ax.bar(x - width/2, [s['mean'] for s in pop_stats_list[:3]], width, label='均值', alpha=0.7)
    ax.bar(x + width/2, [s['std'] for s in pop_stats_list[:3]], width, label='标准差', alpha=0.7)
    ax.set_ylabel('数值')
    ax.set_title('均值与标准差对比')
    ax.set_xticks(x)
    ax.set_xticklabels(box_labels, rotation=15)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, '7_流行度特征分布.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ 保存: 7_流行度特征分布.png")
    
    # 2. 对数比值分析
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('对数比值分析 (点赞/播放, 评论/播放)', fontsize=16, fontweight='bold')
    
    # 点赞/播放比值分布
    ax = axes[0, 0]
    ax.hist(pop_stats_list[3]['data'], bins=50, alpha=0.7, edgecolor='black', color='coral')
    ax.set_xlabel('对数差 (点赞 - 播放)')
    ax.set_ylabel('频数')
    ax.set_title(f"点赞/播放 对数比值\n均值:{pop_stats_list[3]['mean']:.3f}, 标准差:{pop_stats_list[3]['std']:.3f}")
    ax.axvline(0, color='red', linestyle='--', linewidth=2, label='比值=1 (对数差=0)')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # 评论/播放比值分布
    ax = axes[0, 1]
    ax.hist(pop_stats_list[4]['data'], bins=50, alpha=0.7, edgecolor='black', color='skyblue')
    ax.set_xlabel('对数差 (评论 - 播放)')
    ax.set_ylabel('频数')
    ax.set_title(f"评论/播放 对数比值\n均值:{pop_stats_list[4]['mean']:.3f}, 标准差:{pop_stats_list[4]['std']:.3f}")
    ax.axvline(0, color='red', linestyle='--', linewidth=2, label='比值=1 (对数差=0)')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # 散点图：点赞 vs 播放
    ax = axes[1, 0]
    sample_idx = np.random.choice(len(plays), min(5000, len(plays)), replace=False)
    ax.scatter(plays[sample_idx], likes[sample_idx], alpha=0.3, s=5)
    ax.plot([plays.min(), plays.max()], [plays.min(), plays.max()], 
            'r--', linewidth=2, label='点赞=播放')
    ax.set_xlabel('播放量 (对数)')
    ax.set_ylabel('点赞量 (对数)')
    ax.set_title('点赞量 vs 播放量 (抽样5000)')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # 散点图：评论 vs 播放
    ax = axes[1, 1]
    ax.scatter(plays[sample_idx], comments[sample_idx], alpha=0.3, s=5, color='green')
    ax.plot([plays.min(), plays.max()], [plays.min(), plays.max()], 
            'r--', linewidth=2, label='评论=播放')
    ax.set_xlabel('播放量 (对数)')
    ax.set_ylabel('评论量 (对数)')
    ax.set_title('评论量 vs 播放量 (抽样5000)')
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, '8_对数比值分析.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ 保存: 8_对数比值分析.png")
    
    # 3. 相关性热图
    fig, ax = plt.subplots(figsize=(8, 6))
    corr_matrix = np.corrcoef([plays, likes, comments])
    im = ax.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    
    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels(['播放量', '点赞量', '评论量'])
    ax.set_yticklabels(['播放量', '点赞量', '评论量'])
    
    # 添加数值标签
    for i in range(3):
        for j in range(3):
            text = ax.text(j, i, f'{corr_matrix[i, j]:.3f}',
                          ha="center", va="center", color="black", fontsize=12)
    
    ax.set_title('流行度指标相关性矩阵', fontsize=14, fontweight='bold')
    plt.colorbar(im, ax=ax, label='相关系数')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, '9_流行度相关性.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ 保存: 9_流行度相关性.png")

# 生成流行度可视化
pop_stats_list = [plays_stats, likes_stats, comments_stats, like_ratio_stats, comment_ratio_stats]
plot_popularity_analysis(pop_stats_list)

print("\n" + "="*80)
print("生成输入特征可视化图表...")
print("="*80)

def plot_feature_comparison(stats_list, save_dir='feature_analysis'):
    """绘制特征对比图"""
    
    # 1. 基本统计量对比条形图
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('特征数据基本统计量对比', fontsize=16, fontweight='bold')
    
    metrics = ['mean', 'std', 'min', 'max', 'median']
    metric_names = ['均值', '标准差', '最小值', '最大值', '中位数']
    
    for idx, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
        ax = axes[idx // 3, idx % 3]
        
        names = [s['name'] for s in stats_list]
        values = [s[metric] for s in stats_list]
        
        bars = ax.bar(range(len(names)), values, alpha=0.7, label='原始')
        
        ax.set_xlabel('特征类型')
        ax.set_ylabel(metric_name)
        ax.set_title(f'{metric_name}对比')
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels([n.split('(')[0].strip() for n in names], rotation=15)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        # 添加数值标签
        for i, v in enumerate(values):
            ax.text(i, v, f'{v:.2e}', ha='center', va='bottom', fontsize=8)
    
    # 删除多余的子图
    if len(metrics) < 6:
        fig.delaxes(axes[1, 2])
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, '1_基本统计量对比.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ 保存: 1_基本统计量对比.png")
    
    # 2. 样本范数分布对比
    if all('sample_norms' in s for s in stats_list):
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle('样本特征范数分布对比', fontsize=16, fontweight='bold')
        
        for idx, stats in enumerate(stats_list):
            ax = axes[idx]
            norms = stats['sample_norms']
            
            ax.hist(norms, bins=50, alpha=0.6, color='blue', edgecolor='black', label='原始')
            
            ax.set_xlabel('特征范数')
            ax.set_ylabel('样本数量')
            ax.set_title(stats['name'].split('(')[0].strip())
            ax.legend()
            ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, '2_样本范数分布.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print("✓ 保存: 2_样本范数分布.png")
        
        # 3. 样本范数箱线图
        fig, ax = plt.subplots(figsize=(12, 6))
        
        box_data = [s['sample_norms'] for s in stats_list]
        box_labels = [s['name'].split('(')[0].strip() for s in stats_list]
        
        bp = ax.boxplot(box_data, labels=box_labels, patch_artist=True,
                       showmeans=True, meanline=True)
        
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
            patch.set_alpha(0.7)
        
        ax.set_ylabel('特征范数')
        ax.set_title('样本特征范数箱线图对比', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, '3_样本范数箱线图.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print("✓ 保存: 3_样本范数箱线图.png")
        
        # 4. 样本均值和标准差散点图
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle('样本均值 vs 标准差散点图', fontsize=16, fontweight='bold')
        
        for idx, stats in enumerate(stats_list):
            ax = axes[idx]
            means = stats['sample_means']
            stds = stats['sample_stds']
            
            ax.scatter(means, stds, alpha=0.5, s=10, label='原始')
            
            ax.set_xlabel('样本均值')
            ax.set_ylabel('样本标准差')
            ax.set_title(stats['name'].split('(')[0].strip())
            ax.legend()
            ax.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, '4_样本均值标准差散点图.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print("✓ 保存: 4_样本均值标准差散点图.png")
        
        # 5. 特征维度统计热图（仅展示前100个维度）
        fig, axes = plt.subplots(3, 1, figsize=(15, 12))
        fig.suptitle('特征维度统计热图 (前100维)', fontsize=16, fontweight='bold')
        
        for idx, stats in enumerate(stats_list):
            ax = axes[idx]
            
            # 构建统计矩阵：均值和标准差
            n_dims = min(100, len(stats['feature_means']))
            stat_matrix = np.vstack([
                stats['feature_means'][:n_dims],
                stats['feature_stds'][:n_dims]
            ])
            
            im = ax.imshow(stat_matrix, aspect='auto', cmap='coolwarm')
            ax.set_yticks([0, 1])
            ax.set_yticklabels(['均值', '标准差'])
            ax.set_xlabel('特征维度索引')
            ax.set_title(stats['name'].split('(')[0].strip())
            plt.colorbar(im, ax=ax, label='数值')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, '5_特征维度统计热图.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print("✓ 保存: 5_特征维度统计热图.png")
        
        # 6. 三种特征的全局分布对比（小提琴图）
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # 由于数据量大，随机采样部分数据进行可视化
        sample_size = 5000
        violin_data = []
        violin_labels = []
        
        for stats in stats_list:
            name = stats['name'].split('(')[0].strip()
            # 从原始数据中采样
            if name == "Video Features":
                data = video_features.flatten()
            elif name == "Text Features":
                data = text_features.flatten()
            else:
                data = image_features.flatten()
            
            sampled = np.random.choice(data, min(sample_size, len(data)), replace=False)
            violin_data.append(sampled)
            violin_labels.append(name)
        
        parts = ax.violinplot(violin_data, showmeans=True, showmedians=True)
        ax.set_xticks(range(1, len(violin_labels) + 1))
        ax.set_xticklabels(violin_labels)
        ax.set_ylabel('特征值')
        ax.set_title('特征值分布对比 (小提琴图)', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, '6_特征值分布小提琴图.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print("✓ 保存: 6_特征值分布小提琴图.png")

# 生成可视化
stats_list = [video_stats, text_stats, image_stats]

# 生成统计报告
print("\n" + "="*80)
print("生成统计分析报告...")
print("="*80)

def generate_report(stats_list, pop_stats_list=None, save_path='feature_analysis/统计分析报告.txt'):
    """生成详细的统计分析报告"""
    report = []
    report.append("=" * 80)
    report.append("特征数据统计分析报告")
    report.append("=" * 80)
    report.append(f"生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    
    report.append("一、数据概况")
    report.append("-" * 80)
    report.append("\n输入特征:")
    for stats in stats_list:
        report.append(f"  {stats['name']}: 形状 {stats['shape']}, 总元素数 {np.prod(stats['shape'])}")
    
    if pop_stats_list:
        report.append("\n流行度特征(目标变量):")
        for stats in pop_stats_list:
            report.append(f"  {stats['name']}: 样本数 {len(stats['data'])}")
    
    report.append("\n\n二、基本统计量对比")
    report.append("-" * 80)
    report.append(f"{'指标':<20} {'Video':<15} {'Text':<15} {'Image':<15}")
    report.append("-" * 80)
    
    metrics = [('均值', 'mean'), ('中位数', 'median'), ('标准差', 'std'), 
               ('最小值', 'min'), ('最大值', 'max')]
    
    for metric_name, metric_key in metrics:
        values = [f"{s[metric_key]:.6e}" for s in stats_list]
        report.append(f"{metric_name:<20} {values[0]:<15} {values[1]:<15} {values[2]:<15}")
    
    report.append("\n\n四、输入特征差异分析")
    report.append("-" * 80)
    report.append(f"Video Features (768维): 均值 {video_stats['mean']:.6e}, 标准差 {video_stats['std']:.6e}")
    report.append(f"Text Features (1024维): 均值 {text_stats['mean']:.6e}, 标准差 {text_stats['std']:.6e}")
    report.append(f"Image Features (1024维): 均值 {image_stats['mean']:.6e}, 标准差 {image_stats['std']:.6e}")
    report.append("")
    
    if pop_stats_list:
        report.append("\n\n五、流行度特征(目标变量)统计分析")
        report.append("-" * 80)
        report.append(f"{'指标':<20} {'播放量':<15} {'点赞量':<15} {'评论量':<15}")
        report.append("-" * 80)
        
        pop_metrics = [
            ('均值', 'mean'),
            ('中位数', 'median'),
            ('标准差', 'std'),
            ('最小值', 'min'),
            ('最大值', 'max'),
            ('偏度(Skewness)', 'skewness'),
            ('峰度(Kurtosis)', 'kurtosis')
        ]
        
        for metric_name, metric_key in pop_metrics:
            values = [f"{pop_stats_list[i][metric_key]:.4f}" for i in range(3)]
            report.append(f"{metric_name:<20} {values[0]:<15} {values[1]:<15} {values[2]:<15}")
        
        report.append("\n对数比值分析:")
        report.append(f"  点赞/播放 对数差: 均值={pop_stats_list[3]['mean']:.4f}, 标准差={pop_stats_list[3]['std']:.4f}")
        report.append(f"  评论/播放 对数差: 均值={pop_stats_list[4]['mean']:.4f}, 标准差={pop_stats_list[4]['std']:.4f}")
        
        report.append("\n\n七、预处理建议")
        report.append("-" * 80)
        report.append("(1) 输入特征预处理:")
    
    report.append("\n" + "=" * 80)
    
    report_text = "\n".join(report)
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print(report_text)
    return report_text

generate_report(stats_list, pop_stats_list)
