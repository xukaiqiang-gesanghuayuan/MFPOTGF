import torch
import os
import torch.nn as nn
import random
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from my_dataset5 import TripleNiftiDataset
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model
from sklearn.metrics import precision_score, recall_score, f1_score, roc_curve, auc, matthews_corrcoef
import ot  # Import the POT library for optimal transport

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class Block(nn.Module):
    """ ConvNeXt Block adapted for 3D inputs.
    """
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv3d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise 3D conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 4, 1)  # Permute to bring the channel to the last dimension
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 4, 1, 2, 3)  # Permute back

        x = input + self.drop_path(x)
        return x

class ConvNeXt(nn.Module):
    def __init__(self, in_chans=1, depths=[1, 1, 3, 1], dims=[96, 192, 384, 768], drop_path_rate=0., layer_scale_init_value=1e-6):
        super().__init__()

        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv3d(in_chans, dims[0], kernel_size=4, stride=4),
            nn.BatchNorm3d(dims[0])  # Use BatchNorm3d instead of LayerNorm for 3D data
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                nn.BatchNorm3d(dims[i]),  # Use BatchNorm3d instead of LayerNorm
                nn.Conv3d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j], layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv3d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        stage_outputs = []  # 保存每个stage的输出
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            stage_outputs.append(x)
            #print(f"Stage {i+1} output shape: {x.shape}")  # 打印每个stage的输出形状
        return stage_outputs

    def forward(self, x):
        return self.forward_features(x)


class LayerNorm(nn.Module):
    """ LayerNorm that supports two data formats: channels_last (default) or channels_first.
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape,)
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None, None] * x + self.bias[:, None, None, None]
            return x

# ==================== 新增的GAF模块 ====================
class GaussianAlignmentFusion(nn.Module):
    """高斯对齐融合模块 (Gaussian Alignment Fusion, GAF)"""
    def __init__(self):
        super(GaussianAlignmentFusion, self).__init__()
    
    def forward(self, f1_flat, f2_flat, f3_flat):
        """
        实现高斯对齐融合
        Args:
            f1_flat, f2_flat, f3_flat: 展平后的特征，形状为 [B, N]
        Returns:
            fused_feature: 高斯对齐融合后的特征
        """
        # 特征拼接 [B, 3N]
        fcat = torch.cat([f1_flat, f2_flat, f3_flat], dim=1)  # [B, 3N]
        
        # 全局均值计算
        mu = torch.mean(fcat, dim=1, keepdim=True)  # [B, 1]
        # 扩展mu到与fcat相同的形状进行广播
        mu_expanded = mu.expand_as(fcat)  # [B, 3N]
        
        # 全局标准差计算
        sigma = torch.sqrt(torch.mean((fcat - mu_expanded) ** 2, dim=1, keepdim=True))  # [B, 1]
        sigma = torch.clamp(sigma, min=1e-8)  # 防止除零
        sigma_expanded = sigma.expand_as(fcat)  # [B, 3N]
        
        # 特征归一化 - 分别计算每个模态的归一化
        # 由于f1_flat, f2_flat, f3_flat是分开的，我们需要分别计算它们的归一化
        f1_norm = (f1_flat - mu) / sigma  # [B, N]
        f2_norm = (f2_flat - mu) / sigma  # [B, N]  
        f3_norm = (f3_flat - mu) / sigma  # [B, N]
        
        # 高斯特征融合 - 将三个归一化后的特征相加
        fused_feature = f1_norm + f2_norm + f3_norm  # [B, N]
        
        return fused_feature

# ==================== 修改TripleInputConvNeXt类 ====================
class TripleInputConvNeXt(nn.Module):
    def __init__(self, in_chans=1, num_classes=2, depths=[1, 1, 3, 1], dims=[96, 192, 384, 768], drop_path_rate=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.features1 = ConvNeXt(in_chans, depths, dims, drop_path_rate, layer_scale_init_value)
        self.features2 = ConvNeXt(in_chans, depths, dims, drop_path_rate, layer_scale_init_value)
        self.features3 = ConvNeXt(in_chans, depths, dims, drop_path_rate, layer_scale_init_value)

        # 定义 1x1x1 卷积层，将不同的特征转换为 96 通道
        self.conv1x1 = nn.ModuleList([
            nn.Conv3d(in_channels=dims[-1], out_channels=96, kernel_size=1),
            nn.Conv3d(in_channels=dims[2], out_channels=96, kernel_size=1),
            nn.Conv3d(in_channels=dims[1], out_channels=96, kernel_size=1)
        ])

        # 用于上采样
        self.upsample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        
        # 新增：高斯对齐融合模块
        self.gaf = GaussianAlignmentFusion()
        
        # MLP for classification
        self.fusion = nn.Sequential(
            nn.Linear(1536, 512),  # 拼接后的维度 768 + 768
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )

    def ot_fusion(self, f1, f2, f3):
        """
        使用 Optimal Transport 进行特征融合
        """
        f1_flat = f1.view(f1.size(0), -1).detach().cpu().numpy()
        f2_flat = f2.view(f2.size(0), -1).detach().cpu().numpy()
        f3_flat = f3.view(f3.size(0), -1).detach().cpu().numpy()

        C31 = ot.dist(f3_flat, f1_flat, metric='sqeuclidean')
        C21 = ot.dist(f2_flat, f1_flat, metric='sqeuclidean')

        T31 = ot.emd([], [], C31)
        T21 = ot.emd([], [], C21)

        f3_trans = torch.matmul(torch.Tensor(T31).to(f3.device), f3)
        f2_trans = torch.matmul(torch.Tensor(T21).to(f2.device), f2)

        fused = f1 + f2_trans + f3_trans
        return fused

    def combined_fusion(self, f1, f2, f3):
        """
        结合OT融合和GAF融合
        """
        # OT融合
        ot_fused = self.ot_fusion(f1, f2, f3)
        
        # 将特征展平用于GAF
        f1_flat = f1.view(f1.size(0), -1)
        f2_flat = f2.view(f2.size(0), -1)
        f3_flat = f3.view(f3.size(0), -1)
        
        # GAF融合
        gaf_fused = self.gaf(f1_flat, f2_flat, f3_flat)
        
        # 将GAF融合后的特征reshape回原来的形状
        # 这里假设f1, f2, f3的形状相同，取f1的形状作为参考
        batch_size = f1.size(0)
        original_shape = f1.shape[1:]  # 去掉batch维度
        total_elements = np.prod(original_shape)
        
        # 如果GAF融合后的特征维度与原始特征维度不匹配，需要调整
        if gaf_fused.size(1) != total_elements:
            # 使用线性投影调整维度
            if not hasattr(self, 'gaf_proj'):
                self.gaf_proj = nn.Linear(gaf_fused.size(1), total_elements).to(gaf_fused.device)
            gaf_fused = self.gaf_proj(gaf_fused)
        
        gaf_fused = gaf_fused.view(batch_size, *original_shape)
        
        # 拼接OT融合和GAF融合的结果
        combined_fused = torch.cat([ot_fused, gaf_fused], dim=1)
        
        return combined_fused

    def bottom_up_fusion(self, f1_feats, f2_feats, f3_feats):
        """
        自下而上融合 - 使用combined_fusion
        """
        # 对每个stage都进行combined融合
        combined_features = []
        
        for i in range(4):
            f1 = f1_feats[i].mean([2, 3, 4])  # [B, C]
            f2 = f2_feats[i].mean([2, 3, 4])  # [B, C]  
            f3 = f3_feats[i].mean([2, 3, 4])  # [B, C]
            
            # 使用combined融合
            combined = self.combined_fusion(
                f1.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1),  # 添加空间维度
                f2.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1),
                f3.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            )
            
            # 全局平均池化
            pooled = F.adaptive_avg_pool3d(combined, output_size=1).view(combined.size(0), -1)
            combined_features.append(pooled)
        
        # 拼接所有stage的特征
        bottom_up_fused = torch.cat(combined_features, dim=1)
        
        return bottom_up_fused

    def top_down_path(self, features1, features2, features3):
        """
        实现自上而下的金字塔结构，每一层逐级上采样并特征融合。
        在每个stage都使用combined_fusion
        """
        top_down = self.conv1x1[0](features1[-1])  # conv1x1 for stage 4
        top_down1 = self.conv1x1[0](features2[-1])  # conv1x1 for stage 4
        top_down2 = self.conv1x1[0](features3[-1])  # conv1x1 for stage 4
        
        pooled_features = []

        # Stage 4
        combined_stage4 = self.combined_fusion(top_down, top_down1, top_down2)
        pooled_stage4 = F.adaptive_max_pool3d(combined_stage4, output_size=(1, 1, 1)).view(features1[-1].size(0), -1)
        pooled_features.append(pooled_stage4)

        # Stage 3
        top_down = self.upsample(top_down)
        top_down = F.interpolate(top_down, size=features1[-2].shape[2:], mode='trilinear', align_corners=True)
        top_down = top_down + self.conv1x1[1](features1[-2])

        top_down1 = self.upsample(top_down1)
        top_down1 = F.interpolate(top_down1, size=features2[-2].shape[2:], mode='trilinear', align_corners=True)
        top_down1 = top_down1 + self.conv1x1[1](features2[-2])

        top_down2 = self.upsample(top_down2)
        top_down2 = F.interpolate(top_down2, size=features3[-2].shape[2:], mode='trilinear', align_corners=True)
        top_down2 = top_down2 + self.conv1x1[1](features3[-2])

        combined_stage3 = self.combined_fusion(top_down, top_down1, top_down2)
        pooled_stage3 = F.adaptive_max_pool3d(combined_stage3, output_size=(1, 1, 1)).view(features1[-2].size(0), -1)
        pooled_features.append(pooled_stage3)

        # Stage 2
        top_down = self.upsample(top_down)
        top_down = F.interpolate(top_down, size=features1[-3].shape[2:], mode='trilinear', align_corners=True)
        top_down = top_down + self.conv1x1[2](features1[-3])

        top_down1 = self.upsample(top_down1)
        top_down1 = F.interpolate(top_down1, size=features2[-3].shape[2:], mode='trilinear', align_corners=True)
        top_down1 = top_down1 + self.conv1x1[2](features2[-3])

        top_down2 = self.upsample(top_down2)
        top_down2 = F.interpolate(top_down2, size=features3[-3].shape[2:], mode='trilinear', align_corners=True)
        top_down2 = top_down2 + self.conv1x1[2](features3[-3])

        combined_stage2 = self.combined_fusion(top_down, top_down1, top_down2)
        pooled_stage2 = F.adaptive_max_pool3d(combined_stage2, output_size=(1, 1, 1)).view(features1[-3].size(0), -1)
        pooled_features.append(pooled_stage2)

        # Stage 1
        top_down = self.upsample(top_down)
        top_down = top_down + features1[0]  # no conv1x1 for stage 1

        top_down1 = self.upsample(top_down1)
        top_down1 = top_down1 + features2[0]

        top_down2 = self.upsample(top_down2)
        top_down2 = top_down2 + features3[0]

        combined_stage1 = self.combined_fusion(top_down, top_down1, top_down2)
        pooled_stage1 = F.adaptive_max_pool3d(combined_stage1, output_size=(1, 1, 1)).view(features1[0].size(0), -1)
        pooled_features.append(pooled_stage1)

        # 汇总四个 stage 的 pooled 特征
        final_pooled = torch.sum(torch.stack(pooled_features), dim=0)
        
        return final_pooled

    def forward(self, x1, x2, x3):
        f1_feats = self.features1(x1)
        f2_feats = self.features2(x2)
        f3_feats = self.features3(x3)

        # 自下而上 combined 融合后的最终特征
        bottom_up_feature = self.bottom_up_fusion(f1_feats, f2_feats, f3_feats)

        # 自上而下金字塔路径
        top_down_feature = self.top_down_path(f1_feats, f2_feats, f3_feats)

        # 拼接自下而上和自上而下的特征
        final_feature = torch.cat([bottom_up_feature, top_down_feature], dim=1)
        
        # 调整维度以匹配分类器
        if final_feature.size(1) != 1536:
            if not hasattr(self, 'dim_adjust'):
                self.dim_adjust = nn.Linear(final_feature.size(1), 1536).to(final_feature.device)
            final_feature = self.dim_adjust(final_feature)

        # 分类输出
        output = self.fusion(final_feature)
        return output

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss_contrastive

def main():
    set_seed(19)  # Set random seed here
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    num_classes = 2
    gradient_accumulation_steps = 64

    # 数据集和数据加载器
    train_dataset = TripleNiftiDataset(
        root_dir1='/data/xukaiqiang/data/mwc1npytrain',
        root_dir2='/data/xukaiqiang/data/mwc2npytrain',
        root_dir3='/data/xukaiqiang/data/mwc3npytrain'
    )
    val_dataset = TripleNiftiDataset(
        root_dir1='/data/xukaiqiang/data/mwc1npyval',
        root_dir2='/data/xukaiqiang/data/mwc2npyval',
        root_dir3='/data/xukaiqiang/data/mwc3npyval'
    )
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    validate_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    model = TripleInputConvNeXt(in_chans=1, num_classes=2, depths=[1, 1, 3, 1], dims=[96, 192, 384, 768], drop_path_rate=0.0, layer_scale_init_value=1e-6).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    best_val_acc = 0
    no_improvement_count = 0

    for epoch in range(200):
        model.train()
        running_loss = 0.0
        train_correct = 0
        train_total = 0
        all_train_targets = []
        all_train_predictions = []

        # 训练循环
        for step, ((images1, images2, images3), labels) in enumerate(train_loader):
            images1, images2, images3, labels = images1.to(device), images2.to(device), images3.to(device), labels.to(device)
            outputs = model(images1, images2, images3)
            loss = criterion(outputs, labels)
            loss /= gradient_accumulation_steps
            loss.backward()

            if (step + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                running_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            all_train_targets.extend(labels.cpu().numpy())
            all_train_predictions.extend(predicted.cpu().numpy())

        train_accuracy = 100 * train_correct / train_total
        train_loss = running_loss / len(train_loader.dataset)
        train_cm = confusion_matrix(all_train_targets, all_train_predictions)
        print(f"Epoch {epoch + 1}/{200}, Train Loss: {train_loss:.8f}, Train Acc: {train_accuracy:.2f}%")
        print("Train Confusion Matrix:")
        print(train_cm)

        # 验证循环
        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0
        all_val_targets = []
        all_val_predictions = []

        with torch.no_grad():
            for (images1, images2, images3), labels in validate_loader:
                images1, images2, images3, labels = images1.to(device), images2.to(device), images3.to(device), labels.to(device)
                outputs = model(images1, images2, images3)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                all_val_targets.extend(labels.cpu().numpy())
                all_val_predictions.extend(predicted.cpu().numpy())

        val_loss = val_running_loss / len(validate_loader.dataset)
        val_accuracy = 100 * val_correct / val_total
        if np.unique(all_val_predictions).size > 1:  # 确保有多个类别被预测
            precision = precision_score(all_val_targets, all_val_predictions, average='binary')
            recall = recall_score(all_val_targets, all_val_predictions, average='binary')
            f1 = f1_score(all_val_targets, all_val_predictions, average='binary')
            mcc = matthews_corrcoef(all_val_targets, all_val_predictions)
            print(f"Validation Loss: {val_loss:.8f}, Validation Acc: {val_accuracy:.4f}%, Precision: {precision:.2f}, Recall: {recall:.2f}, F1: {f1:.2f}, MCC: {mcc:.2f}")
        else:
            print(f"Validation Loss: {val_loss:.8f}, Validation Acc: {val_accuracy:.4f}%, Insufficient class diversity for precision/recall calculations")
        print("Validation Confusion Matrix:")
        print(confusion_matrix(all_val_targets, all_val_predictions))

        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            no_improvement_count = 0
        else:
            no_improvement_count += 1

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        print(f"Learning rate adjusted to: {current_lr}")

        if no_improvement_count >= 200:
            print(f"Early stopping at epoch {epoch + 1}")
            break

if __name__ == '__main__':
    main()