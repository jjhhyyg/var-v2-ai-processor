# 类别名称映射问题修复

## 问题描述

后端报错：

```
No enum constant ustb.hyy.app.backend.domain.enums.ObjectCategory.电极粘连物
```

## 原因分析

YOLO模型训练时使用的类别名称与AI处理器配置中定义的类别名称不一致：

- **配置中的名称**：`粘连物`
- **YOLO模型返回的名称**：`电极粘连物`

当AI处理器将检测结果发送给后端时，后端尝试将字符串"电极粘连物"转换为`ObjectCategory`枚举，但该枚举中没有对应的值，导致错误。

## 解决方案

在 `ai-processor/config.py` 中的 `CATEGORY_MAPPING` 和 `EVENT_TYPE_MAPPING` 添加了对"电极粘连物"的支持：

### 修改前

```python
CATEGORY_MAPPING = {
    '粘连物': 'ADHESION',
    # ... 其他映射
}

EVENT_TYPE_MAPPING = {
    '形成粘连物': 'ADHESION_FORMED',
    '粘连物脱落': 'ADHESION_DROPPED',
    # ... 其他映射
}
```

### 修改后

```python
CATEGORY_MAPPING = {
    '粘连物': 'ADHESION',
    '电极粘连物': 'ADHESION',  # 新增：支持YOLO模型返回的名称
    '边弧': 'SIDE_ARC',         # 新增：支持简化名称
    '侧弧': 'SIDE_ARC',         # 新增：支持别名
    # ... 其他映射
}

EVENT_TYPE_MAPPING = {
    '形成粘连物': 'ADHESION_FORMED',
    '粘连物脱落': 'ADHESION_DROPPED',
    '电极形成粘连物': 'ADHESION_FORMED',   # 新增
    '电极粘连物脱落': 'ADHESION_DROPPED', # 新增
    '边弧': 'SIDE_ARC',                       # 新增
    '侧弧': 'SIDE_ARC',                       # 新增
    # ... 其他映射
}
```

## 映射逻辑

1. YOLO模型检测到物体时，返回类别名称（如"电极粘连物"）
2. `event_detector.py` 使用 `CATEGORY_MAPPING` 将中文名称转换为英文枚举值
3. 转换后的值（如"ADHESION"）发送给后端
4. 后端将字符串转换为 `ObjectCategory.ADHESION` 枚举

## 预防措施

### 1. 容错处理

如果遇到未映射的类别名称，系统会：

- 使用原始名称作为fallback
- 记录警告日志
- 不会中断处理流程

### 2. 映射覆盖

为常见的类别变体添加了多个映射：

- 完整名称：`电极粘连物` → `ADHESION`
- 简化名称：`粘连物` → `ADHESION`
- 简化名称：`边弧` → `SIDE_ARC`
- 别名：`侧弧` → `SIDE_ARC`

### 3. 文档说明

在 `CLASS_NAMES` 定义中添加了注释，说明实际模型可能返回不同的名称。

## 后端枚举定义

确保后端的 `ObjectCategory` 枚举与映射匹配：

```java
public enum ObjectCategory {
    POOL_NOT_REACHED(0, "熔池未到边"),
    ADHESION(1, "粘连物"),           // 对应"粘连物"或"电极粘连物"
    CROWN(2, "锭冠"),
    GLOW(3, "辉光"),
    SIDE_ARC(4, "边弧"),            // 对应"边弧"、"侧弧"或"边弧（侧弧）"
    CREEPING_ARC(5, "爬弧");
}
```

## 验证

修复后需要验证：

1. 上传包含粘连物的视频
2. 检查AI处理器日志，确认类别名称正确映射
3. 检查后端日志，确认没有枚举转换错误
4. 查看前端任务结果，确认追踪物体显示正确

## 未来改进

### 建议1：统一命名

与模型训练团队协调，统一类别命名规范，避免多个变体。

### 建议2：动态映射

考虑从YOLO模型元数据中读取类别名称，自动建立映射关系。

### 建议3：配置外部化

将类别映射配置移到独立的配置文件中，便于维护和更新。
