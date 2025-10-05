# 物体类别映射说明

## 概述

AI处理器返回的 `trackingObjects` 中的 `category` 字段与后端 `ObjectCategory` 枚举完全对应。

## 映射关系

| YOLO类别ID | 中文名称 | AI返回的category | 后端枚举 | 后端描述 |
|-----------|---------|-----------------|---------|---------|
| 0 | 熔池未到边 | `POOL_NOT_REACHED` | `ObjectCategory.POOL_NOT_REACHED` | 熔池未到边 |
| 1 | 粘连物 | `ADHESION` | `ObjectCategory.ADHESION` | 粘连物 |
| 2 | 锭冠 | `CROWN` | `ObjectCategory.CROWN` | 锭冠 |
| 3 | 辉光 | `GLOW` | `ObjectCategory.GLOW` | 辉光 |
| 4 | 边弧（侧弧） | `SIDE_ARC` | `ObjectCategory.SIDE_ARC` | 边弧 |
| 5 | 爬弧 | `CREEPING_ARC` | `ObjectCategory.CREEPING_ARC` | 爬弧 |

## 配置位置

映射配置在 `config.py` 中的 `CATEGORY_MAPPING` 字典：

```python
CATEGORY_MAPPING = {
    '熔池未到边': 'POOL_NOT_REACHED',
    '粘连物': 'ADHESION',
    '锭冠': 'CROWN',
    '辉光': 'GLOW',
    '边弧（侧弧）': 'SIDE_ARC',
    '爬弧': 'CREEPING_ARC'
}
```

## 返回数据示例

```json
{
  "trackingObjects": [
    {
      "objectId": 1,
      "category": "ADHESION",
      "className": "粘连物",
      "classId": 1,
      "firstFrame": 100,
      "lastFrame": 250,
      "firstTime": 3.33,
      "lastTime": 8.33,
      "duration": 151
    },
    {
      "objectId": 2,
      "category": "GLOW",
      "className": "辉光",
      "classId": 3,
      "firstFrame": 50,
      "lastFrame": 120,
      "firstTime": 1.67,
      "lastTime": 4.0,
      "duration": 71
    }
  ]
}
```

## 后端验证

后端会使用以下方法验证类别：

```java
ObjectCategory category = ObjectCategory.valueOf(trackingObject.getCategory());
// 或
ObjectCategory category = ObjectCategory.fromClassId(trackingObject.getClassId());
```

两种方式都能正确映射到后端枚举。

## 注意事项

1. **category 字段必填**：后端使用 `@NotBlank` 验证，不能为空
2. **必须是有效枚举值**：category 必须是后端 `ObjectCategory` 枚举中定义的值
3. **classId 对应关系**：classId 必须与 category 对应，否则后端验证会失败
4. **className 仅供显示**：中文名称仅用于前端显示，后端主要使用 category 和 classId
