# AI处理模块变更日志

## [v1.0.0] - 2025-10-03

### 新增功能

#### 核心功能
- ✅ 基于Ultralytics YOLO的目标检测
- ✅ ByteTrack多目标追踪
- ✅ 自动事件检测和推断
- ✅ 动态参数计算（假数据接口）
- ✅ 实时进度回调
- ✅ 超时检测和预警

#### 模块实现
- ✅ `analyzer/yolo_tracker.py` - YOLO检测和追踪
- ✅ `analyzer/event_detector.py` - 事件检测
- ✅ `analyzer/metrics_calculator.py` - 动态参数计算
- ✅ `analyzer/video_processor.py` - 视频处理主逻辑
- ✅ `utils/callback.py` - 后端回调工具

#### 配置和部署
- ✅ 完整的环境变量配置（.env.example）
- ✅ ByteTrack配置文件（bytetrack.yaml）
- ✅ 启动脚本（start.sh）
- ✅ 依赖管理（requirements.txt）

#### 文档
- ✅ README.md - 项目介绍和快速开始
- ✅ USAGE.md - 详细使用指南
- ✅ CHANGELOG.md - 变更日志

### 技术栈

- **Web框架**: Flask 3.0.0
- **深度学习**: PyTorch 2.0+, Ultralytics 8.0+
- **计算机视觉**: OpenCV 4.8+
- **追踪算法**: ByteTrack
- **设备支持**: CUDA, MPS, CPU

### 支持的事件类型

1. **粘连物相关**
   - 电极形成粘连物（ADHESION_FORMED）
   - 电极粘连物脱落（ADHESION_DROPPED）

2. **锭冠相关**
   - 锭冠脱落（CROWN_DROPPED）

3. **电弧异常**
   - 辉光（GLOW）
   - 边弧/侧弧（SIDE_ARC）
   - 爬弧（CLIMBING_ARC）

4. **熔池状态**
   - 熔池未到边（POOL_NOT_EDGE）

### 已知限制

- 动态参数计算使用假数据（需后续实现真实算法）
- 粘连物脱落位置判断使用简化逻辑
- 锭冠脱落判断使用简化逻辑

### 与Demo代码的一致性

✅ 完全兼容demo代码的：
- Tensor到NumPy的转换逻辑
- 边界框格式（xyxy）
- 检测结果数据结构
- ByteTrack配置参数
- 追踪结果输出格式

### 未来计划

- [ ] 实现真实的动态参数计算算法
- [ ] 优化事件推断逻辑
- [ ] 添加WebSocket实时推送
- [ ] 支持批量视频处理
- [ ] 添加模型热更新功能
- [ ] 性能优化和并发处理

---

## 更新说明

基于MOT demo代码（`track_video.py`）完善：

1. **配置文件更新**
   - `config.py`: 添加ByteTrack详细参数配置
   - `.env.example`: 添加完整的环境变量说明
   - `bytetrack.yaml`: 标准ByteTrack配置文件

2. **追踪模块改进**
   - `yolo_tracker.py`: 
     - 改进Tensor到NumPy转换逻辑（兼容demo代码）
     - 边界框格式统一为xyxy
     - 添加center_x, center_y, width, height字段
     - 使用安全的属性检查

3. **事件检测优化**
   - `event_detector.py`:
     - 修正bbox格式处理（xyxy）
     - 更新中心点计算逻辑
     - 修正轨迹分析算法

4. **文档完善**
   - 添加详细的使用指南（USAGE.md）
   - 创建启动脚本（start.sh）
   - 完善README文档

### 兼容性说明

本次更新确保与demo代码的完全兼容：
- ✅ 使用相同的模型路径约定（weights/best.pt）
- ✅ 使用相同的tracker配置文件（bytetrack.yaml）
- ✅ 使用相同的数据结构和格式
- ✅ 支持相同的参数配置方式

### 测试建议

1. 使用demo代码中的模型和配置进行测试
2. 对比demo代码和AI模块的检测结果
3. 验证追踪ID的连续性
4. 检查事件推断的准确性
