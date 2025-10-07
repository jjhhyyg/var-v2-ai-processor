"""
AI处理模块主应用
基于Flask实现的视频分析服务
"""
from flask import Flask, request, jsonify
import logging
import torch
import threading
from config import Config
from analyzer.video_processor import VideoAnalyzer
from mq_consumer import RabbitMQConsumer

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 创建Flask应用
app = Flask(__name__)
app.config.from_object(Config)

# 全局变量存储分析器实例和MQ消费者
analyzer = None
mq_consumer = None


def init_analyzer():
    """初始化视频分析器"""
    global analyzer
    if analyzer is None:
        try:
            analyzer = VideoAnalyzer(
                model_path=Config.MODEL_PATH,
                device=Config.DEVICE
            )
            logger.info("Video analyzer initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize video analyzer: {e}")
            raise


def init_mq_consumer():
    """初始化并启动MQ消费者"""
    global mq_consumer
    if mq_consumer is None:
        try:
            mq_consumer = RabbitMQConsumer()
            mq_consumer.init_analyzer()
            mq_consumer.connect()
            
            # 在后台线程中启动消费者
            consumer_thread = threading.Thread(
                target=mq_consumer.start_consuming,
                daemon=True,
                name="MQ-Consumer"
            )
            consumer_thread.start()
            
            logger.info("RabbitMQ consumer started successfully")
        except Exception as e:
            logger.error(f"Failed to initialize MQ consumer: {e}")
            logger.warning("MQ consumer not available, will only use HTTP API")



@app.route('/health', methods=['GET'])
def health_check():
    """
    健康检查接口
    返回服务状态、模型加载状态、加速器可用性等信息
    """
    try:
        # 确保分析器已初始化
        if analyzer is None:
            init_analyzer()

        # 检测可用的加速器
        cuda_available = torch.cuda.is_available()
        mps_available = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()

        device_info = {
            'cuda': {
                'available': cuda_available,
                'name': torch.cuda.get_device_name(0) if cuda_available else None
            },
            'mps': {
                'available': mps_available,
                'name': 'Apple Metal Performance Shaders' if mps_available else None
            }
        }

        return jsonify({
            'status': 'healthy',
            'model_loaded': analyzer is not None,
            'model_version': Config.MODEL_VERSION,
            'device_info': device_info,
            'current_device': str(analyzer.device) if analyzer else 'unknown',
            'version': '1.0.0'
        }), 200
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({
            'status': 'unhealthy',
            'error': str(e)
        }), 500


@app.route('/api/analyze', methods=['POST'])
def analyze_video():
    """
    视频分析接口
    接收任务信息，异步处理视频分析
    """
    try:
        # 解析请求数据
        data = request.get_json()

        required_fields = ['taskId', 'videoPath', 'videoDuration', 'timeoutThreshold']
        for field in required_fields:
            if field not in data:
                return jsonify({
                    'status': 'error',
                    'message': f'Missing required field: {field}'
                }), 400

        task_id = data['taskId']
        video_path = data['videoPath']
        video_duration = data['videoDuration']
        timeout_threshold = data['timeoutThreshold']
        config = data.get('config', {})

        logger.info(f"Received analysis request for task {task_id}")
        
        # 将相对路径转换为绝对路径
        absolute_video_path = Config.resolve_path(video_path)
        logger.info(f"Task {task_id}: Resolved video path from '{video_path}' to '{absolute_video_path}'")

        # 确保分析器已初始化
        if analyzer is None:
            init_analyzer()

        # 提取配置参数
        confidence_threshold = config.get('confidenceThreshold', Config.DEFAULT_CONFIDENCE_THRESHOLD)
        iou_threshold = config.get('iouThreshold', Config.DEFAULT_IOU_THRESHOLD)

        # 启动异步分析（在生产环境中应该使用Celery或类似的任务队列）
        # 这里为了简化，直接在后台线程中执行
        import threading
        thread = threading.Thread(
            target=analyzer.analyze_video_task,
            args=(task_id, absolute_video_path, video_duration, timeout_threshold,
                  confidence_threshold, iou_threshold),
            daemon=True
        )
        thread.start()

        return jsonify({
            'status': 'accepted',
            'taskId': task_id,
            'message': '任务已接受，开始处理'
        }), 202

    except Exception as e:
        logger.error(f"Failed to process analysis request: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@app.errorhandler(404)
def not_found(error):
    """404错误处理"""
    return jsonify({
        'status': 'error',
        'message': 'Endpoint not found'
    }), 404


@app.errorhandler(500)
def internal_error(error):
    """500错误处理"""
    logger.error(f"Internal server error: {error}")
    return jsonify({
        'status': 'error',
        'message': 'Internal server error'
    }), 500


@app.route('/api/export-video', methods=['POST'])
def export_annotated_video():
    """
    导出带标注的视频接口
    生成包含bbox、标签和ID的结果视频
    """
    try:
        # 解析请求数据
        data = request.get_json()

        required_fields = ['taskId', 'videoPath', 'outputPath']
        for field in required_fields:
            if field not in data:
                return jsonify({
                    'status': 'error',
                    'message': f'Missing required field: {field}'
                }), 400

        task_id = data['taskId']
        video_path = data['videoPath']
        output_path = data['outputPath']
        config = data.get('config', {})

        logger.info(f"Received export request for task {task_id}")
        
        # 将相对路径转换为绝对路径
        absolute_video_path = Config.resolve_path(video_path)
        logger.info(f"Task {task_id}: Resolved video path from '{video_path}' to '{absolute_video_path}'")

        # 确保分析器已初始化
        if analyzer is None:
            init_analyzer()

        # 提取配置参数
        confidence_threshold = config.get('confidenceThreshold', Config.DEFAULT_CONFIDENCE_THRESHOLD)
        iou_threshold = config.get('iouThreshold', Config.DEFAULT_IOU_THRESHOLD)

        # 异步导出视频
        import threading
        def export_task():
            success = analyzer.export_annotated_video(
                task_id, absolute_video_path, output_path,
                confidence_threshold, iou_threshold
            )
            if success:
                logger.info(f"Task {task_id}: Video export completed successfully")

                # 计算相对于 codes/ 目录的相对路径
                import os as os_module
                abs_output_path = os_module.path.abspath(output_path)
                relative_path = Config.to_relative_path(abs_output_path)

                # 通知后端更新结果视频路径
                try:
                    import requests
                    update_url = f"{Config.BACKEND_BASE_URL}/api/tasks/{task_id}/result-video"
                    response = requests.put(
                        update_url,
                        json={'resultVideoPath': relative_path},
                        timeout=10
                    )
                    if response.status_code == 200:
                        logger.info(f"Task {task_id}: Result video path updated in backend: {relative_path}")
                    else:
                        logger.warning(f"Task {task_id}: Failed to update result video path: {response.status_code}")
                except Exception as e:
                    logger.error(f"Task {task_id}: Failed to notify backend: {e}")
            else:
                logger.error(f"Task {task_id}: Video export failed")

        thread = threading.Thread(target=export_task, daemon=True)
        thread.start()

        return jsonify({
            'status': 'accepted',
            'taskId': task_id,
            'message': '视频导出任务已开始'
        }), 202

    except Exception as e:
        logger.error(f"Failed to process export request: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


if __name__ == '__main__':
    logger.info(f"Starting AI processor on {Config.FLASK_HOST}:{Config.FLASK_PORT}")
    logger.info(f"Model: {Config.MODEL_PATH}")
    logger.info(f"Device: {Config.DEVICE if Config.DEVICE else 'auto'}")

    # 初始化分析器
    try:
        init_analyzer()
    except Exception as e:
        logger.error(f"Failed to initialize analyzer: {e}")
        exit(1)

    # 初始化并启动MQ消费者
    try:
        init_mq_consumer()
    except Exception as e:
        logger.warning(f"MQ consumer initialization failed: {e}")
        # 不退出，仍然可以使用HTTP API

    # 启动Flask应用
    app.run(
        host=Config.FLASK_HOST,
        port=Config.FLASK_PORT,
        debug=Config.DEBUG,
        threaded=True
    )
