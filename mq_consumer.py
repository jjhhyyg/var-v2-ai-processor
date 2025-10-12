"""
RabbitMQ消息队列消费者
从video_analysis_queue队列消费分析任务
"""
import json
import logging
import pika
import threading
from config import Config
from analyzer.video_processor import VideoAnalyzer

logger = logging.getLogger(__name__)


class RabbitMQConsumer:
    """RabbitMQ消费者"""

    def __init__(self):
        self.connection = None
        self.channel = None
        # 移除共享的分析器实例，改为每个任务独立创建
        # self.analyzer = None

        # RabbitMQ连接参数
        self.rabbitmq_host = Config.RABBITMQ_HOST
        self.rabbitmq_port = Config.RABBITMQ_PORT
        self.rabbitmq_user = Config.RABBITMQ_USER
        self.rabbitmq_password = Config.RABBITMQ_PASSWORD
        self.queue_name = Config.RABBITMQ_VIDEO_ANALYSIS_QUEUE
        
        # 并发控制：使用信号量限制最大并发数
        self.max_concurrent_tasks = 3
        self.semaphore = threading.Semaphore(self.max_concurrent_tasks)
        self.active_tasks = 0
        self.active_tasks_lock = threading.Lock()

        # 优雅关闭机制
        self.shutdown_event = threading.Event()
        self.active_threads = []
        self.threads_lock = threading.Lock()

    def init_analyzer(self):
        """
        初始化视频分析器（已弃用）
        现在每个任务创建独立的分析器实例
        保留此方法以保持向后兼容性
        """
        logger.warning("init_analyzer() is deprecated. Each task now creates its own analyzer instance.")
        pass

    def connect(self):
        """连接到RabbitMQ"""
        try:
            credentials = pika.PlainCredentials(
                self.rabbitmq_user,
                self.rabbitmq_password
            )
            parameters = pika.ConnectionParameters(
                host=self.rabbitmq_host,
                port=self.rabbitmq_port,
                credentials=credentials,
                heartbeat=600,
                blocked_connection_timeout=300
            )

            self.connection = pika.BlockingConnection(parameters)
            self.channel = self.connection.channel()

            # 声明队列
            self.channel.queue_declare(
                queue=self.queue_name,
                durable=True  # 队列持久化
            )

            logger.info(f"Connected to RabbitMQ at {self.rabbitmq_host}:{self.rabbitmq_port}")
            logger.info(f"Listening on queue: {self.queue_name}")

        except Exception as e:
            logger.error(f"Failed to connect to RabbitMQ: {e}")
            raise

    def callback(self, ch, method, properties, body):
        """处理消息回调"""
        try:
            # 解析消息
            message = json.loads(body.decode('utf-8'))
            logger.info(f"Received message: {message}")

            task_id = message.get('taskId')
            video_path = message.get('videoPath')
            video_duration = message.get('videoDuration')
            timeout_threshold = message.get('timeoutThreshold')
            callback_url = message.get('callbackUrl')
            config = message.get('config', {})

            # 验证必需字段
            if not all([task_id, video_path, video_duration, timeout_threshold]):
                logger.error(f"Missing required fields in message: {message}")
                ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
                return
            
            # 将相对路径转换为绝对路径
            # 后端传来的是相对于codes/目录的路径，需要转换为绝对路径
            absolute_video_path = Config.resolve_path(video_path)
            logger.info(f"Task {task_id}: Resolved video path from '{video_path}' to '{absolute_video_path}'")

            # 使用.env中配置的参数，忽略消息中的配置
            confidence_threshold = Config.DEFAULT_CONFIDENCE_THRESHOLD
            iou_threshold = Config.DEFAULT_IOU_THRESHOLD

            # 获取预处理配置
            enable_preprocessing = config.get('enablePreprocessing', False)
            preprocessing_strength = config.get('preprocessingStrength', 'moderate')
            preprocessing_enhance_pool = config.get('preprocessingEnhancePool', True)

            # 获取追踪合并配置
            enable_tracking_merge = config.get('enableTrackingMerge', True)
            tracking_merge_strategy = config.get('trackingMergeStrategy', 'auto')

            # 从后端配置中获取视频帧率（由FFmpeg解析得到）
            frame_rate = config.get('frameRate', 25.0)
            logger.info(f"Task {task_id}: Using frame rate from backend config: {frame_rate} fps")

            # 定义处理任务的函数
            def process_task():
                # 检查是否需要关闭
                if self.shutdown_event.is_set():
                    logger.info(f"Task {task_id}: Skipped due to shutdown")
                    return

                # 获取信号量并执行任务（确保异常安全）
                # 将 acquire 放在 try 外面，因为 acquire 本身不会失败
                # 但用 try-finally 确保信号量一定被释放
                self.semaphore.acquire(blocking=True)
                try:
                    # 更新活跃任务计数
                    with self.active_tasks_lock:
                        self.active_tasks += 1
                        logger.info(f"Task {task_id}: Started (Active tasks: {self.active_tasks}/{self.max_concurrent_tasks})")

                    # 为每个任务创建独立的视频分析器实例
                    task_analyzer = VideoAnalyzer(
                        model_path=Config.MODEL_PATH,
                        device=Config.DEVICE
                    )
                    logger.info(f"Task {task_id}: Created independent analyzer instance")

                    # 获取模型版本并更新到后端
                    try:
                        model_version = task_analyzer.yolo_tracker.model_version
                        logger.info(f"Task {task_id}: Model version: {model_version}")
                        
                        # 向后端更新模型版本
                        import requests
                        update_url = f"{Config.BACKEND_BASE_URL}/api/tasks/{task_id}/model-version"
                        response = requests.put(
                            update_url,
                            json={'modelVersion': model_version},
                            timeout=10
                        )
                        if response.status_code == 200:
                            logger.info(f"Task {task_id}: Model version updated in backend: {model_version}")
                        else:
                            logger.warning(f"Task {task_id}: Failed to update model version: {response.status_code}")
                    except Exception as e:
                        logger.error(f"Task {task_id}: Failed to update model version: {e}")

                    # 1. 执行视频分析（包含预处理）
                    # 返回值：(任务状态, 实际分析的视频路径)
                    analysis_status, analyzed_video_path = task_analyzer.analyze_video_task(
                        task_id, absolute_video_path, video_duration, timeout_threshold,
                        confidence_threshold, iou_threshold,
                        enable_preprocessing, preprocessing_strength, preprocessing_enhance_pool,
                        enable_tracking_merge, tracking_merge_strategy,
                        callback_url,
                        frame_rate  # 传递从后端获取的帧率
                    )
                    
                    logger.info(f"Task {task_id}: Analysis finished with status: {analysis_status}")
                    logger.info(f"Task {task_id}: Analyzed video path: {analyzed_video_path}")
                    
                    # 2. 生成结果视频（仅当分析成功完成时）
                    # 重要：使用实际分析的视频路径（analyzed_video_path），而不是原始视频路径
                    # 这样可以确保结果视频基于与AI分析相同的视频生成，保持检测框的准确性
                    if analysis_status in ['COMPLETED', 'COMPLETED_TIMEOUT']:
                        import os
                        from utils.filename_utils import add_or_update_timestamp, extract_base_name
                        
                        # 使用推荐的 get_storage_path() 方法获取结果视频目录
                        result_video_dir = Config.get_storage_path(Config.STORAGE_RESULT_VIDEOS_SUBDIR)
                        os.makedirs(result_video_dir, exist_ok=True)
                        
                        # 生成输出文件名（提取基础名称，添加_result后缀和时间戳）
                        video_filename = os.path.basename(analyzed_video_path)
                        
                        # 提取原始基础名称（去掉时间戳和 _preprocessed 后缀）
                        base_name = extract_base_name(video_filename, remove_suffixes=['_preprocessed'])
                        
                        # 生成结果视频文件名：基础名_result，然后添加时间戳
                        base_output_filename = f"{base_name}_result.mp4"
                        output_filename = os.path.basename(add_or_update_timestamp(base_output_filename, update_existing=True))
                        output_path = os.path.join(result_video_dir, output_filename)
                        
                        logger.info(f"Task {task_id}: Starting result video generation")
                        logger.info(f"Task {task_id}: Using analyzed video: {analyzed_video_path}")
                        success = task_analyzer.export_annotated_video(
                            task_id, analyzed_video_path, output_path,
                            confidence_threshold, iou_threshold, callback_url,
                            frame_rate  # 传递帧率
                        )
                        
                        if success:
                            logger.info(f"Task {task_id}: Result video generated successfully")

                            # 计算相对于 codes/ 目录的相对路径
                            # 数据库存储格式: storage/result_videos/xxx.mp4 (相对于codes/)
                            abs_output_path = os.path.abspath(output_path)
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
                                logger.error(f"Task {task_id}: Failed to notify backend about result video: {e}")
                        else:
                            logger.error(f"Task {task_id}: Result video generation failed")
                    elif analysis_status == 'PAUSED':
                        logger.info(f"Task {task_id}: Task was paused, skipping result video generation")
                    elif analysis_status == 'FAILED':
                        logger.error(f"Task {task_id}: Task failed during analysis, skipping result video generation")
                    else:
                        logger.warning(f"Task {task_id}: Unknown analysis status '{analysis_status}', skipping result video generation")
                        
                finally:
                    # 释放信号量并更新活跃任务计数
                    with self.active_tasks_lock:
                        self.active_tasks -= 1
                        logger.info(f"Task {task_id}: Completed (Active tasks: {self.active_tasks}/{self.max_concurrent_tasks})")
                    
                    self.semaphore.release()

            # 在后台线程中执行分析（避免阻塞消费者）
            # 使用非daemon线程确保任务完整性
            thread = threading.Thread(
                target=process_task,
                daemon=False,
                name=f"Task-{task_id}"
            )

            # 添加到活跃线程列表（用于优雅关闭）
            with self.threads_lock:
                self.active_threads.append(thread)

            thread.start()

            # 确认消息（任务已接受）
            ch.basic_ack(delivery_tag=method.delivery_tag)
            logger.info(f"Task {task_id} accepted and processing started")

        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode message: {e}")
            ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
        except Exception as e:
            logger.error(f"Error processing message: {e}", exc_info=True)

            # 智能错误重试策略：区分可重试和不可重试的错误
            # 不可重试的错误（永久性失败）
            NON_RETRIABLE_ERRORS = (
                FileNotFoundError,      # 文件不存在
                ValueError,             # 参数错误
                KeyError,              # 缺少必需字段
                TypeError,             # 类型错误
            )

            # 可重试的错误（临时性失败）
            RETRIABLE_ERRORS = (
                ConnectionError,        # 网络连接错误
                TimeoutError,          # 超时错误
                IOError,               # IO错误（可能是临时的）
                OSError,               # 系统错误（可能是临时的）
            )

            # 判断是否应该重试
            should_requeue = False

            if isinstance(e, NON_RETRIABLE_ERRORS):
                logger.error(f"Non-retriable error detected: {type(e).__name__}, message will not be requeued")
                should_requeue = False
            elif isinstance(e, RETRIABLE_ERRORS):
                logger.warning(f"Retriable error detected: {type(e).__name__}, message will be requeued")
                should_requeue = True
            else:
                # 未知错误，默认不重试（避免无限循环）
                logger.error(f"Unknown error type: {type(e).__name__}, message will not be requeued by default")
                should_requeue = False

            ch.basic_nack(delivery_tag=method.delivery_tag, requeue=should_requeue)

    def start_consuming(self):
        """开始消费消息"""
        try:
            # 设置预取数量为最大并发数，允许队列预先分配任务
            # 但实际并发由信号量控制
            self.channel.basic_qos(prefetch_count=self.max_concurrent_tasks)

            logger.info(f"Maximum concurrent tasks set to: {self.max_concurrent_tasks}")

            # 开始消费
            self.channel.basic_consume(
                queue=self.queue_name,
                on_message_callback=self.callback
            )

            logger.info("Started consuming messages. Press Ctrl+C to exit.")
            self.channel.start_consuming()

        except KeyboardInterrupt:
            logger.info("Stopping consumer...")
            self.stop()
        except Exception as e:
            logger.error(f"Error in consumer: {e}")
            self.stop()
            raise

    def shutdown(self, timeout: float = 30.0):
        """
        优雅关闭：等待所有活跃任务完成

        Args:
            timeout: 每个任务的等待超时时间（秒）
        """
        logger.info("Initiating graceful shutdown...")

        # 设置关闭标志，不再接受新任务
        self.shutdown_event.set()

        # 等待所有活跃线程完成
        with self.threads_lock:
            active_count = len(self.active_threads)
            if active_count == 0:
                logger.info("No active tasks to wait for")
                return

            logger.info(f"Waiting for {active_count} active tasks to complete (timeout: {timeout}s per task)...")

        completed = 0
        failed = 0

        for thread in self.active_threads[:]:  # 复制列表，避免迭代时修改
            try:
                thread.join(timeout=timeout)
                if thread.is_alive():
                    logger.warning(f"Thread {thread.name} did not finish within timeout")
                    failed += 1
                else:
                    completed += 1
            except Exception as e:
                logger.error(f"Error waiting for thread {thread.name}: {e}")
                failed += 1

        logger.info(f"Shutdown complete: {completed} tasks completed, {failed} tasks timed out")

        # 清空线程列表
        with self.threads_lock:
            self.active_threads.clear()

    def stop(self):
        """停止消费者"""
        try:
            # 先优雅关闭所有任务
            self.shutdown(timeout=30.0)

            # 然后关闭RabbitMQ连接
            if self.channel and self.channel.is_open:
                self.channel.stop_consuming()
                self.channel.close()

            if self.connection and self.connection.is_open:
                self.connection.close()

            logger.info("RabbitMQ consumer stopped")
        except Exception as e:
            logger.error(f"Error stopping consumer: {e}")


def main():
    """主函数"""
    # 使用统一的日志级别配置
    logging.basicConfig(
        level=getattr(logging, Config.LOG_LEVEL, logging.INFO),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger.info(f"日志级别设置为: {Config.LOG_LEVEL}")

    consumer = RabbitMQConsumer()

    try:
        # 初始化分析器
        consumer.init_analyzer()

        # 连接到RabbitMQ
        consumer.connect()

        # 开始消费消息
        consumer.start_consuming()

    except Exception as e:
        logger.error(f"Fatal error: {e}")
        exit(1)


if __name__ == '__main__':
    main()
