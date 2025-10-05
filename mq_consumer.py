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
        self.queue_name = Config.RABBITMQ_QUEUE
        
        # 并发控制：使用信号量限制最大并发数
        self.max_concurrent_tasks = 3
        self.semaphore = threading.Semaphore(self.max_concurrent_tasks)
        self.active_tasks = 0
        self.active_tasks_lock = threading.Lock()

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

            # 定义处理任务的函数
            def process_task():
                # 获取信号量，限制并发数
                acquired = self.semaphore.acquire(blocking=True)
                
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
                    
                    # 1. 执行视频分析
                    task_analyzer.analyze_video_task(
                        task_id, absolute_video_path, video_duration, timeout_threshold,
                        confidence_threshold, iou_threshold, callback_url
                    )
                    
                    # 2. 生成结果视频
                    import os
                    result_video_dir = Config.RESULT_VIDEO_PATH
                    os.makedirs(result_video_dir, exist_ok=True)
                    
                    # 生成输出文件名
                    video_filename = os.path.basename(absolute_video_path)
                    name_without_ext = os.path.splitext(video_filename)[0]
                    output_filename = f"{task_id}_{name_without_ext}_result.mp4"
                    output_path = os.path.join(result_video_dir, output_filename)
                    
                    logger.info(f"Task {task_id}: Starting result video generation")
                    success = task_analyzer.export_annotated_video(
                        task_id, absolute_video_path, output_path,
                        confidence_threshold, iou_threshold, callback_url
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
                        
                finally:
                    # 释放信号量并更新活跃任务计数
                    with self.active_tasks_lock:
                        self.active_tasks -= 1
                        logger.info(f"Task {task_id}: Completed (Active tasks: {self.active_tasks}/{self.max_concurrent_tasks})")
                    
                    self.semaphore.release()

            # 在后台线程中执行分析（避免阻塞消费者）
            thread = threading.Thread(
                target=process_task,
                daemon=True,
                name=f"Task-{task_id}"
            )
            thread.start()

            # 确认消息（任务已接受）
            ch.basic_ack(delivery_tag=method.delivery_tag)
            logger.info(f"Task {task_id} accepted and processing started")

        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode message: {e}")
            ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            # 重新入队（如果是临时错误）
            ch.basic_nack(delivery_tag=method.delivery_tag, requeue=True)

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

    def stop(self):
        """停止消费者"""
        try:
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
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

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
