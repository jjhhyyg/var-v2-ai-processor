"""
并发分析任务测试脚本
用于测试系统在处理多个并发分析任务时的性能和稳定性
"""
import pika
import json
import time
import logging
from pathlib import Path
import sys

# 添加父目录到路径
sys.path.append(str(Path(__file__).parent))
from config import Config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ConcurrencyTester:
    """并发测试器"""
    
    def __init__(self):
        """初始化测试器"""
        self.rabbitmq_host = Config.RABBITMQ_HOST
        self.rabbitmq_port = Config.RABBITMQ_PORT
        self.rabbitmq_user = Config.RABBITMQ_USER
        self.rabbitmq_password = Config.RABBITMQ_PASSWORD
        self.queue_name = Config.RABBITMQ_QUEUE
        
    def connect_rabbitmq(self):
        """连接到RabbitMQ"""
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
        
        connection = pika.BlockingConnection(parameters)
        channel = connection.channel()
        
        # 确保队列存在
        channel.queue_declare(queue=self.queue_name, durable=True)
        
        return connection, channel
    
    def send_task(self, channel, task_id, video_path, video_duration=60):
        """
        发送一个分析任务到队列
        
        Args:
            channel: RabbitMQ通道
            task_id: 任务ID
            video_path: 视频路径（相对于codes/目录）
            video_duration: 视频时长（秒）
        """
        message = {
            'taskId': task_id,
            'videoPath': video_path,
            'videoDuration': video_duration,
            'timeoutThreshold': 300,  # 5分钟超时
            'callbackUrl': f"{Config.BACKEND_BASE_URL}/api/tasks/{task_id}/progress",
            'config': {}
        }
        
        channel.basic_publish(
            exchange='',
            routing_key=self.queue_name,
            body=json.dumps(message),
            properties=pika.BasicProperties(
                delivery_mode=2,  # 持久化消息
            )
        )
        
        logger.info(f"Task {task_id} sent to queue: {video_path}")
    
    def get_test_videos(self):
        """
        获取用于测试的视频文件列表
        
        Returns:
            视频路径列表（相对于codes/目录）
        """
        storage_dir = Path(Config.CODES_DIR) / 'storage' / 'videos'
        
        if not storage_dir.exists():
            logger.error(f"Storage directory not found: {storage_dir}")
            return []
        
        # 查找所有.mp4文件
        video_files = list(storage_dir.glob('*.mp4'))
        
        # 转换为相对路径
        relative_paths = []
        for video_file in video_files:
            relative_path = Config.to_relative_path(str(video_file))
            relative_paths.append(relative_path)
        
        logger.info(f"Found {len(relative_paths)} video files for testing")
        return relative_paths
    
    def test_concurrent_tasks(self, num_tasks=2):
        """
        测试并发任务处理
        
        Args:
            num_tasks: 并发任务数量
        """
        logger.info(f"Starting concurrency test with {num_tasks} tasks")
        
        # 获取测试视频
        test_videos = self.get_test_videos()
        
        if len(test_videos) == 0:
            logger.error("No test videos found!")
            return
        
        if num_tasks > len(test_videos):
            logger.warning(f"Requested {num_tasks} tasks but only {len(test_videos)} videos available")
            num_tasks = len(test_videos)
        
        # 连接到RabbitMQ
        connection, channel = self.connect_rabbitmq()
        
        try:
            # 发送多个任务
            start_task_id = 900000000  # 使用特殊的任务ID范围以便识别测试任务
            
            logger.info(f"Sending {num_tasks} tasks to queue...")
            for i in range(num_tasks):
                task_id = start_task_id + i
                video_path = test_videos[i % len(test_videos)]
                
                self.send_task(channel, task_id, video_path)
                
                # 稍微延迟以避免瞬间发送
                time.sleep(0.1)
            
            logger.info(f"All {num_tasks} tasks sent successfully!")
            logger.info("Monitor the AI processor logs to observe concurrent processing")
            
        finally:
            connection.close()
    
    def check_queue_status(self):
        """检查队列状态"""
        connection, channel = self.connect_rabbitmq()
        
        try:
            # 获取队列信息
            queue_info = channel.queue_declare(
                queue=self.queue_name,
                durable=True,
                passive=True  # 只查询，不创建
            )
            
            message_count = queue_info.method.message_count
            consumer_count = queue_info.method.consumer_count
            
            logger.info(f"Queue status:")
            logger.info(f"  - Messages in queue: {message_count}")
            logger.info(f"  - Active consumers: {consumer_count}")
            
            return message_count, consumer_count
            
        finally:
            connection.close()


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='并发分析任务测试工具')
    parser.add_argument(
        '--tasks',
        type=int,
        default=2,
        help='并发任务数量 (默认: 2)'
    )
    parser.add_argument(
        '--status',
        action='store_true',
        help='检查队列状态'
    )
    
    args = parser.parse_args()
    
    tester = ConcurrencyTester()
    
    if args.status:
        tester.check_queue_status()
    else:
        tester.test_concurrent_tasks(args.tasks)


if __name__ == '__main__':
    main()
