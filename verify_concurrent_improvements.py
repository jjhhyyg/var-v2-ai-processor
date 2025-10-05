#!/usr/bin/env python3
"""
并发处理验证脚本
用于验证改进后的系统能够正确处理并发任务
"""
import time
import subprocess
import sys
from pathlib import Path

# 添加父目录到路径
sys.path.append(str(Path(__file__).parent))
from test_concurrent_analysis import ConcurrencyTester

def print_section(title):
    """打印分节标题"""
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60 + "\n")

def verify_concurrent_processing():
    """验证并发处理功能"""
    
    print_section("并发处理验证测试")
    
    tester = ConcurrencyTester()
    
    # 1. 检查初始队列状态
    print_section("步骤 1: 检查队列初始状态")
    try:
        message_count, consumer_count = tester.check_queue_status()
        
        if consumer_count == 0:
            print("❌ 错误: 没有活跃的消费者")
            print("   请先启动 AI 处理器: python app.py")
            return False
        
        print(f"✅ 队列状态正常")
        print(f"   - 待处理消息: {message_count}")
        print(f"   - 活跃消费者: {consumer_count}")
        
    except Exception as e:
        print(f"❌ 连接队列失败: {e}")
        print("   请确保 RabbitMQ 正在运行")
        return False
    
    # 2. 发送并发测试任务
    print_section("步骤 2: 发送 3 个并发测试任务")
    try:
        tester.test_concurrent_tasks(num_tasks=3)
        print("✅ 成功发送 3 个测试任务")
        
    except Exception as e:
        print(f"❌ 发送任务失败: {e}")
        return False
    
    # 3. 等待并监控处理过程
    print_section("步骤 3: 监控任务处理")
    print("正在监控任务处理状态...")
    print("提示: 查看 AI 处理器日志以观察并发处理情况")
    print("\n应该看到的关键日志:")
    print("  - 'Task XXX: Created independent analyzer instance'")
    print("  - 'Task XXX: Started (Active tasks: N/3)'")
    print("  - 'Task XXX: Completed (Active tasks: N/3)'")
    
    # 等待一段时间让任务开始处理
    print("\n等待 10 秒让任务开始处理...")
    for i in range(10, 0, -1):
        print(f"  {i}...", end="\r")
        time.sleep(1)
    print("\n")
    
    # 4. 再次检查队列状态
    print_section("步骤 4: 检查队列状态")
    try:
        message_count, consumer_count = tester.check_queue_status()
        print(f"当前队列状态:")
        print(f"  - 待处理消息: {message_count}")
        print(f"  - 活跃消费者: {consumer_count}")
        
        if message_count == 0:
            print("\n✅ 所有任务已被接收处理")
        else:
            print(f"\n⏳ 还有 {message_count} 个任务等待处理")
        
    except Exception as e:
        print(f"⚠️  检查队列状态失败: {e}")
    
    # 5. 验证指南
    print_section("步骤 5: 手动验证指南")
    print("请按以下步骤验证并发处理是否正确:")
    print()
    print("1. 检查 AI 处理器日志")
    print("   - 应该看到 3 个任务几乎同时开始")
    print("   - 每个任务都创建了独立的分析器实例")
    print("   - 活跃任务数在 1-3 之间变化")
    print()
    print("2. 检查结果视频")
    print("   - 位置: storage/result_videos/")
    print("   - 应该生成 3 个结果视频")
    print("   - 文件名格式: 900000XXX_xxx_result.mp4")
    print()
    print("3. 验证 Track ID 独立性")
    print("   - 打开不同任务的结果视频")
    print("   - 确认追踪标注正确")
    print("   - Track ID 应该在各自视频内连续")
    print()
    print("4. 检查系统资源")
    print("   - CPU 使用率应该提高")
    print("   - 内存使用应该增加约 1-1.5GB")
    print("   - 没有内存泄漏或崩溃")
    
    print_section("验证完成")
    return True

def main():
    """主函数"""
    print("""
╔══════════════════════════════════════════════════════════╗
║                                                          ║
║        并发处理改进验证脚本                              ║
║                                                          ║
║  本脚本将验证系统能够正确处理并发分析任务                ║
║                                                          ║
╚══════════════════════════════════════════════════════════╝
    """)
    
    # 确认 AI 处理器正在运行
    print("⚠️  开始验证前，请确保:")
    print("   1. AI 处理器正在运行 (python app.py)")
    print("   2. RabbitMQ 服务正在运行")
    print("   3. 已激活 pytorch conda 环境")
    print()
    
    response = input("准备好了吗? (y/n): ").strip().lower()
    if response != 'y':
        print("已取消验证")
        return
    
    # 执行验证
    success = verify_concurrent_processing()
    
    if success:
        print("\n✅ 验证脚本执行完成!")
        print("   请按照上述指南进行手动验证")
    else:
        print("\n❌ 验证过程中遇到错误")
        print("   请检查错误信息并解决问题后重试")

if __name__ == '__main__':
    main()
