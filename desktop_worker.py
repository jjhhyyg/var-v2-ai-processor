"""
桌面版 AI worker 入口。

职责：
1. 读取 Rust 侧写入的 job JSON
2. 本地执行视频分析与结果视频导出
3. 通过 stdout 输出 NDJSON 事件流，供桌面端 Rust 核心消费
"""
import os

# 打包后的 worker 不是 Python 解释器，禁掉 Ultralytics 的运行时自动装包，
# 避免它递归调用自身执行 `-m pip` 并污染任务状态。
os.environ.setdefault('ULTRALYTICS_SKIP_REQUIREMENTS_CHECKS', '1')

import json
import logging
from logging.handlers import RotatingFileHandler
import sys
import traceback
from pathlib import Path


logger = logging.getLogger(__name__)
LOG_MAX_BYTES = 10 * 1024 * 1024
LOG_BACKUP_COUNT = 200


def emit_event(event_type: str, payload: dict) -> None:
    # stdout is reserved for the Rust NDJSON protocol. Human-readable logs must use logging/stderr.
    print(json.dumps({
        'type': event_type,
        'payload': payload
    }, ensure_ascii=False), flush=True)


def configure_logging(log_path: str) -> None:
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = RotatingFileHandler(
        log_path,
        mode='a',
        maxBytes=LOG_MAX_BYTES,
        backupCount=LOG_BACKUP_COUNT,
        encoding='utf-8'
    )
    file_handler.namer = rotate_log_name
    file_handler.setFormatter(formatter)

    logging.basicConfig(
        level=logging.INFO,
        handlers=[file_handler],
        force=True
    )


def rotate_log_name(default_name: str) -> str:
    path = Path(default_name)
    base = path.name
    if '.log.' not in base:
        return default_name
    stem, index = base.rsplit('.log.', 1)
    return str(path.with_name(f'{stem}.{index}.log'))


def load_job(job_path: str) -> dict:
    with open(job_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def run_self_check() -> int:
    try:
        import ultralytics

        # Self-check is a CLI JSON contract, not the worker NDJSON event stream.
        print(json.dumps({
            'status': 'ok',
            'checks': {
                'ultralytics': getattr(ultralytics, '__version__', 'unknown')
            }
        }, ensure_ascii=False), flush=True)
        return 0
    except Exception:
        traceback.print_exc(file=sys.stderr)
        return 1


def main() -> int:
    if len(sys.argv) >= 2 and sys.argv[1] == '--self-check':
        return run_self_check()

    if len(sys.argv) < 2:
        emit_event('failed', {'message': '缺少 job 文件路径'})
        return 2

    analyzer = None

    try:
        from analyzer.video_processor import VideoAnalyzer
        from utils.callback import BackendCallback

        job = load_job(sys.argv[1])
        configure_logging(job['logPath'])

        task_id = int(job['taskId'])
        config = job.get('config', {})

        analyzer = VideoAnalyzer(
            model_path=job['modelPath'],
            device=job.get('device', '')
        )

        emit_event('model_version', {
            'modelVersion': analyzer.yolo_tracker.model_version
        })

        status, analyzed_video_path = analyzer.analyze_video_task(
            task_id=task_id,
            video_path=job['videoPath'],
            video_duration=int(job['videoDuration']),
            timeout_threshold=int(job['timeoutThreshold']),
            confidence_threshold=float(config.get('confidenceThreshold', 0.5)),
            iou_threshold=float(config.get('iouThreshold', 0.45)),
            enable_preprocessing=bool(config.get('enablePreprocessing', False)),
            preprocessing_strength=config.get('preprocessingStrength', 'moderate'),
            preprocessing_enhance_pool=bool(config.get('preprocessingEnhancePool', False)),
            callback_url=BackendCallback.STDOUT_CALLBACK_URL,
            frame_rate=float(config.get('frameRate', 25.0)),
            preprocessed_output_path=job.get('preprocessedOutputPath')
        )

        logger.info("Task %s analysis finished with status=%s analyzed_video_path=%s", task_id, status, analyzed_video_path)

        if status in ['COMPLETED', 'COMPLETED_TIMEOUT']:
            result_output_path = job['resultOutputPath']
            success = analyzer.export_annotated_video(
                task_id=task_id,
                video_path=analyzed_video_path,
                output_path=result_output_path,
                confidence_threshold=float(config.get('confidenceThreshold', 0.5)),
                iou_threshold=float(config.get('iouThreshold', 0.45)),
                callback_url=BackendCallback.STDOUT_CALLBACK_URL,
                frame_rate=float(config.get('frameRate', 25.0)),
                progress_status=status,
            )

            if success:
                emit_event('result_video_ready', {
                    'path': str(Path(result_output_path).resolve())
                })

        return 0
    except Exception as exc:
        message = str(exc)
        traceback.print_exc(file=sys.stderr)
        emit_event('failed', {
            'message': message,
            'traceback': traceback.format_exc()
        })
        return 1
    finally:
        if analyzer is not None:
            try:
                analyzer.cleanup()
            except Exception:
                logger.exception("Failed to cleanup analyzer")


if __name__ == '__main__':
    raise SystemExit(main())
