#!/usr/bin/env python3
"""
测试从模型检查点文件中读取模型版本
"""
import torch
import sys

def test_model_version(model_path: str):
    """测试读取模型版本"""
    try:
        print(f"正在加载模型检查点: {model_path}")
        # PyTorch 2.6+ 需要设置 weights_only=False 来加载包含自定义类的检查点
        ckpt = torch.load(model_path, map_location='cpu', weights_only=False)
        
        print("\n检查点文件中的关键字:")
        print(list(ckpt.keys()))
        
        if 'train_args' in ckpt:
            print("\ntrain_args 内容:")
            print(ckpt['train_args'])
            
            if 'model' in ckpt['train_args']:
                model_version = ckpt['train_args']['model']
                print(f"\n原始模型版本: {model_version}")
                
                # 去掉.pt后缀
                if model_version.endswith('.pt'):
                    model_version = model_version[:-3]
                    print(f"处理后的模型版本: {model_version}")
                else:
                    print(f"模型版本(无.pt后缀): {model_version}")
            else:
                print("\n警告: train_args 中没有 'model' 字段")
        else:
            print("\n警告: 检查点文件中没有 'train_args' 字段")
            
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    model_path = 'weights/best.pt' if len(sys.argv) < 2 else sys.argv[1]
    test_model_version(model_path)
