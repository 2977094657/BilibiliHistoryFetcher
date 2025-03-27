#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
自动检测CUDA版本并安装对应的PyTorch版本的脚本
"""

import os
import sys
import subprocess
import re
import platform
import time
import argparse

def run_command(command, timeout=None, show_output=False):
    """运行系统命令并返回输出"""
    print(f"执行: {command}")
    
    try:
        process = subprocess.Popen(
            command, 
            shell=True, 
            stdout=subprocess.PIPE if not show_output else None,
            stderr=subprocess.PIPE if not show_output else None,
            universal_newlines=True
        )
        
        if show_output:
            # 如果显示输出，等待进程完成
            return_code = process.wait(timeout=timeout)
            if return_code != 0:
                print(f"命令执行失败，返回代码: {return_code}")
                return None
            return "命令输出已直接显示"
        else:
            # 定期检查进程是否完成，并显示进度
            start_time = time.time()
            while timeout is None or (time.time() - start_time) < timeout:
                # 检查进程是否完成
                return_code = process.poll()
                if return_code is not None:
                    stdout, stderr = process.communicate()
                    if return_code != 0:
                        print(f"命令执行失败，返回代码: {return_code}")
                        if stderr:
                            print(f"错误信息: {stderr}")
                        return None
                    return stdout
                
                # 显示进度指示
                elapsed = time.time() - start_time
                if elapsed > 5:  # 如果命令执行超过5秒才显示进度
                    sys.stdout.write(f"\r正在执行命令... {elapsed:.1f}s 已过")
                    sys.stdout.flush()
                
                # 短暂休眠以减少CPU使用
                time.sleep(0.5)
            
            # 如果超时
            print(f"\n命令执行超时 (>{timeout}s)")
            process.kill()
            return None
    
    except KeyboardInterrupt:
        print("\n操作被用户中断")
        try:
            process.kill()
        except:
            pass
        raise
    except Exception as e:
        print(f"错误: 执行命令 '{command}' 失败: {str(e)}")
        return None

def get_cuda_version():
    """检测系统CUDA版本"""
    # 首先尝试使用nvidia-smi
    print("检测CUDA版本...")
    output = run_command("nvidia-smi")
    if output:
        # 尝试从输出中提取CUDA版本
        match = re.search(r"CUDA Version: (\d+\.\d+)", output)
        if match:
            return match.group(1)
    
    # 如果nvidia-smi失败，尝试检查CUDA路径
    if platform.system() == "Windows":
        cuda_paths = [
            r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA",
            os.path.join(os.environ.get('ProgramFiles', ''), "NVIDIA GPU Computing Toolkit", "CUDA"),
        ]
        for base_path in cuda_paths:
            if os.path.exists(base_path):
                # 列出所有版本目录
                cuda_versions = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d)) and d.startswith('v')]
                if cuda_versions:
                    # 返回最高版本
                    latest = sorted(cuda_versions, reverse=True)[0]
                    match = re.search(r"v(\d+\.\d+)", latest)
                    if match:
                        return match.group(1)
    else:  # Linux/macOS
        cuda_path = "/usr/local/cuda"
        if os.path.exists(cuda_path) and os.path.islink(cuda_path):
            cuda_version_path = os.readlink(cuda_path)
            match = re.search(r"cuda-(\d+\.\d+)", cuda_version_path)
            if match:
                return match.group(1)
    
    return None

def get_pytorch_version_for_cuda(cuda_version):
    """根据CUDA版本获取兼容的PyTorch版本"""
    # PyTorch版本与CUDA版本的对应关系
    # 格式: CUDA版本: {"torch": PyTorch版本, "cuda_label": CUDA标签, "cuda_date": CUDA发布时间, "torch_date": PyTorch发布时间}
    pytorch_cuda_map = {
        # CUDA 12.x 系列
        "12.8": {"torch": "2.6.0+cu126", "cuda_label": "cu126", "cuda_date": "2024年7月", "torch_date": "2024年6月"},  # 12.8向下兼容12.6
        "12.7": {"torch": "2.6.0+cu126", "cuda_label": "cu126", "cuda_date": "2024年5月", "torch_date": "2024年6月"},  # 12.7向下兼容12.6
        "12.6": {"torch": "2.6.0+cu126", "cuda_label": "cu126", "cuda_date": "2024年3月", "torch_date": "2024年6月"},
        "12.5": {"torch": "2.6.0+cu121", "cuda_label": "cu121", "cuda_date": "2024年1月", "torch_date": "2024年6月"},  # 12.5向下兼容12.1
        "12.4": {"torch": "2.6.0+cu121", "cuda_label": "cu121", "cuda_date": "2023年11月", "torch_date": "2024年6月"},  # 12.4向下兼容12.1
        "12.3": {"torch": "2.6.0+cu121", "cuda_label": "cu121", "cuda_date": "2023年10月", "torch_date": "2024年6月"},  # 12.3向下兼容12.1
        "12.2": {"torch": "2.6.0+cu121", "cuda_label": "cu121", "cuda_date": "2023年8月", "torch_date": "2024年6月"},   # 12.2向下兼容12.1
        "12.1": {"torch": "2.6.0+cu121", "cuda_label": "cu121", "cuda_date": "2023年6月", "torch_date": "2024年6月"},
        "12.0": {"torch": "2.1.0+cu121", "cuda_label": "cu121", "cuda_date": "2023年3月", "torch_date": "2023年10月"},  # 12.0向下兼容12.1版PyTorch
        
        # CUDA 11.x 系列
        "11.8": {"torch": "2.2.0+cu118", "cuda_label": "cu118", "cuda_date": "2022年11月", "torch_date": "2024年1月"},
        "11.7": {"torch": "2.0.0+cu117", "cuda_label": "cu117", "cuda_date": "2022年6月", "torch_date": "2023年3月"},
        "11.6": {"torch": "1.13.1+cu116", "cuda_label": "cu116", "cuda_date": "2022年1月", "torch_date": "2022年12月"},
        "11.5": {"torch": "1.12.1+cu115", "cuda_label": "cu115", "cuda_date": "2021年10月", "torch_date": "2022年8月"},
        "11.4": {"torch": "1.12.1+cu113", "cuda_label": "cu113", "cuda_date": "2021年7月", "torch_date": "2022年8月"},  # 11.4向下兼容11.3的PyTorch
        "11.3": {"torch": "1.12.1+cu113", "cuda_label": "cu113", "cuda_date": "2021年4月", "torch_date": "2022年8月"},
        "11.2": {"torch": "1.10.2+cu113", "cuda_label": "cu113", "cuda_date": "2020年12月", "torch_date": "2021年10月"},  # 11.2向下兼容11.3的PyTorch
        "11.1": {"torch": "1.10.2+cu111", "cuda_label": "cu111", "cuda_date": "2020年11月", "torch_date": "2021年10月"},
        "11.0": {"torch": "1.9.1+cu111", "cuda_label": "cu111", "cuda_date": "2020年7月", "torch_date": "2021年8月"},
        
        # CUDA 10.x 系列
        "10.2": {"torch": "1.12.1+cu102", "cuda_label": "cu102", "cuda_date": "2019年11月", "torch_date": "2022年8月"},
        "10.1": {"torch": "1.7.1+cu101", "cuda_label": "cu101", "cuda_date": "2019年2月", "torch_date": "2020年11月"},
        "10.0": {"torch": "1.6.0+cu100", "cuda_label": "cu100", "cuda_date": "2018年9月", "torch_date": "2020年7月"},
        
        # CUDA 9.x 系列
        "9.2": {"torch": "1.5.1+cu92", "cuda_label": "cu92", "cuda_date": "2018年5月", "torch_date": "2020年4月"},
        "9.0": {"torch": "1.1.0", "cuda_label": "cu90", "cuda_date": "2017年9月", "torch_date": "2019年5月"},
    }
    
    # 获取主要版本和次要版本
    match = re.match(r"(\d+)\.(\d+)", cuda_version)
    if match:
        major, minor = match.groups()
        version_key = f"{major}.{minor}"
        
        if version_key in pytorch_cuda_map:
            return pytorch_cuda_map[version_key]
    
    # 如果没有精确匹配，尝试找到兼容版本
    major_version = float(cuda_version.split(".")[0])
    compatible_versions = []
    
    for cuda_ver, pytorch_info in pytorch_cuda_map.items():
        cuda_major = float(cuda_ver.split(".")[0])
        # 同一主版本内，较新的CUDA通常兼容较旧版本的PyTorch
        if cuda_major == major_version and float(cuda_ver) <= float(cuda_version):
            compatible_versions.append((cuda_ver, pytorch_info))
    
    if compatible_versions:
        # 选择最高的兼容版本
        compatible_versions.sort(key=lambda x: float(x[0]), reverse=True)
        print(f"注意: CUDA {cuda_version} 没有精确匹配的PyTorch版本，使用兼容版本 CUDA {compatible_versions[0][0]}")
        return compatible_versions[0][1]
    
    # 如果没有找到兼容版本，返回CPU版本
    print(f"警告: 找不到与CUDA {cuda_version}兼容的PyTorch版本，将使用CPU版本")
    return {"torch": "2.6.0", "cuda_label": "cpu", "cuda_date": "N/A", "torch_date": "2024年6月"}

def install_dependencies(cuda_version=None, skip_torch=False, force_cpu=False):
    """安装依赖，包括匹配CUDA版本的PyTorch"""
    # 安装基本依赖
    print("\n===== 步骤 1/3: 安装基本依赖 =====")
    run_command("pip install -r requirements.txt --upgrade", show_output=True)
    
    # 如果跳过PyTorch安装，直接返回
    if skip_torch:
        print("\n跳过PyTorch安装")
        return
        
    # 检测CUDA版本并安装相应的PyTorch
    print("\n===== 步骤 2/3: 安装PyTorch =====")
    
    if force_cpu:
        print("强制使用CPU版本的PyTorch...")
        run_command("pip uninstall -y torch torchvision torchaudio", show_output=True)
        run_command("pip install torch torchvision torchaudio", show_output=True)
        return
        
    if cuda_version is None:
        cuda_version = get_cuda_version()
    
    if cuda_version and not force_cpu:
        print(f"检测到CUDA版本: {cuda_version}")
        pytorch_info = get_pytorch_version_for_cuda(cuda_version)
        cuda_label = pytorch_info["cuda_label"]
        
        # 显示版本信息和发布时间
        torch_version = pytorch_info["torch"]
        cuda_date = pytorch_info.get("cuda_date", "未知")
        torch_date = pytorch_info.get("torch_date", "未知")
        
        print(f"为CUDA {cuda_version}(发布于{cuda_date})安装PyTorch {torch_version}(发布于{torch_date})...")
        print(f"CUDA标签: {cuda_label}")
        
        # 先卸载现有的PyTorch相关包
        run_command("pip uninstall -y torch torchvision torchaudio", show_output=True)
        
        # 安装对应版本的PyTorch
        if cuda_label == "cpu":
            run_command("pip install torch torchvision torchaudio", show_output=True)
        else:
            # PyTorch安装可能需要较长时间，显示输出以提供反馈
            install_cmd = f"pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/{cuda_label}"
            run_command(install_cmd, show_output=True)
    else:
        print("未检测到CUDA或强制使用CPU，安装CPU版本的PyTorch...")
        run_command("pip uninstall -y torch torchvision torchaudio", show_output=True)
        run_command("pip install torch torchvision torchaudio", show_output=True)
    
    # 安装其他特定依赖
    print("\n===== 步骤 3/3: 安装其他依赖 =====")
    if platform.system() == "Windows" and cuda_version and not force_cpu:
        print("安装Windows特定的CUDA依赖...")
        run_command("pip install faster-whisper>=0.9.0 huggingface-hub>=0.19.0", show_output=True)
    
    print("\n所有依赖安装完成！")

def check_installation():
    """检查安装是否成功"""
    print("\n===== 验证PyTorch安装 =====")
    try:
        import torch
        print(f"PyTorch版本: {torch.__version__}")
        print(f"CUDA是否可用: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"CUDA版本: {torch.version.cuda}")
            print(f"当前设备: {torch.cuda.current_device()}")
            print(f"设备数量: {torch.cuda.device_count()}")
            print(f"设备名称: {torch.cuda.get_device_name(0)}")
            
            # 尝试简单的CUDA操作
            x = torch.tensor([1.0, 2.0, 3.0]).cuda()
            print(f"CUDA张量: {x}")
            print(f"CUDA张量的设备: {x.device}")
            print("\nCUDA功能测试成功！")
            return True
        else:
            print("CUDA不可用 - 使用CPU模式")
            return False
    except ImportError:
        print("PyTorch安装验证失败")
        return False
    except Exception as e:
        print(f"验证时发生错误: {str(e)}")
        return False

def get_environment_info():
    """收集环境信息"""
    print("\n===== 系统环境信息 =====")
    
    # 基本系统信息
    print(f"操作系统: {platform.system()} {platform.release()} {platform.version()}")
    print(f"处理器: {platform.processor()}")
    print(f"Python版本: {platform.python_version()}")
    print(f"Python可执行文件: {sys.executable}")
    
    # CUDA和GPU信息
    try:
        output = run_command("nvidia-smi")
        if output:
            print("\nNVIDIA GPU信息:")
            lines = output.split('\n')
            for line in lines[:10]:  # 只显示前10行
                print(f"  {line}")
            if len(lines) > 10:
                print("  ...")
    except:
        print("无法获取NVIDIA GPU信息")
    
    # 已安装的PyTorch信息
    try:
        import torch
        print(f"\nPyTorch信息:")
        print(f"  版本: {torch.__version__}")
        print(f"  安装路径: {os.path.dirname(torch.__file__)}")
        print(f"  CUDA可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  CUDA版本: {torch.version.cuda}")
            print(f"  GPU数量: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    except ImportError:
        print("\nPyTorch未安装")
    except Exception as e:
        print(f"\nPyTorch信息获取失败: {str(e)}")
    
    # 显示CUDA版本和PyTorch版本对应关系
    print("\n===== CUDA和PyTorch版本对应关系 =====")
    print("| CUDA版本 | PyTorch版本 | CUDA发布时间 | PyTorch发布时间 |")
    print("|---------|------------|------------|---------------|")
    
    pytorch_map = get_pytorch_version_for_cuda("12.7")  # 获取映射表
    pytorch_cuda_map = {}
    
    # 获取原始的映射表
    for name, attr in globals().items():
        if name == 'get_pytorch_version_for_cuda':
            # 提取函数中定义的pytorch_cuda_map变量
            import inspect
            source = inspect.getsource(attr)
            # 简单解析获取pytorch_cuda_map
            start_idx = source.find('pytorch_cuda_map = {')
            if start_idx != -1:
                # 找到映射表的开始，现在查找结束大括号
                bracket_count = 1
                end_idx = start_idx + len('pytorch_cuda_map = {')
                while bracket_count > 0 and end_idx < len(source):
                    if source[end_idx] == '{':
                        bracket_count += 1
                    elif source[end_idx] == '}':
                        bracket_count -= 1
                    end_idx += 1
                
                if bracket_count == 0:
                    # 提取映射表的源代码
                    map_source = source[start_idx:end_idx]
                    try:
                        # 尝试使用一个安全的方式执行这段代码
                        local_vars = {}
                        exec(map_source, globals(), local_vars)
                        if 'pytorch_cuda_map' in local_vars:
                            pytorch_cuda_map = local_vars['pytorch_cuda_map']
                    except:
                        # 如果失败，手动构建一个简单的映射关系
                        pass
    
    # 如果无法从源码中提取，至少显示一些重要版本
    if not pytorch_cuda_map:
        pytorch_cuda_map = {
            "12.7": {"torch": "2.6.0+cu126", "cuda_label": "cu126", "cuda_date": "2024年5月", "torch_date": "2024年6月"},
            "12.1": {"torch": "2.6.0+cu121", "cuda_label": "cu121", "cuda_date": "2023年6月", "torch_date": "2024年6月"},
            "11.8": {"torch": "2.2.0+cu118", "cuda_label": "cu118", "cuda_date": "2022年11月", "torch_date": "2024年1月"},
            "11.7": {"torch": "2.0.0+cu117", "cuda_label": "cu117", "cuda_date": "2022年6月", "torch_date": "2023年3月"},
        }
    
    # 按CUDA版本排序并显示
    for cuda_ver in sorted(pytorch_cuda_map.keys(), key=lambda v: float(v), reverse=True):
        info = pytorch_cuda_map[cuda_ver]
        torch_ver = info.get("torch", "未知")
        cuda_date = info.get("cuda_date", "未知")
        torch_date = info.get("torch_date", "未知")
        print(f"| {cuda_ver} | {torch_ver} | {cuda_date} | {torch_date} |")
    
    # 显示安装信息
    print("\n===== 安装选项 =====")
    print("使用以下命令安装您所需的PyTorch版本:")
    print("  python install_dependencies.py              # 自动检测并安装")
    print("  python install_dependencies.py --force-cpu  # 强制安装CPU版本")
    print("  python install_dependencies.py --force-cuda 12.7  # 安装指定CUDA版本")
    print("  python install_dependencies.py --skip-torch # 跳过PyTorch安装")

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="B站历史记录获取与分析工具 - 依赖安装脚本")
    
    parser.add_argument("--skip-torch", action="store_true", help="跳过PyTorch安装")
    parser.add_argument("--force-cpu", action="store_true", help="强制使用CPU版本的PyTorch")
    parser.add_argument("--force-cuda", metavar="VERSION", help="强制使用指定版本的CUDA (例如: 11.8)")
    parser.add_argument("--info", action="store_true", help="显示系统环境信息并退出")
    
    return parser.parse_args()

if __name__ == "__main__":
    print("===== B站历史记录获取与分析工具 - 依赖安装脚本 =====")
    
    # 解析命令行参数
    args = parse_arguments()
    
    # 如果只是显示环境信息
    if args.info:
        get_environment_info()
        sys.exit(0)
    
    # 开始安装过程
    try:
        # 判断是否有命令行参数指定CUDA版本
        if args.force_cuda:
            if args.force_cpu:
                raise ValueError("错误：不能同时使用 --force_cuda 和 --force_cpu。") 
            cuda_version = args.force_cuda
            print(f"强制使用CUDA版本: {cuda_version}")
            install_dependencies(cuda_version, skip_torch=args.skip_torch, force_cpu=args.force_cpu)
        else:
            install_dependencies(skip_torch=args.skip_torch, force_cpu=args.force_cpu)
        
        # 验证安装
        if not args.skip_torch:
            success = check_installation()
            if success:
                print("\n==================================")
                print("✅ 安装成功！系统已准备好使用GPU加速。")
                print("==================================")
            else:
                print("\n==================================")
                print("⚠️ 安装完成，但CUDA可能不可用。")
                print("如果需要GPU加速，请尝试以下命令:")
                print("  python install_dependencies.py --force-cpu")
                print("这将安装CPU版本的PyTorch，确保兼容性。")
                print("==================================")
        else:
            print("\n==================================")
            print("✅ 基本依赖安装成功！")
            print("==================================")
    except KeyboardInterrupt:
        print("\n\n安装过程被用户中断。")
        print("您可以稍后重新运行脚本来完成安装。")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n安装过程中出现错误: {str(e)}")
        print("请尝试以下命令使用CPU版本:")
        print("  python install_dependencies.py --force-cpu")
        sys.exit(1) 