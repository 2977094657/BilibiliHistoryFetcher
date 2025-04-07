import os
import re
import shutil
import subprocess
import sys

import yaml


def build(build_type):
    """执行打包过程
    
    Args:
        build_type: "lite" (不含torch), "full" (含torch), 或 "all" (两者都打包)
    """
    try:
        # 检查当前目录下的所有文件
        print("\n=== 当前目录文件检查 ===\n")
        current_dir = os.getcwd()
        print(f"当前目录: {current_dir}")
        print(f"目录内内容: {os.listdir(current_dir)}")
        
        # 检查关键文件
        key_files = [
            'app_launcher.py',
            'main.py',
            'requirements.txt',
            'config',
            'scripts',
            'routers'
        ]
        
        print("\n=== 关键文件检查 ===\n")
        for file in key_files:
            exists = os.path.exists(file)
            if exists and os.path.isdir(file):
                print(f"{file}: 存在 (目录) - 内容: {os.listdir(file)[:5]}...")
            else:
                print(f"{file}: {'存在' if exists else '不存在'}")
        
        # 确保 yutto.exe 存在
        yutto_exe = os.path.join('.venv', 'Scripts', 'yutto.exe')
        if not os.path.exists(yutto_exe):
            raise FileNotFoundError(f"找不到 yutto.exe: {yutto_exe}")

        print(f"\n找到 yutto.exe: {yutto_exe}")
        
        # 检查PyInstaller是否安装
        try:
            pyinstaller_version = subprocess.check_output(['pyinstaller', '--version'], text=True).strip()
            print(f"PyInstaller 版本: {pyinstaller_version}")
        except subprocess.CalledProcessError:
            print("\n错误: 无法运行 PyInstaller，请确保已安装")
            subprocess.run([sys.executable, '-m', 'pip', 'install', 'pyinstaller'])
            print("\n已安装 PyInstaller")
        except FileNotFoundError:
            print("\n错误: 无法找到 PyInstaller，请确保已安装")
            subprocess.run([sys.executable, '-m', 'pip', 'install', 'pyinstaller'])
            print("\n已安装 PyInstaller")
        
        # 确保 app.spec 存在
        if not os.path.exists('app.spec'):
            # 创建基本的spec文件
            print("\n未找到app.spec文件，正在创建...")
            
            try:
                makespec_cmd = [
                    'pyi-makespec',
                    '--name=BilibiliHistoryAnalyzer',
                    '--add-binary', f"{yutto_exe};.",
                    '--add-data', "config/*;config", 
                    '--add-data', "scripts;scripts",
                    '--add-data', "routers;routers",
                    '--add-data', "main.py;.",
                    'app_launcher.py'
                ]
                print(f"\n运行命令: {' '.join(makespec_cmd)}")
                
                result = subprocess.run(makespec_cmd, capture_output=True, text=True)
                if result.returncode != 0:
                    print(f"\n错误: pyi-makespec 运行失败")
                    print(f"\n标准输出: {result.stdout}")
                    print(f"\n错误输出: {result.stderr}")
                else:
                    print("\n成功创建app.spec文件")
            except Exception as e:
                print(f"\n创建spec文件时出错: {e}")
            
        # 检查app.spec是否已存在    
        if not os.path.exists('app.spec'):
            raise FileNotFoundError("无法创建app.spec文件")
        else:
            print(f"\napp.spec 文件大小: {os.path.getsize('app.spec')} 字节")
        
        # 创建一个干净的配置目录用于打包，避免敏感信息泄露
        original_config_dir = 'config'
        clean_config_dir = 'config_clean'
        
        # 确保临时配置目录不存在
        if os.path.exists(clean_config_dir):
            import shutil
            shutil.rmtree(clean_config_dir)
        
        # 创建临时配置目录
        os.makedirs(clean_config_dir)
        
        # 1. 复制所有非敏感配置文件到临时目录
        import glob
        import shutil  # 确保在这里导入shutil
        for config_file in glob.glob(os.path.join(original_config_dir, '*')):
            file_name = os.path.basename(config_file)
            # 跳过敏感文件和备份文件
            if file_name.endswith('.bak') or file_name == 'config.yaml':
                continue
            if os.path.isfile(config_file):
                shutil.copy2(config_file, os.path.join(clean_config_dir, file_name))
        
        # 2. 创建清理过敏感信息的配置文件
        clean_config_file = cleanup_sensitive_config()
        if clean_config_file:
            shutil.copy2(clean_config_file, os.path.join(clean_config_dir, 'config.yaml'))
            print(f"\n已创建不含敏感信息的临时配置目录: {clean_config_dir}")
            
            # 删除临时配置文件（我们已经复制到临时目录了）
            if os.path.exists(clean_config_file):
                try:
                    os.remove(clean_config_file)
                except Exception as e:
                    print(f"\n清理临时配置文件时出错: {e}")
        else:
            print("\n无法创建清理过敏感信息的配置文件")
            return False
        
        try:
            # 创建必要的spec文件并修改其中的配置路径
            if build_type == "lite" or build_type == "all":
                # 复制并修改app.spec为lite版本
                lite_spec_path = 'app_lite.spec'
                success = create_lite_spec('app.spec', lite_spec_path)
                if not success or not os.path.exists(lite_spec_path):
                    raise FileNotFoundError(f"无法创建{lite_spec_path}文件")
                
                # 修改spec文件中的配置路径，确保只使用临时目录的配置
                if clean_config_dir:
                    modify_spec_config_path(lite_spec_path, original_config_dir, clean_config_dir)
                
                # 执行PyInstaller打包Lite版本
                print("\n===== 开始打包Lite版本（不含torch） =====\n")
                
                # 使用新的run_pyinstaller函数打包
                if not run_pyinstaller('app_lite.spec'):
                    print("\n打包失败！")
                    return False
                
                # 复制 yutto.exe 到Lite输出目录
                lite_dist_dir = os.path.join('dist', 'BilibiliHistoryAnalyzer_Lite')
                if os.path.exists(yutto_exe) and os.path.exists(lite_dist_dir):
                    try:
                        shutil.copy2(yutto_exe, lite_dist_dir)
                        print(f"\n已复制 yutto.exe 到 {lite_dist_dir}")
                    except Exception as e:
                        print(f"\n复制 yutto.exe 时出错: {e}")
                else:
                    print(f"\n警告: 无法复制yutto.exe - 目标目录 {lite_dist_dir} 不存在")
                
                # 复制配置文件到应用根目录
                if os.path.exists(lite_dist_dir):
                    print("\n===== 复制配置文件到Lite版本根目录 =====")
                    post_build_copy(lite_dist_dir)
                    print(f"\nLite版本打包完成！程序位于 {lite_dist_dir} 文件夹中")
                    print(f"{os.listdir(lite_dist_dir)[:10]}..." if len(os.listdir(lite_dist_dir)) > 10 else os.listdir(lite_dist_dir))
                else:
                    print(f"\n警告: 找不到输出目录 {lite_dist_dir}，打包可能失败")

            if build_type == "full" or build_type == "all":
                # 复制并修改app.spec为完整版
                full_spec_path = 'app_full.spec'
                success = create_full_spec('app.spec', full_spec_path)
                if not success or not os.path.exists(full_spec_path):
                    raise FileNotFoundError(f"无法创建{full_spec_path}文件")
                
                # 修改spec文件中的配置路径，确保只使用临时目录的配置
                if clean_config_dir:
                    modify_spec_config_path(full_spec_path, original_config_dir, clean_config_dir)
                
                # 执行PyInstaller打包完整版
                print("\n===== 开始打包完整版（含torch） =====\n")
                
                # 使用新的run_pyinstaller函数打包
                if not run_pyinstaller('app_full.spec'):
                    print("\n打包失败！")
                    return False
                
                # 复制 yutto.exe 到完整版输出目录
                full_dist_dir = os.path.join('dist', 'BilibiliHistoryAnalyzer_Full')
                if os.path.exists(yutto_exe) and os.path.exists(full_dist_dir):
                    try:
                        shutil.copy2(yutto_exe, full_dist_dir)
                        print(f"\n已复制 yutto.exe 到 {full_dist_dir}")
                    except Exception as e:
                        print(f"\n复制 yutto.exe 时出错: {e}")
                else:
                    print(f"\n警告: 无法复制yutto.exe - 目标目录 {full_dist_dir} 不存在")
                
                # 复制配置文件到应用根目录
                if os.path.exists(full_dist_dir):
                    print("\n===== 复制配置文件到Full版本根目录 =====")
                    post_build_copy(full_dist_dir)
                    print(f"\n完整版打包完成！程序位于 {full_dist_dir} 文件夹中")
                    print(f"{os.listdir(full_dist_dir)[:10]}..." if len(os.listdir(full_dist_dir)) > 10 else os.listdir(full_dist_dir))
                else:
                    print(f"\n警告: 找不到输出目录 {full_dist_dir}，打包可能失败")
                    
            return True
        finally:
            # 清理临时配置目录
            if clean_config_dir and os.path.exists(clean_config_dir):
                try:
                    import shutil
                    shutil.rmtree(clean_config_dir)
                    print(f"\n临时配置目录 {clean_config_dir} 已删除")
                except Exception as e:
                    print(f"\n删除临时配置目录时出错: {e}")
            
    except subprocess.CalledProcessError as e:
        print(f"\n打包过程出错: {e}")
        print("\n标准输出:")
        if hasattr(e, 'output') and e.output:
            print(e.output[:500] + "..." if len(e.output) > 500 else e.output)
        print("\n错误输出:")
        if hasattr(e, 'stderr') and e.stderr:
            print(e.stderr[:500] + "..." if len(e.stderr) > 500 else e.stderr)
        return False
    except Exception as e:
        print(f"\n发生错误: {e}")
        import traceback
        print(traceback.format_exc())
        return False

def create_lite_spec(source_spec, target_spec):
    """创建Lite版本的spec文件"""
    try:
        with open(source_spec, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 修改名称为Lite版本
        content = content.replace("name='BilibiliHistoryAnalyzer'", "name='BilibiliHistoryAnalyzer_Lite'")
        
        # 修改excludes列表，排除torch相关模块
        if "excludes=[" in content:
            content = content.replace("excludes=[", "excludes=['torch', 'torchvision', 'torchaudio', 'faster_whisper', ")
        else:
            # 如果没有找到excludes，添加一个
            content = content.replace("hiddenimports=[", "excludes=['torch', 'torchvision', 'torchaudio', 'faster_whisper'],\n    hiddenimports=[")
        
        with open(target_spec, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"\n已创建Lite版本spec文件: {target_spec} ({os.path.getsize(target_spec)} 字节)")
        return True
    except Exception as e:
        print(f"\n创建Lite版本spec文件时出错: {e}")
        import traceback
        print(traceback.format_exc())
        return False

def create_full_spec(source_spec, target_spec):
    """创建完整版的spec文件"""
    try:
        with open(source_spec, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 修改名称为完整版
        content = content.replace("name='BilibiliHistoryAnalyzer'", "name='BilibiliHistoryAnalyzer_Full'")
        
        # 删除排除torch相关的代码
        if "excludes=['torch', 'torchvision', 'torchaudio', 'faster_whisper'" in content:
            content = content.replace("excludes=['torch', 'torchvision', 'torchaudio', 'faster_whisper', ", "excludes=[")
        elif "excludes=['torch', 'torchvision', 'torchaudio', 'faster_whisper'" in content:
            content = content.replace("excludes=['torch', 'torchvision', 'torchaudio', 'faster_whisper'],", "excludes=[]")
        
        # 如果有任何torch相关的排除内容，全部删除
        content = content.replace("'torch',", "")
        content = content.replace("'torchvision',", "")
        content = content.replace("'torchaudio',", "")
        content = content.replace("'faster_whisper',", "")
        
        # 添加faster_whisper及其资产到datas列表
        content = content.replace(
            "(os.path.join(venv_site_packages, 'yutto'), 'yutto'),",
            "(os.path.join(venv_site_packages, 'yutto'), 'yutto'),\n        (os.path.join(venv_site_packages, 'faster_whisper/assets'), 'faster_whisper/assets'),\n        (os.path.join(venv_site_packages, 'faster_whisper'), 'faster_whisper'),"
        )

        # 添加faster_whisper到hiddenimports列表
        content = content.replace(
            "'email_validator',",
            "'email_validator',\n        'faster_whisper',\n        'faster_whisper.audio',\n        'faster_whisper.tokenizer',\n        'faster_whisper.transcribe',\n        'faster_whisper.utils',\n        'faster_whisper.vad',\n        'faster_whisper.feature_extractor',"
        )
        
        with open(target_spec, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"\n已创建完整版spec文件: {target_spec} ({os.path.getsize(target_spec)} 字节)")
        return True
    except Exception as e:
        print(f"\n创建完整版spec文件时出错: {e}")
        import traceback
        print(traceback.format_exc())
        return False

def run_pyinstaller(spec_file):
    """运行PyInstaller命令"""
    try:
        print(f"\n运行命令: pyinstaller --clean --noconfirm {spec_file}\n")
        print("正在执行PyInstaller打包过程...这可能需要一些时间...")
        
        # 使用subprocess.Popen实现实时输出
        process = subprocess.Popen(
            ['pyinstaller', '--clean', '--noconfirm', spec_file],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            bufsize=1
        )
        
        # 初始化输出列表
        stdout_lines = []
        stderr_lines = []
        
        print("打包进度信息:")  
        # 导入threading和time模块
        import threading
        import time
        
        # 定义读取输出线程函数
        def read_output(pipe, store, prefix=""):
            last_progress_time = time.time()
            for line in iter(pipe.readline, ''):
                store.append(line)
                # 过滤特定类型的输出
                if 'INFO:' in line or 'WARNING:' in line or 'ERROR:' in line or 'LOADER:' in line:
                    print(f"{prefix}{line.strip()}")
                # 每隔5秒打印一个进度点
                current_time = time.time()
                if current_time - last_progress_time > 5:
                    print(".", end="", flush=True)
                    last_progress_time = current_time
        
        # 创建读取输出线程
        stdout_thread = threading.Thread(target=read_output, args=(process.stdout, stdout_lines))
        stderr_thread = threading.Thread(target=read_output, args=(process.stderr, stderr_lines, "错误: "))
        
        # 设置为守护线程
        stdout_thread.daemon = True
        stderr_thread.daemon = True
        stdout_thread.start()
        stderr_thread.start()
        
        # 每隔5秒打印一个进度点
        while process.poll() is None:
            print(".", end="", flush=True)
            time.sleep(5)
        
        # 等待线程结束
        stdout_thread.join()
        stderr_thread.join()
        
        # 关闭管道
        process.stdout.close()
        process.stderr.close()
        
        # 获取返回值
        return_code = process.returncode
        
        print("\n打包完成")  
        if return_code == 0:
            print("打包成功！")
            return True
        else:
            print(f"\n命令执行失败，返回值: {return_code}")
            print("\n错误输出:")
            for line in stderr_lines[-20:]:  # 打印最后20行错误输出
                print(line.strip())
            return False
    except Exception as e:
        print(f"\n运行PyInstaller出错: {e}")
        import traceback
        print(traceback.format_exc())
        return False

def pack_lite():
    """打包Lite版本（不含torch）"""
    try:
        # 创建Lite版本spec文件
        if not create_lite_spec('app.spec', 'app_lite.spec'):
            print("创建Lite版本spec文件失败，打包终止。")
            return False
        
        print("\n===== 开始打包Lite版本（不含torch） =====")
        
        # 运行PyInstaller命令
        if not run_pyinstaller('app_lite.spec'):
            print("打包失败！")
            return False
        
        # 复制yutto.exe到输出目录
        yutto_path = os.path.join('.venv', 'Scripts', 'yutto.exe')
        output_dir = os.path.join('dist', 'BilibiliHistoryAnalyzer_Lite')
        if os.path.exists(yutto_path) and os.path.exists(output_dir):
            shutil.copy2(yutto_path, output_dir)
            print(f"\n已复制yutto.exe到{output_dir}")
        else:
            print(f"\n警告：无法复制yutto.exe到{output_dir}，目标目录不存在")
        
        print("\nLite版本打包完成！程序位于dist/BilibiliHistoryAnalyzer_Lite文件夹中")
        if os.path.exists(output_dir):
            print(os.listdir(output_dir))
        return True
    except Exception as e:
        print(f"\n打包过程出错：{e}")
        import traceback
        print(traceback.format_exc())
        return False

def pack_full():
    """打包完整版（含torch）"""
    try:
        # 创建完整版spec文件
        if not create_full_spec('app.spec', 'app_full.spec'):
            print("创建完整版spec文件失败，打包终止。")
            return False
        
        print("\n===== 开始打包完整版（含torch） =====")
        
        # 运行PyInstaller命令
        if not run_pyinstaller('app_full.spec'):
            print("打包失败！")
            return False
            
        # 复制yutto.exe到输出目录
        yutto_path = os.path.join('.venv', 'Scripts', 'yutto.exe')
        output_dir = os.path.join('dist', 'BilibiliHistoryAnalyzer_Full')
        if os.path.exists(yutto_path) and os.path.exists(output_dir):
            shutil.copy2(yutto_path, output_dir)
            print(f"\n已复制yutto.exe到{output_dir}")
        else:
            print(f"\n警告：无法复制yutto.exe到{output_dir}，目标目录不存在")
                
        print("\n完整版打包完成！程序位于dist/BilibiliHistoryAnalyzer_Full文件夹中")
        if os.path.exists(output_dir):
            print(os.listdir(output_dir))
        return True
    except Exception as e:
        print(f"\n打包过程出错：{e}")
        import traceback
        print(traceback.format_exc())
        return False

def print_usage():
    print("使用方法: python build.py [lite|full|all]")
    print("  lite: 打包Lite版本（不含torch，体积小，不支持音频转文字）")
    print("  full: 打包完整版（含torch，体积大，支持所有功能）")
    print("  all: 同时打包Lite版本和完整版")

def cleanup_sensitive_config():
    """清理config.yaml文件中的敏感信息，返回临时文件路径"""
    config_path = os.path.join('config', 'config.yaml')
    temp_config_path = os.path.join('config', 'config_temp.yaml')
    
    if not os.path.exists(config_path):
        print(f"\n警告: 找不到配置文件 {config_path}")
        return None
    
    try:
        # 行处理方式 - 保留原始格式和注释
        with open(config_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # 先读取YAML获取结构信息，用于识别敏感字段
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # 敏感字段路径列表和对应的示例值
        sensitive_fields = {
            "SESSDATA": "你的SESSDATA",
            "bili_jct": "你的bili_jct",
            "DedeUserID": "你的DedeUserID",
            "DedeUserID__ckMd5": "你的DedeUserID__ckMd5",
            "email.password": "你的邮箱授权码",
            "email.sender": "example@example.com",
            "email.receiver": "example@example.com",
            "server.ssl_certfile": "path/to/cert.pem",
            "server.ssl_keyfile": "path/to/key.pem",
            "deepseek.api_key": "你的API密钥"
        }
        
        # 将一些独立的顶级字段也加入检查
        for field in ['SESSDATA', 'bili_jct', 'DedeUserID', 'DedeUserID__ckMd5', 'email', 'ssl_certfile', 'ssl_keyfile', 'api_key']:
            if field in config and not isinstance(config[field], dict) and field not in sensitive_fields:
                sensitive_fields[field] = f"你的{field}"
        
        # 处理每一行，根据冒号和缩进识别敏感字段
        new_lines = []
        for line in lines:
            # 跳过注释和空行
            if line.strip().startswith('#') or not line.strip():
                new_lines.append(line)
                continue
            
            # 检查当前行是否包含敏感字段
            matching_sensitive_field = None
            
            # 计算当前行的缩进级别
            indent_level = len(line) - len(line.lstrip())
            field_line = line.strip()
            
            # 如果行包含冒号，可能是一个字段定义
            if ':' in field_line:
                field_name = field_line.split(':', 1)[0].strip()
                
                # 构建完整路径来检查是否匹配敏感字段
                for sensitive_path, example_value in sensitive_fields.items():
                    # 检查是否是叶子节点或完整匹配
                    path_parts = sensitive_path.split('.')
                    if field_name == path_parts[-1]: 
                        # 潜在匹配，保存找到的敏感字段
                        matching_sensitive_field = sensitive_path
                        break
            
            if matching_sensitive_field and ':' in line:
                # 获取示例值
                example_value = sensitive_fields.get(matching_sensitive_field, "示例值")
                
                # 保持原始缩进和键名，但将值替换为示例值
                key_part = line.split(':', 1)[0] + ': '
                remainder = line.split(':', 1)[1]
                
                # 如果值部分只包含简单值，则替换为示例值
                if not remainder.strip().startswith('{') and not remainder.strip().startswith('['):
                    new_line = key_part + f'"{example_value}"' + '\n'
                    new_lines.append(new_line)
                else:
                    # 对于复杂结构，保持不变
                    new_lines.append(line)
            else:
                new_lines.append(line)
        
        # 始终写入临时文件，无论是否修改
        print("\n清理配置文件中的敏感信息...")
        with open(temp_config_path, 'w', encoding='utf-8') as f:
            f.writelines(new_lines)
        print("临时配置文件已创建，敏感信息已替换为示例值")
        return temp_config_path
        
    except Exception as e:
        print(f"\n清理配置文件时出错: {e}")
        import traceback
        print(traceback.format_exc())
        return None

def modify_spec_config_path(spec_file, original_path, new_path):
    """修改spec文件中的配置路径，用临时配置目录替换原始配置目录"""
    try:
        with open(spec_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 首先替换数据路径，确保只使用临时配置目录
        pattern1 = fr"['\"]({re.escape(original_path)}/\*;{re.escape(original_path)})['\"]"
        replacement1 = f"'{new_path}/*;{original_path}'"
        content = re.sub(pattern1, replacement1, content)
        
        # 还需要替换 datas 中的 ('config/*', 'config') 这种模式
        pattern2 = r"\(['\"]config/\*['\"], ['\"]config['\"]"
        replacement2 = f"('{new_path}/*', '{original_path}'"
        content = re.sub(pattern2, replacement2, content)
        
        # 保存修改后的spec文件
        with open(spec_file, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"\n已修改 {spec_file} 中的配置路径: {original_path} -> {new_path}")
        return True
    except Exception as e:
        print(f"\n修改spec文件配置路径时出错: {e}")
        import traceback
        print(traceback.format_exc())
        return False

def post_build_copy(dist_dir):
    """构建后处理，确认_internal/config目录存在
    
    Args:
        dist_dir: 构建输出目录的路径，如 dist/BilibiliHistoryAnalyzer_Full
    """
    try:
        # 检查_internal/config目录是否存在
        internal_config = os.path.join(dist_dir, '_internal', 'config')
        if not os.path.exists(internal_config):
            print(f"\n警告: 找不到内部配置目录: {internal_config}")
            return False
        
        print(f"\n确认内部配置目录存在: {internal_config}")
        print(f"应用将从此目录加载所有配置文件")
        
        # 添加README提示用户修改配置文件位置
        readme_path = os.path.join(dist_dir, 'README_CONFIG.txt')
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write("配置文件说明\n")
            f.write("===========\n\n")
            f.write("所有配置文件位于 _internal/config 目录中\n")
            f.write("- config.yaml: 主配置文件，包含B站认证、邮件、API等配置\n")
            f.write("- scheduler_config.yaml: 计划任务配置文件\n")
        
        print(f"已创建配置说明文件: {readme_path}")
        return True
    except Exception as e:
        print(f"\n构建后处理时出错: {e}")
        import traceback
        print(traceback.format_exc())
        return False

if __name__ == '__main__':
    # 检查命令行参数
    if len(sys.argv) > 1 and sys.argv[1] in ["lite", "full", "all"]:
        if sys.argv[1] == "lite":
            build("lite")
        elif sys.argv[1] == "full":
            build("full")
        elif sys.argv[1] == "all":
            build("all")
    else:
        print_usage()