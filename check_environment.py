"""
环境检查脚本
用于检查病虫害识别系统所需的依赖是否正确安装
"""

import sys
from pathlib import Path

def check_python_version():
    """检查Python版本"""
    print("=" * 60)
    print("检查 Python 版本...")
    print("=" * 60)
    version = sys.version_info
    print(f"当前 Python 版本: {version.major}.{version.minor}.{version.micro}")
    
    if version.major >= 3 and version.minor >= 8:
        print("✅ Python 版本符合要求 (>= 3.8)")
        return True
    else:
        print("❌ Python 版本过低，需要 3.8 或更高版本")
        return False

def check_package(package_name, import_name=None):
    """检查单个包是否安装"""
    if import_name is None:
        import_name = package_name
    
    try:
        module = __import__(import_name)
        version = getattr(module, '__version__', '未知版本')
        print(f"✅ {package_name:20s} - 已安装 (版本: {version})")
        return True
    except ImportError:
        print(f"❌ {package_name:20s} - 未安装")
        return False

def check_packages():
    """检查所有必需的包"""
    print("\n" + "=" * 60)
    print("检查依赖包...")
    print("=" * 60)

    # 核心依赖（必需）
    core_packages = [
        ("PyQt5", "PyQt5"),
        ("OpenCV", "cv2"),
        ("NumPy", "numpy"),
    ]

    # ONNX 相关依赖
    onnx_packages = [
        ("ONNX Runtime", "onnxruntime"),
    ]

    # PyTorch 相关依赖（可选，用于 .pt 模型）
    pytorch_packages = [
        ("Ultralytics", "ultralytics"),
        ("PyTorch", "torch"),
        ("TorchVision", "torchvision"),
    ]

    print("\n【核心依赖】")
    core_results = []
    for package_name, import_name in core_packages:
        core_results.append(check_package(package_name, import_name))

    print("\n【ONNX 模型支持】")
    onnx_results = []
    for package_name, import_name in onnx_packages:
        onnx_results.append(check_package(package_name, import_name))

    print("\n【PyTorch 模型支持（可选）】")
    pytorch_results = []
    for package_name, import_name in pytorch_packages:
        pytorch_results.append(check_package(package_name, import_name))

    # 核心依赖必须全部安装
    if not all(core_results):
        print("\n⚠️  核心依赖缺失，程序无法运行！")
        return False

    # 至少需要一种模型支持
    has_onnx = all(onnx_results)
    has_pytorch = all(pytorch_results)

    if not has_onnx and not has_pytorch:
        print("\n⚠️  警告：未安装任何模型推理引擎！")
        print("   请至少安装以下之一：")
        print("   - ONNX Runtime (推荐): pip install onnxruntime")
        print("   - Ultralytics + PyTorch: pip install ultralytics torch torchvision")
        return False

    if has_onnx:
        print("\n✅ ONNX 模型支持已启用（推荐）")
    if has_pytorch:
        print("\n✅ PyTorch 模型支持已启用")

    return True

def check_model_file():
    """检查模型文件是否存在"""
    print("\n" + "=" * 60)
    print("检查模型文件...")
    print("=" * 60)

    from config import MODEL_PATH

    model_path = Path(MODEL_PATH)
    if model_path.exists():
        size_mb = model_path.stat().st_size / (1024 * 1024)
        print(f"✅ 模型文件存在: {MODEL_PATH}")
        print(f"   文件大小: {size_mb:.2f} MB")

        # 判断模型类型
        if MODEL_PATH.endswith('.onnx'):
            print(f"   模型类型: ONNX 格式")
        elif MODEL_PATH.endswith('.pt'):
            print(f"   模型类型: PyTorch 格式")
        else:
            print(f"   模型类型: 未知格式")

        return True
    else:
        print(f"❌ 模型文件不存在: {MODEL_PATH}")
        print("   请将训练好的模型文件放在程序目录下")
        print("   支持的格式：")
        print("   - ONNX 格式: best.onnx (推荐)")
        print("   - PyTorch 格式: best.pt")
        return False

def check_config():
    """检查配置文件"""
    print("\n" + "=" * 60)
    print("检查配置文件...")
    print("=" * 60)
    
    try:
        import config
        print("✅ 配置文件加载成功")
        print(f"   模型路径: {config.MODEL_PATH}")
        print(f"   置信度阈值: {config.CONFIDENCE_THRESHOLD}")
        print(f"   窗口大小: {config.WINDOW_WIDTH} x {config.WINDOW_HEIGHT}")
        return True
    except Exception as e:
        print(f"❌ 配置文件加载失败: {e}")
        return False

def main():
    """主函数"""
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 10 + "番茄病虫害识别系统 - 环境检查" + " " * 16 + "║")
    print("╚" + "=" * 58 + "╝")
    print("\n")
    
    results = []
    
    # 检查Python版本
    results.append(check_python_version())
    
    # 检查依赖包
    results.append(check_packages())
    
    # 检查配置文件
    results.append(check_config())
    
    # 检查模型文件
    results.append(check_model_file())
    
    # 总结
    print("\n" + "=" * 60)
    print("检查结果总结")
    print("=" * 60)
    
    if all(results):
        print("✅ 所有检查通过！环境配置正确。")
        print("\n可以运行以下命令启动程序：")
        print("   python disease_detection_ui.py")
        print("\n或双击 run.bat 文件启动（Windows）")
    else:
        print("❌ 部分检查未通过，请根据上述提示解决问题。")
        print("\n常见解决方案：")
        print("\n1. 安装核心依赖包：")
        print("   pip install PyQt5 opencv-python numpy")
        print("\n2. 安装 ONNX Runtime（推荐）：")
        print("   pip install onnxruntime")
        print("   或使用 GPU 版本：pip install onnxruntime-gpu")
        print("\n3. 或安装完整依赖（包括 PyTorch）：")
        print("   pip install -r requirements.txt")
        print("\n4. 添加模型文件：")
        print("   将训练好的 best.onnx 或 best.pt 文件放在程序目录下")
    
    print("\n" + "=" * 60)
    input("\n按回车键退出...")

if __name__ == "__main__":
    main()

