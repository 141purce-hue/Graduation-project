# 病虫害识别系统配置文件

# 模型配置
MODEL_PATH = "best.onnx"  # 模型文件路径（支持 .onnx 或 .pt 格式）
CONFIDENCE_THRESHOLD = 0.5  # 置信度阈值 (0.0-1.0)

# 窗口配置
WINDOW_WIDTH = 1400  # 窗口宽度
WINDOW_HEIGHT = 900  # 窗口高度
WINDOW_TITLE = "🌿 番茄病虫害智能识别系统"

# 类别配置
CLASS_NAMES = [
    "Healthy Leaf",
    "Leaf Mold",
    "Septoria leaf spot",
    "Tomato leaf bacterial spot"
]

# 类别中文名称
CLASS_NAMES_CN = {
    "Healthy Leaf": "健康叶片",
    "Leaf Mold": "叶霉病",
    "Septoria leaf spot": "斑点病",
    "Tomato leaf bacterial spot": "细菌性斑点病"
}

# 类别颜色配置（用于显示）
CLASS_COLORS = {
    "Healthy Leaf": "#27ae60",  # 绿色
    "Leaf Mold": "#e67e22",  # 橙色
    "Septoria leaf spot": "#e74c3c",  # 红色
    "Tomato leaf bacterial spot": "#9b59b6"  # 紫色
}

# 病害治理方式
DISEASE_TREATMENTS = {
    "Healthy Leaf": {
        "name": "健康叶片",
        "description": "叶片健康，无需处理",
        "treatment": "• 继续保持良好的种植环境\n• 定期检查叶片状态\n• 适当施肥和浇水\n• 保持通风良好"
    },
    "Leaf Mold": {
        "name": "叶霉病",
        "description": "由真菌引起的叶部病害",
        "treatment": "• 及时摘除病叶并销毁\n• 喷施百菌清或甲基托布津\n• 降低棚内湿度，加强通风\n• 避免叶面长时间潮湿"
    },
    "Septoria leaf spot": {
        "name": "斑点病",
        "description": "叶片出现褐色斑点",
        "treatment": "• 清除病叶，减少病源\n• 喷施代森锰锌或百菌清\n• 避免过度密植\n• 合理灌溉，避免叶面积水"
    },
    "Tomato leaf bacterial spot": {
        "name": "细菌性斑点病",
        "description": "由细菌引起的叶部病害",
        "treatment": "• 发病初期喷施铜制剂\n• 使用农用链霉素防治\n• 避免田间作业时传播\n• 实行轮作，减少病菌积累"
    }
}

# 视频处理配置
VIDEO_FRAME_DELAY = 1  # 视频帧延迟（毫秒），设为1实现正常速度播放

# ESP32S3CAM 摄像头配置
CAMERA_URL = "http://192.168.5.1:81/stream"  # 摄像头流地址
CAMERA_WIFI_SSID = "HW_ESP32S3CAM"  # 摄像头 WiFi 名称（注意是下划线）
CAMERA_FRAME_DELAY = 10  # 摄像头帧延迟（毫秒），值越小越流畅但CPU占用越高
CAMERA_DETECT_INTERVAL = 2  # 检测间隔（帧数），每N帧检测一次，值越大性能越好但检测延迟越高

# 支持的文件格式
IMAGE_FORMATS = "图片文件 (*.jpg *.jpeg *.png *.bmp);;所有文件 (*.*)"
VIDEO_FORMATS = "视频文件 (*.mp4 *.avi *.mov *.mkv);;所有文件 (*.*)"

