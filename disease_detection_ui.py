import sys
import cv2
import numpy as np
import subprocess
from pathlib import Path
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QLabel, QFileDialog,
                             QRadioButton, QButtonGroup, QFrame, QTextEdit,
                             QProgressBar, QGroupBox, QMessageBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QPixmap, QImage, QFont, QPalette, QColor
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False

from config import *


def check_wifi_connection(target_ssid="HW_ESP32S3CAM"):
    """
    检查当前 WiFi 连接是否为指定的 SSID

    Args:
        target_ssid: 目标 WiFi 名称

    Returns:
        tuple: (是否连接, 当前SSID, 错误信息)
    """
    try:
        # Windows 系统使用 netsh 命令
        result = subprocess.run(
            ['netsh', 'wlan', 'show', 'interfaces'],
            capture_output=True,
            text=False  # 获取字节数据，避免编码问题
        )

        if result.returncode == 0:
            # 尝试多种编码方式解码，避免编码错误
            output = None
            for encoding in ['gbk', 'utf-8', 'cp936', 'gb2312']:
                try:
                    output = result.stdout.decode(encoding, errors='ignore')
                    break
                except:
                    continue

            if output is None:
                return False, "", "无法解码 WiFi 信息"

            # 查找 SSID 行
            for line in output.split('\n'):
                if 'SSID' in line and ':' in line:
                    # 提取 SSID 名称
                    parts = line.split(':', 1)
                    if len(parts) == 2:
                        current_ssid = parts[1].strip()
                        # 跳过 BSSID 行
                        if 'BSSID' not in line:
                            is_connected = (current_ssid == target_ssid)
                            return is_connected, current_ssid, None

            return False, "未连接到任何 WiFi", None
        else:
            return False, "", "无法获取 WiFi 信息"

    except Exception as e:
        return False, "", f"检查 WiFi 连接时出错: {str(e)}"


class ONNXModel:
    """ONNX模型推理类"""
    def __init__(self, model_path, class_names):
        """
        初始化ONNX模型

        Args:
            model_path: ONNX模型文件路径
            class_names: 类别名称列表
        """
        self.model_path = model_path
        self.class_names = class_names
        self.session = None
        self.input_name = None
        self.output_names = None
        self.input_shape = None

        # 加载模型
        self._load_model()

    def _load_model(self):
        """加载ONNX模型"""
        # 创建推理会话
        providers = ['CPUExecutionProvider']

        # 如果有GPU，优先使用CUDA
        if 'CUDAExecutionProvider' in ort.get_available_providers():
            providers.insert(0, 'CUDAExecutionProvider')

        self.session = ort.InferenceSession(self.model_path, providers=providers)

        # 获取输入输出信息
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]
        self.input_shape = self.session.get_inputs()[0].shape

        # 获取输入尺寸（通常是 [batch, channels, height, width]）
        if len(self.input_shape) == 4:
            self.img_size = self.input_shape[2]  # 假设高宽相同
        else:
            self.img_size = 640  # 默认值

    def preprocess(self, image):
        """
        预处理图像

        Args:
            image: OpenCV格式的图像 (BGR)

        Returns:
            preprocessed: 预处理后的图像
            ratio: 缩放比例
            (dw, dh): 填充大小
        """
        # 获取原始图像尺寸
        img_h, img_w = image.shape[:2]

        # 计算缩放比例
        ratio = min(self.img_size / img_h, self.img_size / img_w)
        new_h, new_w = int(img_h * ratio), int(img_w * ratio)

        # 缩放图像
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # 创建填充后的图像
        dh, dw = (self.img_size - new_h) // 2, (self.img_size - new_w) // 2
        padded = np.full((self.img_size, self.img_size, 3), 114, dtype=np.uint8)
        padded[dh:dh+new_h, dw:dw+new_w] = resized

        # 转换为RGB并归一化
        padded = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        padded = padded.transpose(2, 0, 1).astype(np.float32) / 255.0

        # 添加batch维度
        padded = np.expand_dims(padded, axis=0)

        return padded, ratio, (dw, dh)

    def postprocess(self, outputs, ratio, pad, conf_threshold=0.5, iou_threshold=0.45):
        """
        后处理模型输出

        Args:
            outputs: 模型输出
            ratio: 缩放比例
            pad: 填充大小
            conf_threshold: 置信度阈值
            iou_threshold: NMS的IOU阈值

        Returns:
            boxes: 检测框 [x1, y1, x2, y2, conf, cls]
        """
        # 获取输出
        predictions = outputs[0]

        # 处理不同的输出格式
        # YOLOv8 ONNX 输出格式: [batch, 4+num_classes, num_boxes]
        # 需要转置为: [batch, num_boxes, 4+num_classes]
        if len(predictions.shape) == 3:
            # 如果是 [1, 8, 2100] 格式，转置为 [1, 2100, 8]
            if predictions.shape[1] < predictions.shape[2]:
                predictions = predictions.transpose(0, 2, 1)

            # 移除batch维度
            predictions = predictions[0]  # 现在是 [num_boxes, 4+num_classes]

        # 分离坐标和类别概率
        # predictions 现在应该是 [num_boxes, 4+num_classes]
        # 前4列是边界框坐标 [x, y, w, h]
        # 后面的列是类别概率
        boxes = predictions[:, :4]  # [num_boxes, 4]
        class_scores = predictions[:, 4:]  # [num_boxes, num_classes]

        # 获取每个框的最大类别概率和对应的类别ID
        class_ids = np.argmax(class_scores, axis=1)
        confidences = np.max(class_scores, axis=1)

        # 过滤低置信度的框
        mask = confidences > conf_threshold
        boxes = boxes[mask]
        confidences = confidences[mask]
        class_ids = class_ids[mask]

        if len(boxes) == 0:
            return np.array([])

        # 转换坐标格式 (中心点+宽高 -> 左上角+右下角)
        x_center, y_center, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        x1 = x_center - w / 2
        y1 = y_center - h / 2
        x2 = x_center + w / 2
        y2 = y_center + h / 2

        # 还原到原始图像坐标
        dw, dh = pad
        x1 = (x1 - dw) / ratio
        y1 = (y1 - dh) / ratio
        x2 = (x2 - dw) / ratio
        y2 = (y2 - dh) / ratio

        # NMS
        indices = cv2.dnn.NMSBoxes(
            boxes.tolist(),
            confidences.tolist(),
            conf_threshold,
            iou_threshold
        )

        if len(indices) == 0:
            return np.array([])

        # 组合结果
        results = []
        for i in indices.flatten():
            results.append([x1[i], y1[i], x2[i], y2[i], confidences[i], class_ids[i]])

        return np.array(results)

    def predict(self, image, conf_threshold=0.5):
        """
        预测图像

        Args:
            image: OpenCV格式的图像
            conf_threshold: 置信度阈值

        Returns:
            boxes: 检测结果 [x1, y1, x2, y2, conf, cls]
        """
        # 预处理
        input_data, ratio, pad = self.preprocess(image)

        # 推理
        outputs = self.session.run(self.output_names, {self.input_name: input_data})

        # 后处理
        boxes = self.postprocess(outputs, ratio, pad, conf_threshold)

        return boxes

    def draw_boxes(self, image, boxes):
        """
        在图像上绘制检测框

        Args:
            image: 原始图像
            boxes: 检测框

        Returns:
            annotated_image: 标注后的图像
        """
        annotated = image.copy()

        # 定义颜色
        colors = {
            0: (39, 174, 96),   # 绿色 - Healthy Leaf
            1: (230, 126, 34),  # 橙色 - Leaf Mold
            2: (231, 76, 60),   # 红色 - Septoria leaf spot
            3: (155, 89, 182)   # 紫色 - Tomato leaf bacterial spot
        }

        for box in boxes:
            x1, y1, x2, y2, conf, cls = box
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cls = int(cls)

            # 获取颜色
            color = colors.get(cls, (52, 152, 219))  # 默认蓝色

            # 绘制矩形框
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # 准备标签文本
            label = f"{self.class_names[cls]}: {conf:.2%}"

            # 计算文本大小
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )

            # 绘制标签背景
            cv2.rectangle(
                annotated,
                (x1, y1 - text_height - baseline - 5),
                (x1 + text_width, y1),
                color,
                -1
            )

            # 绘制文本
            cv2.putText(
                annotated,
                label,
                (x1, y1 - baseline - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

        return annotated


class VideoThread(QThread):
    """视频处理线程"""
    change_pixmap_signal = pyqtSignal(np.ndarray)
    result_signal = pyqtSignal(str, float)
    finished_signal = pyqtSignal()

    def __init__(self, video_path, model, is_onnx=False):
        super().__init__()
        self.video_path = video_path
        self.model = model
        self.is_onnx = is_onnx
        self.running = True

    def run(self):
        cap = cv2.VideoCapture(self.video_path)

        # 获取视频的帧率，用于正常速度播放
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30  # 默认30fps
        frame_delay = int(1000 / fps)  # 计算每帧延迟（毫秒）

        while self.running and cap.isOpened():
            ret, frame = cap.read()
            if ret:
                if self.is_onnx:
                    # ONNX模型推理
                    boxes = self.model.predict(frame, conf_threshold=CONFIDENCE_THRESHOLD)

                    # 绘制结果
                    annotated_frame = self.model.draw_boxes(frame, boxes)

                    # 发送检测结果
                    for box in boxes:
                        cls_id = int(box[5])
                        conf = float(box[4])
                        class_name = self.model.class_names[cls_id]
                        self.result_signal.emit(class_name, conf)
                else:
                    # Ultralytics YOLO推理
                    results = self.model.predict(source=frame, conf=CONFIDENCE_THRESHOLD, verbose=False)

                    # 绘制结果
                    annotated_frame = results[0].plot()

                    # 获取预测结果
                    if len(results[0].boxes) > 0:
                        for box in results[0].boxes:
                            cls_id = int(box.cls[0])
                            conf = float(box.conf[0])
                            class_name = results[0].names[cls_id]
                            self.result_signal.emit(class_name, conf)

                self.change_pixmap_signal.emit(annotated_frame)
                self.msleep(frame_delay)  # 根据视频帧率控制播放速度
            else:
                break

        cap.release()
        self.finished_signal.emit()

    def stop(self):
        self.running = False
        self.wait()


class CameraThread(QThread):
    """ESP32S3CAM 摄像头处理线程"""
    change_pixmap_signal = pyqtSignal(np.ndarray)
    result_signal = pyqtSignal(str, float)
    error_signal = pyqtSignal(str)

    def __init__(self, camera_url, model, is_onnx=False):
        super().__init__()
        self.camera_url = camera_url
        self.model = model
        self.is_onnx = is_onnx
        self.running = True

    def run(self):
        """运行摄像头流识别"""
        cap = None
        try:
            # 连接摄像头流（设置超时）
            import os
            os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'timeout;5000000'  # 5秒超时

            cap = cv2.VideoCapture(self.camera_url, cv2.CAP_FFMPEG)

            # 等待连接建立（最多等待5秒）
            max_retries = 10
            retry_count = 0
            while retry_count < max_retries and not cap.isOpened() and self.running:
                self.msleep(500)
                retry_count += 1

            if not cap.isOpened():
                self.error_signal.emit("无法连接到摄像头流\n\n可能原因：\n1. WiFi 未连接到摄像头\n2. 摄像头未开启\n3. 摄像头地址不正确\n4. 网络连接问题")
                return

            # 优化设置以减少延迟
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 最小缓冲区，减少延迟
            cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)
            cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000)

            # 尝试设置更低的分辨率以提高帧率（如果摄像头支持）
            # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

            frame_count = 0
            consecutive_failures = 0
            max_consecutive_failures = 10

            # 跳帧检测：每N帧检测一次，其他帧直接显示
            detect_interval = CAMERA_DETECT_INTERVAL  # 从配置文件读取
            last_annotated_frame = None
            last_boxes = []

            while self.running:
                ret, frame = cap.read()
                if ret:
                    frame_count += 1
                    consecutive_failures = 0

                    # 跳帧检测策略：只在特定帧进行检测
                    should_detect = (frame_count % detect_interval == 0)

                    if should_detect:
                        # 进行目标检测
                        if self.is_onnx:
                            # ONNX模型推理
                            boxes = self.model.predict(frame, conf_threshold=CONFIDENCE_THRESHOLD)
                            annotated_frame = self.model.draw_boxes(frame.copy(), boxes)
                            last_boxes = boxes

                            # 发送检测结果（只在检测时发送）
                            for box in boxes:
                                cls_id = int(box[5])
                                conf = float(box[4])
                                class_name = self.model.class_names[cls_id]
                                self.result_signal.emit(class_name, conf)
                        else:
                            # YOLO模型推理
                            results = self.model(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)
                            annotated_frame = results[0].plot()

                            # 发送检测结果
                            for box in results[0].boxes:
                                cls_id = int(box.cls[0])
                                conf = float(box.conf[0])
                                class_name = results[0].names[cls_id]
                                self.result_signal.emit(class_name, conf)

                        last_annotated_frame = annotated_frame
                    else:
                        # 不检测的帧：使用上一次的检测结果绘制
                        if last_annotated_frame is not None and self.is_onnx and len(last_boxes) > 0:
                            # 在当前帧上绘制上一次的检测框
                            annotated_frame = self.model.draw_boxes(frame.copy(), last_boxes)
                        elif last_annotated_frame is not None:
                            # 如果有上一帧的结果，直接使用当前帧（不绘制框）
                            annotated_frame = frame
                        else:
                            # 第一帧，直接显示原始画面
                            annotated_frame = frame

                    # 发送画面更新
                    self.change_pixmap_signal.emit(annotated_frame)

                    # 使用配置的延迟时间
                    self.msleep(CAMERA_FRAME_DELAY)

                else:
                    # 读取失败
                    consecutive_failures += 1

                    if consecutive_failures >= max_consecutive_failures:
                        self.error_signal.emit("摄像头连接不稳定，已断开\n\n请检查：\n1. WiFi 信号强度\n2. 摄像头是否正常工作")
                        break

                    # 短暂等待后继续尝试
                    self.msleep(100)

        except Exception as e:
            import traceback
            error_detail = traceback.format_exc()
            self.error_signal.emit(f"摄像头识别错误: {str(e)}\n\n详细信息:\n{error_detail}")
        finally:
            # 确保释放摄像头资源
            if cap is not None:
                cap.release()

    def stop(self):
        """停止摄像头识别"""
        self.running = False
        # 等待线程结束，最多等待3秒
        if not self.wait(3000):
            # 如果3秒后还没结束，强制终止
            self.terminate()
            self.wait()


class DiseaseDetectionUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.model = None
        self.video_thread = None
        self.camera_thread = None
        self.class_names = CLASS_NAMES
        self.is_onnx = False  # 标记是否使用ONNX模型
        self.camera_url = "http://192.168.5.1:81/stream"  # ESP32S3CAM 摄像头地址
        self.camera_wifi_ssid = "HW_ESP32S3CAM"  # 摄像头 WiFi 名称
        self.init_ui()
        self.load_model()

    def init_ui(self):
        """初始化UI界面"""
        self.setWindowTitle(WINDOW_TITLE)
        self.setGeometry(100, 100, WINDOW_WIDTH, WINDOW_HEIGHT)

        # 设置整体样式
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f0f4f8;
            }
            QLabel {
                color: #2c3e50;
            }
            QPushButton {
                background-color: #3498db;
                color: white;
                border: none;
                padding: 12px 24px;
                border-radius: 8px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
            QPushButton:pressed {
                background-color: #21618c;
            }
            QPushButton:disabled {
                background-color: #bdc3c7;
            }
            QRadioButton {
                color: #2c3e50;
                font-size: 13px;
                spacing: 8px;
            }
            QRadioButton::indicator {
                width: 18px;
                height: 18px;
            }
            QGroupBox {
                font-weight: bold;
                font-size: 14px;
                color: #2c3e50;
                border: 2px solid #bdc3c7;
                border-radius: 8px;
                margin-top: 12px;
                padding-top: 12px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 15px;
                padding: 0 5px;
            }
            QTextEdit {
                background-color: white;
                border: 2px solid #bdc3c7;
                border-radius: 8px;
                padding: 8px;
                font-size: 12px;
            }
        """)

        # 主窗口部件
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)
        main_layout.setSpacing(20)
        main_layout.setContentsMargins(20, 20, 20, 20)

        # 左侧控制面板
        left_panel = self.create_left_panel()
        main_layout.addWidget(left_panel, 1)

        # 右侧显示区域
        right_panel = self.create_right_panel()
        main_layout.addWidget(right_panel, 2)

    def create_left_panel(self):
        """创建左侧控制面板"""
        panel = QFrame()
        panel.setStyleSheet("""
            QFrame {
                background-color: white;
                border-radius: 12px;
                padding: 15px;
            }
        """)
        layout = QVBoxLayout(panel)
        layout.setSpacing(20)

        # 标题
        title = QLabel("🎯 控制面板")
        title.setFont(QFont("Arial", 18, QFont.Bold))
        title.setStyleSheet("color: #2c3e50; padding: 10px;")
        layout.addWidget(title)

        # 识别模式选择
        mode_group = QGroupBox("识别模式")
        mode_layout = QVBoxLayout()

        self.mode_group = QButtonGroup()
        self.image_radio = QRadioButton("📷 图片识别")
        self.video_radio = QRadioButton("🎥 视频识别")
        self.camera_radio = QRadioButton("📹 摄像头识别 (ESP32S3CAM)")
        self.image_radio.setChecked(True)

        self.mode_group.addButton(self.image_radio, 1)
        self.mode_group.addButton(self.video_radio, 2)
        self.mode_group.addButton(self.camera_radio, 3)

        mode_layout.addWidget(self.image_radio)
        mode_layout.addWidget(self.video_radio)
        mode_layout.addWidget(self.camera_radio)
        mode_group.setLayout(mode_layout)
        layout.addWidget(mode_group)

        # 连接模式切换信号
        self.mode_group.buttonClicked.connect(self.on_mode_changed)

        # 文件选择按钮
        self.select_btn = QPushButton("📁 选择文件")
        self.select_btn.clicked.connect(self.select_file)
        self.select_btn.setStyleSheet("""
            QPushButton {
                background-color: #27ae60;
                font-size: 15px;
                padding: 15px;
            }
            QPushButton:hover {
                background-color: #229954;
            }
        """)
        layout.addWidget(self.select_btn)

        # 开始识别按钮
        self.detect_btn = QPushButton("🔍 开始识别")
        self.detect_btn.clicked.connect(self.start_detection)
        self.detect_btn.setEnabled(False)
        self.detect_btn.setStyleSheet("""
            QPushButton {
                background-color: #e74c3c;
                font-size: 15px;
                padding: 15px;
            }
            QPushButton:hover {
                background-color: #c0392b;
            }
        """)
        layout.addWidget(self.detect_btn)

        # 停止按钮（仅视频模式）
        self.stop_btn = QPushButton("⏹ 停止识别")
        self.stop_btn.clicked.connect(self.stop_detection)
        self.stop_btn.setEnabled(False)
        self.stop_btn.setStyleSheet("""
            QPushButton {
                background-color: #95a5a6;
                font-size: 15px;
                padding: 15px;
            }
            QPushButton:hover {
                background-color: #7f8c8d;
            }
        """)
        layout.addWidget(self.stop_btn)

        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 2px solid #bdc3c7;
                border-radius: 8px;
                text-align: center;
                height: 25px;
            }
            QProgressBar::chunk {
                background-color: #3498db;
                border-radius: 6px;
            }
        """)
        layout.addWidget(self.progress_bar)

        # 类别说明
        info_group = QGroupBox("📋 识别类别")
        info_layout = QVBoxLayout()

        for i, class_name in enumerate(self.class_names):
            color = ["#27ae60", "#e67e22", "#e74c3c", "#9b59b6"][i]
            # 使用中文名称
            cn_name = CLASS_NAMES_CN.get(class_name, class_name)
            label = QLabel(f"● {cn_name}")
            label.setStyleSheet(f"color: {color}; font-size: 13px; padding: 5px;")
            info_layout.addWidget(label)

        info_group.setLayout(info_layout)
        layout.addWidget(info_group)

        # 病害治理说明
        treatment_group = QGroupBox("💊 病害治理方式")
        treatment_layout = QVBoxLayout()

        self.treatment_text = QTextEdit()
        self.treatment_text.setReadOnly(True)
        self.treatment_text.setMaximumHeight(250)
        self.treatment_text.setStyleSheet("""
            QTextEdit {
                background-color: #fefefe;
                border: 1px solid #d5d8dc;
                border-radius: 6px;
                padding: 10px;
                font-size: 12px;
                line-height: 1.6;
                color: #2c3e50;
            }
        """)
        self.treatment_text.setHtml("""
            <div style='color: #7f8c8d; text-align: center; padding: 20px;'>
                <p>👆 请先进行识别</p>
                <p style='font-size: 11px;'>识别后将显示对应的治理方式</p>
            </div>
        """)

        treatment_layout.addWidget(self.treatment_text)
        treatment_group.setLayout(treatment_layout)
        layout.addWidget(treatment_group)

        # 状态信息
        self.status_label = QLabel("📊 状态: 就绪")
        self.status_label.setStyleSheet("""
            background-color: #ecf0f1;
            padding: 10px;
            border-radius: 8px;
            font-size: 13px;
        """)
        layout.addWidget(self.status_label)

        layout.addStretch()
        return panel

    def create_right_panel(self):
        """创建右侧显示区域"""
        panel = QFrame()
        panel.setStyleSheet("""
            QFrame {
                background-color: white;
                border-radius: 12px;
                padding: 15px;
            }
        """)
        layout = QVBoxLayout(panel)
        layout.setSpacing(15)

        # 标题
        title = QLabel("🖼️ 显示区域")
        title.setFont(QFont("Arial", 18, QFont.Bold))
        title.setStyleSheet("color: #2c3e50; padding: 10px;")
        layout.addWidget(title)

        # 图像显示区域
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("""
            QLabel {
                background-color: #ecf0f1;
                border: 3px dashed #bdc3c7;
                border-radius: 12px;
                min-height: 450px;
                font-size: 16px;
                color: #7f8c8d;
            }
        """)
        self.image_label.setText("📸 请选择图片或视频文件")
        layout.addWidget(self.image_label, 3)

        # 结果显示区域
        result_group = QGroupBox("🔬 识别结果")
        result_layout = QVBoxLayout()

        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        self.result_text.setMaximumHeight(200)
        self.result_text.setStyleSheet("""
            QTextEdit {
                font-family: 'Consolas', 'Monaco', monospace;
                font-size: 13px;
                line-height: 1.5;
            }
        """)
        result_layout.addWidget(self.result_text)

        result_group.setLayout(result_layout)
        layout.addWidget(result_group, 1)

        return panel

    def load_model(self):
        """加载模型（支持ONNX和YOLO格式）"""
        try:
            self.status_label.setText("📊 状态: 正在加载模型...")
            self.progress_bar.setValue(50)

            # 从配置文件读取模型路径
            model_path = MODEL_PATH

            # 检查模型文件是否存在
            if not Path(model_path).exists():
                self.status_label.setText("📊 状态: 模型文件未找到 ✗")
                self.add_result_text(f"❌ 模型文件未找到: {model_path}", "#e74c3c")
                self.add_result_text("请将训练好的模型文件放在程序目录下", "#e67e22")
                return

            # 判断模型类型
            if model_path.endswith('.onnx'):
                # 加载ONNX模型
                if not ONNX_AVAILABLE:
                    self.status_label.setText("📊 状态: ONNX Runtime未安装 ✗")
                    self.add_result_text("❌ 请先安装 onnxruntime: pip install onnxruntime", "#e74c3c")
                    return

                self.add_result_text("🔄 正在加载ONNX模型...", "#3498db")
                self.model = ONNXModel(model_path, self.class_names)
                self.is_onnx = True
                self.status_label.setText("📊 状态: ONNX模型加载成功 ✓")
                self.progress_bar.setValue(100)
                self.add_result_text("✅ ONNX模型加载成功！", "#27ae60")
                self.add_result_text(f"📐 输入尺寸: {self.model.img_size}x{self.model.img_size}", "#3498db")
            else:
                # 加载Ultralytics YOLO模型
                if not ULTRALYTICS_AVAILABLE:
                    self.status_label.setText("📊 状态: Ultralytics未安装 ✗")
                    self.add_result_text("❌ 请先安装 ultralytics: pip install ultralytics", "#e74c3c")
                    return

                self.add_result_text("🔄 正在加载YOLO模型...", "#3498db")
                self.model = YOLO(model_path)
                self.is_onnx = False
                self.status_label.setText("📊 状态: YOLO模型加载成功 ✓")
                self.progress_bar.setValue(100)
                self.add_result_text("✅ YOLO模型加载成功！", "#27ae60")

        except Exception as e:
            self.status_label.setText("📊 状态: 模型加载失败 ✗")
            self.add_result_text(f"❌ 模型加载失败: {str(e)}", "#e74c3c")
            import traceback
            self.add_result_text(f"详细错误: {traceback.format_exc()}", "#95a5a6")

    def on_mode_changed(self):
        """识别模式切换"""
        if self.camera_radio.isChecked():
            # 摄像头模式，隐藏文件选择按钮
            self.select_btn.setEnabled(False)
            self.select_btn.setText("📹 摄像头模式")
            self.detect_btn.setEnabled(True)
            self.status_label.setText("📊 状态: 摄像头模式就绪")
        else:
            # 文件模式，显示文件选择按钮
            self.select_btn.setEnabled(True)
            if self.image_radio.isChecked():
                self.select_btn.setText("📁 选择图片")
            else:
                self.select_btn.setText("📁 选择视频")
            self.detect_btn.setEnabled(False)
            self.status_label.setText("📊 状态: 请选择文件")

    def select_file(self):
        """选择文件"""
        if self.camera_radio.isChecked():
            # 摄像头模式不需要选择文件
            return

        if self.image_radio.isChecked():
            file_path, _ = QFileDialog.getOpenFileName(
                self, "选择图片", "",
                IMAGE_FORMATS
            )
        else:
            file_path, _ = QFileDialog.getOpenFileName(
                self, "选择视频", "",
                VIDEO_FORMATS
            )

        if file_path:
            self.current_file = file_path
            self.detect_btn.setEnabled(True)
            self.status_label.setText(f"📊 状态: 已选择文件")
            self.add_result_text(f"📁 已选择: {Path(file_path).name}", "#3498db")

            # 如果是图片，显示预览
            if self.image_radio.isChecked():
                self.display_image(file_path)

    def display_image(self, image_path):
        """显示图片"""
        pixmap = QPixmap(image_path)
        scaled_pixmap = pixmap.scaled(
            self.image_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.image_label.setPixmap(scaled_pixmap)

    def display_frame(self, frame):
        """显示视频帧"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        scaled_pixmap = pixmap.scaled(
            self.image_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.image_label.setPixmap(scaled_pixmap)

    def start_detection(self):
        """开始识别"""
        if not self.model:
            self.add_result_text("❌ 模型未加载，无法进行识别", "#e74c3c")
            return

        self.result_text.clear()
        self.progress_bar.setValue(0)

        if self.image_radio.isChecked():
            self.detect_image()
        elif self.video_radio.isChecked():
            self.detect_video()
        else:  # 摄像头模式
            self.detect_camera()

    def detect_image(self):
        """图片识别"""
        try:
            self.status_label.setText("📊 状态: 正在识别...")
            self.progress_bar.setValue(30)
            self.add_result_text("🔍 开始图片识别...", "#3498db")

            # 读取图像
            image = cv2.imread(self.current_file)

            if self.is_onnx:
                # ONNX模型推理
                self.add_result_text("🔄 使用ONNX模型进行推理...", "#3498db")
                boxes = self.model.predict(image, conf_threshold=CONFIDENCE_THRESHOLD)

                self.progress_bar.setValue(70)

                # 绘制结果
                annotated_img = self.model.draw_boxes(image, boxes)
                self.display_frame(annotated_img)

                # 解析结果
                if len(boxes) > 0:
                    self.add_result_text("\n✅ 检测结果:", "#27ae60")
                    detected_classes = set()
                    for i, box in enumerate(boxes):
                        cls_id = int(box[5])
                        conf = float(box[4])
                        class_name = self.class_names[cls_id]
                        cn_name = CLASS_NAMES_CN.get(class_name, class_name)
                        detected_classes.add(class_name)

                        color = CLASS_COLORS.get(class_name, "#3498db")
                        self.add_result_text(
                            f"  {i+1}. {cn_name} - 置信度: {conf:.2%}",
                            color
                        )

                    # 更新治理方式显示
                    self.update_treatment_info(detected_classes)
                else:
                    self.add_result_text("ℹ️ 未检测到病虫害", "#95a5a6")
                    self.clear_treatment_info()
            else:
                # Ultralytics YOLO推理
                self.add_result_text("🔄 使用YOLO模型进行推理...", "#3498db")
                results = self.model.predict(source=self.current_file, conf=CONFIDENCE_THRESHOLD, verbose=False)

                self.progress_bar.setValue(70)

                # 显示结果图片
                annotated_img = results[0].plot()
                self.display_frame(annotated_img)

                # 解析结果
                if len(results[0].boxes) > 0:
                    self.add_result_text("\n✅ 检测结果:", "#27ae60")
                    detected_classes = set()
                    for i, box in enumerate(results[0].boxes):
                        cls_id = int(box.cls[0])
                        conf = float(box.conf[0])
                        class_name = results[0].names[cls_id]
                        cn_name = CLASS_NAMES_CN.get(class_name, class_name)
                        detected_classes.add(class_name)

                        color = CLASS_COLORS.get(class_name, "#3498db")
                        self.add_result_text(
                            f"  {i+1}. {cn_name} - 置信度: {conf:.2%}",
                            color
                        )

                    # 更新治理方式显示
                    self.update_treatment_info(detected_classes)
                else:
                    self.add_result_text("ℹ️ 未检测到病虫害", "#95a5a6")
                    self.clear_treatment_info()

            self.progress_bar.setValue(100)
            self.status_label.setText("📊 状态: 识别完成 ✓")

        except Exception as e:
            self.add_result_text(f"❌ 识别失败: {str(e)}", "#e74c3c")
            self.status_label.setText("📊 状态: 识别失败 ✗")
            import traceback
            self.add_result_text(f"详细错误: {traceback.format_exc()}", "#95a5a6")

    def detect_video(self):
        """视频识别"""
        try:
            self.status_label.setText("📊 状态: 正在识别视频...")
            self.add_result_text("🎥 开始视频识别...", "#3498db")

            if self.is_onnx:
                self.add_result_text("🔄 使用ONNX模型进行视频推理...", "#3498db")
            else:
                self.add_result_text("🔄 使用YOLO模型进行视频推理...", "#3498db")

            # 禁用按钮
            self.detect_btn.setEnabled(False)
            self.select_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)

            # 创建并启动视频线程
            self.video_thread = VideoThread(self.current_file, self.model, self.is_onnx)
            self.video_thread.change_pixmap_signal.connect(self.display_frame)
            self.video_thread.result_signal.connect(self.update_video_result)
            self.video_thread.finished_signal.connect(self.video_finished)
            self.video_thread.start()

        except Exception as e:
            self.add_result_text(f"❌ 视频识别失败: {str(e)}", "#e74c3c")
            self.status_label.setText("📊 状态: 识别失败 ✗")
            self.detect_btn.setEnabled(True)
            self.select_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
            import traceback
            self.add_result_text(f"详细错误: {traceback.format_exc()}", "#95a5a6")

    def detect_camera(self):
        """摄像头识别"""
        try:
            # 检查 WiFi 连接
            self.add_result_text("🔍 正在检查 WiFi 连接...", "#3498db")
            is_connected, current_ssid, error = check_wifi_connection(self.camera_wifi_ssid)

            if error:
                self.add_result_text(f"❌ WiFi 检查失败: {error}", "#e74c3c")
                QMessageBox.warning(self, "WiFi 检查失败", f"无法检查 WiFi 连接状态\n\n错误: {error}")
                return

            if not is_connected:
                self.add_result_text(f"⚠️  未连接到摄像头 WiFi", "#e67e22")
                self.add_result_text(f"   当前连接: {current_ssid}", "#95a5a6")
                self.add_result_text(f"   需要连接: {self.camera_wifi_ssid}", "#95a5a6")

                reply = QMessageBox.question(
                    self,
                    "WiFi 未连接",
                    f"当前未连接到摄像头 WiFi\n\n"
                    f"当前连接: {current_ssid}\n"
                    f"需要连接: {self.camera_wifi_ssid}\n\n"
                    f"请连接到正确的 WiFi 后重试。\n\n"
                    f"是否继续尝试连接摄像头？",
                    QMessageBox.Yes | QMessageBox.No
                )

                if reply == QMessageBox.No:
                    return
            else:
                self.add_result_text(f"✅ 已连接到摄像头 WiFi: {self.camera_wifi_ssid}", "#27ae60")

            # 开始摄像头识别
            self.status_label.setText("📊 状态: 正在连接摄像头...")
            self.add_result_text("📹 开始摄像头识别...", "#3498db")
            self.add_result_text(f"📡 摄像头地址: {self.camera_url}", "#95a5a6")

            if self.is_onnx:
                self.add_result_text("🔄 使用ONNX模型进行实时推理...", "#3498db")
            else:
                self.add_result_text("🔄 使用YOLO模型进行实时推理...", "#3498db")

            # 禁用按钮
            self.detect_btn.setEnabled(False)
            self.select_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)

            # 创建并启动摄像头线程
            self.camera_thread = CameraThread(self.camera_url, self.model, self.is_onnx)
            self.camera_thread.change_pixmap_signal.connect(self.display_frame)
            self.camera_thread.result_signal.connect(self.update_video_result)
            self.camera_thread.error_signal.connect(self.camera_error)
            self.camera_thread.start()

            self.status_label.setText("📊 状态: 摄像头识别中...")
            self.add_result_text("✅ 摄像头连接成功，开始实时识别", "#27ae60")

        except Exception as e:
            self.add_result_text(f"❌ 摄像头识别失败: {str(e)}", "#e74c3c")
            self.status_label.setText("📊 状态: 识别失败 ✗")
            self.detect_btn.setEnabled(True)
            self.select_btn.setEnabled(False)
            self.stop_btn.setEnabled(False)
            import traceback
            self.add_result_text(f"详细错误: {traceback.format_exc()}", "#95a5a6")

    def camera_error(self, error_msg):
        """摄像头错误处理"""
        self.add_result_text(f"❌ 摄像头错误: {error_msg}", "#e74c3c")
        QMessageBox.critical(self, "摄像头错误", error_msg)
        self.stop_detection()

    def update_video_result(self, class_name, confidence):
        """更新视频识别结果"""
        cn_name = CLASS_NAMES_CN.get(class_name, class_name)
        color = CLASS_COLORS.get(class_name, "#3498db")
        self.add_result_text(f"🔍 检测到: {cn_name} (置信度: {confidence:.2%})", color)

        # 更新治理方式显示（只显示最新检测到的类别）
        self.update_treatment_info({class_name})

    def stop_detection(self):
        """停止视频/摄像头识别"""
        if self.video_thread:
            self.video_thread.stop()
            self.video_thread = None
            self.add_result_text("⏹ 已停止视频识别", "#95a5a6")

        if self.camera_thread:
            self.camera_thread.stop()
            self.camera_thread = None
            self.add_result_text("⏹ 已停止摄像头识别", "#95a5a6")

        # 恢复按钮状态
        self.detect_btn.setEnabled(True)
        self.select_btn.setEnabled(not self.camera_radio.isChecked())
        self.stop_btn.setEnabled(False)
        self.status_label.setText("📊 状态: 已停止")

    def video_finished(self):
        """视频识别完成"""
        self.status_label.setText("📊 状态: 视频识别完成 ✓")
        self.detect_btn.setEnabled(True)
        self.select_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.progress_bar.setValue(100)
        self.add_result_text("✅ 视频识别完成", "#27ae60")

    def update_treatment_info(self, detected_classes):
        """更新治理方式显示"""
        if not detected_classes:
            self.clear_treatment_info()
            return

        html_content = ""
        for class_name in detected_classes:
            if class_name in DISEASE_TREATMENTS:
                treatment = DISEASE_TREATMENTS[class_name]
                color = CLASS_COLORS.get(class_name, "#3498db")

                html_content += f"""
                <div style='margin-bottom: 15px; padding: 10px; background-color: #f8f9fa; border-left: 4px solid {color}; border-radius: 4px;'>
                    <h3 style='color: {color}; margin: 0 0 8px 0; font-size: 14px;'>
                        ● {treatment['name']}
                    </h3>
                    <p style='margin: 5px 0; color: #7f8c8d; font-size: 11px;'>
                        {treatment['description']}
                    </p>
                    <div style='margin-top: 8px; padding: 8px; background-color: white; border-radius: 3px;'>
                        <p style='margin: 0; color: #2c3e50; font-size: 11px; white-space: pre-line;'>
                            {treatment['treatment']}
                        </p>
                    </div>
                </div>
                """

        if html_content:
            self.treatment_text.setHtml(html_content)
        else:
            self.clear_treatment_info()

    def clear_treatment_info(self):
        """清空治理方式显示"""
        self.treatment_text.setHtml("""
            <div style='color: #7f8c8d; text-align: center; padding: 20px;'>
                <p>👆 请先进行识别</p>
                <p style='font-size: 11px;'>识别后将显示对应的治理方式</p>
            </div>
        """)

    def add_result_text(self, text, color="#2c3e50"):
        """添加结果文本"""
        self.result_text.append(f'<span style="color: {color};">{text}</span>')
        # 自动滚动到底部
        self.result_text.verticalScrollBar().setValue(
            self.result_text.verticalScrollBar().maximum()
        )

    def closeEvent(self, event):
        """关闭窗口时停止视频线程"""
        if self.video_thread and self.video_thread.isRunning():
            self.video_thread.stop()
        event.accept()


def main():
    app = QApplication(sys.argv)

    # 设置应用程序字体
    font = QFont("Microsoft YaHei", 10)
    app.setFont(font)

    window = DiseaseDetectionUI()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

