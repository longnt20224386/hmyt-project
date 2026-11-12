# segmentation10_optimized.py
import sys
import cc3d
import numpy as np
import SimpleITK as sitk
import cv2
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QPushButton, QFileDialog, QMessageBox, QAction,
                             QToolBar, QStatusBar, QComboBox, QSpinBox, QDoubleSpinBox, QCheckBox,
                             QDockWidget, QFormLayout, QLineEdit, QDialog, QTextEdit, QGroupBox)
from PyQt5.QtCore import Qt, QPoint, QRect, QTimer
from PyQt5.QtGui import QPainter, QPen, QColor, QImage, QFont
from PIL import Image, ImageDraw
from scipy import ndimage
from scipy.ndimage import measurements


class MedicalSegmentationApp(QMainWindow):
    def __init__(self):
        super().__init__()

        # Data
        self.mask_array = None           # 3D uint8 mask (z,y,x)
        self.ct_image = None
        self.suv_image = None
        self.ct_array = None
        self.suv_array = None
        self.current_slice_idx = 0
        self.current_label = 1
        self.refine_done = False
        self.max_labels = 1000  # tăng giới hạn label để không bị cắt sớm

        # Segmentation
        self.seg_image = None
        self.seg_array = None
        self.loaded_segmentation = False

        # Editing mode
        self.editing_mode = False
        self.selected_lesion_label = None
        self.lesion_edit_threshold = 20.0  # GIẢM NGƯỠNG MẶC ĐỊNH
        self.edited_lesions = set()
        self.manual_editing_done = False
        self.base_refine_mask = None  # lưu mask do người dùng bôi để refine ổn định nhiều lần
        self.edit_mask_array = None
        self.edit_changes_made = False

        #  THÊM: Coordinate system fix
        self.flip_y_coordinates = True  # Khớp ITK-SNAP
        # Giảm mặc định để không cắt mất rìa tổn thương khi refine
        self.refine_threshold_percent = 20.0  # NGƯỠNG REFINE MẶC ĐỊNH

        # THÊM: Transform tracking
        self.transform_history = []
        self.ct_original = None
        self.suv_original = None
        self.original_flipped = False

        # Label colors
        self.label_colors = {
            1: QColor(255, 0, 0, 120),       2: QColor(0, 255, 0, 120),
            3: QColor(0, 0, 255, 120),       4: QColor(255, 255, 0, 120),
            5: QColor(255, 0, 255, 120),     6: QColor(0, 255, 255, 120),
            7: QColor(255, 128, 0, 120),     8: QColor(128, 0, 255, 120),
            9: QColor(0, 128, 255, 120),     10: QColor(255, 0, 128, 120),
            11: QColor(128, 255, 0, 120),    12: QColor(0, 255, 128, 120),
            13: QColor(255, 128, 128, 120),  14: QColor(128, 255, 128, 120),
            15: QColor(128, 128, 255, 120),  16: QColor(192, 192, 192, 120),
            17: QColor(128, 64, 0, 120),     18: QColor(64, 128, 0, 120),
            19: QColor(0, 128, 128, 120),    20: QColor(128, 0, 128, 120),
            21: QColor(212, 175, 55, 120),   22: QColor(250, 128, 114, 120),
            23: QColor(30, 144, 255, 120),   24: QColor(60, 179, 113, 120),
            25: QColor(106, 90, 205, 120),   26: QColor(255, 140, 0, 120),
            27: QColor(255, 20, 147, 120),   28: QColor(70, 130, 180, 120),
            29: QColor(189, 183, 107, 120),  30: QColor(85, 107, 47, 120),
            31: QColor(100, 149, 237, 120),  32: QColor(188, 143, 143, 120),
            33: QColor(210, 105, 30, 120),   34: QColor(186, 85, 211, 120),
            35: QColor(233, 150, 122, 120),  36: QColor(95, 158, 160, 120),
            37: QColor(0, 206, 209, 120),    38: QColor(199, 21, 133, 120),
            39: QColor(160, 82, 45, 120),    40: QColor(219, 112, 147, 120),
            41: QColor(119, 136, 153, 120),  42: QColor(72, 61, 139, 120),
            43: QColor(34, 139, 34, 120),    44: QColor(25, 25, 112, 120),
            45: QColor(205, 133, 63, 120),   46: QColor(64, 224, 208, 120),
            47: QColor(65, 105, 225, 120),   48: QColor(205, 92, 92, 120),
            49: QColor(0, 255, 127, 120),    50: QColor(75, 0, 130, 120),
        }
        # Độ mờ tăng lên để mask rõ hơn trên ảnh
        for color in self.label_colors.values():
            color.setAlpha(200)
        self.base_palette = list(self.label_colors.values())
        self.ensure_label_colors(self.max_labels)

        # View state
        self.ct_view = None
        self.suv_view = None
        self.ct_view_data = {'image_array': None, 'mask': None, 'zoom_factor': 1.0, 'pan_offset': QPoint(0, 0),
                             'window_level': 50, 'window_width': 150, 'original_size': QPoint(512, 512),
                             'show_mask': True, 'suv_colormap': 'jet', 'widget': None}
        self.suv_view_data = {'image_array': None, 'mask': None, 'zoom_factor': 1.0, 'pan_offset': QPoint(0, 0),
                              'window_level': 7.5, 'window_width': 15.0, 'original_size': QPoint(512, 512),
                              'show_mask': True, 'suv_colormap': 'jet', 'widget': None}

        # Interaction
        self.brush_size = 5
        self.drawing = False
        self.last_point = QPoint()
        self.pan_start = QPoint()
        self.sync_views = True
        self.brush_enabled = True
        self.eraser_mode = False

        # Slice wheel debounce
        self.slice_timer = QTimer()
        self.slice_timer.setInterval(50)
        self.slice_timer.timeout.connect(self.process_slice_change)
        self.pending_slice_delta = 0

        # Click info
        self.last_clicked_label = None
        self.last_clicked_position = None
        self.last_clicked_value = None
        self.last_clicked_view = None

        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Công cụ Segmentation PET/CT ")
        self.setGeometry(100, 100, 1400, 900)
        self.create_toolbar()
        self.create_central_widget()
        self.create_status_bar()
        self.create_dock_widgets()
        self.create_edit_toolbar()

    def create_toolbar(self):
        self.toolbar = QToolBar()
        self.addToolBar(Qt.TopToolBarArea, self.toolbar)

        # Brush / Eraser controls (đặt đầu tiên)
        self.brush_toggle_btn = QPushButton("Brush")
        self.brush_toggle_btn.setCheckable(True)
        self.brush_toggle_btn.setChecked(True)
        self.brush_toggle_btn.clicked.connect(self.toggle_brush_enabled)
        self.brush_toggle_btn.setStyleSheet("QPushButton { font-weight: bold; }"
                                            "QPushButton:checked { background-color: #4CAF50; color: white; }"
                                            "QPushButton:!checked { background-color: #777; color: white; }")
        self.toolbar.addWidget(self.brush_toggle_btn)

        self.eraser_toggle_btn = QPushButton("Tẩy")
        self.eraser_toggle_btn.setCheckable(True)
        self.eraser_toggle_btn.setChecked(False)
        self.eraser_toggle_btn.clicked.connect(self.toggle_eraser_mode)
        self.eraser_toggle_btn.setStyleSheet("QPushButton { font-weight: bold; }"
                                             "QPushButton:checked { background-color: #f44336; color: white; }"
                                             "QPushButton:!checked { background-color: #555; color: white; }")
        self.toolbar.addWidget(self.eraser_toggle_btn)

        self.toolbar.addSeparator()

        # File operations
        file_actions = [
            ("Tải ảnh CT và SUV", self.load_images),
            ("Tải Segmentation", self.load_segmentation_only),
            ("Export Mask 3D", self.export_3d_mask)
        ]
        for text, slot in file_actions:
            a = QAction(text, self)
            a.triggered.connect(slot)
            self.toolbar.addAction(a)

        self.toolbar.addSeparator()

        # Mask operations
        mask_actions = [
            ("Xóa Mask Hiện Tại", self.clear_current_slice_mask),
            ("Xóa Tất cả Mask", self.clear_all_masks),
            ("Refine", self.refine_all_masks)
        ]
        for text, slot in mask_actions:
            a = QAction(text, self)
            a.triggered.connect(slot)
            self.toolbar.addAction(a)

        self.toolbar.addSeparator()

        # Statistics
        stats_actions = [
            ("Thống kê Label", self.calculate_volumes_statistics),
            ("Thống kê Slice", self.calculate_current_slice_statistics)
        ]
        for text, slot in stats_actions:
            a = QAction(text, self)
            a.triggered.connect(slot)
            self.toolbar.addAction(a)

        self.toolbar.addSeparator()

        # Label selector
        self.label_selector = QComboBox()
        for label in range(1, self.max_labels + 1):
            color = self.label_colors.get(
                label, self.generate_label_color(label))
            color_name = self.get_color_name(color)
            self.label_selector.addItem(f"Label {label} - {color_name}")
            self.label_selector.setItemData(label-1, color, Qt.DecorationRole)
        self.label_selector.currentIndexChanged.connect(
            self.change_current_label)
        self.toolbar.addWidget(QLabel("Label:"))
        self.toolbar.addWidget(self.label_selector)

        self.clear_label_btn = QPushButton("Xóa Label Hiện Tại")
        self.clear_label_btn.clicked.connect(self.clear_current_label)
        self.toolbar.addWidget(self.clear_label_btn)

        self.toolbar.addSeparator()

        # Slice control
        self.slice_control = QSpinBox()
        self.slice_control.valueChanged.connect(self.change_slice)
        self.toolbar.addWidget(QLabel("Slice:"))
        self.toolbar.addWidget(self.slice_control)

        #  THÊM: Refine threshold control
        self.refine_threshold_control = QDoubleSpinBox()
        self.refine_threshold_control.setRange(10, 60)
        self.refine_threshold_control.setValue(20.0)
        self.refine_threshold_control.setSuffix("%")
        self.refine_threshold_control.valueChanged.connect(
            self.update_refine_threshold)
        self.toolbar.addWidget(QLabel("Ngưỡng Refine:"))
        self.toolbar.addWidget(self.refine_threshold_control)

        # View controls
        self.mask_visibility = QCheckBox("Hiện Mask")
        self.mask_visibility.setChecked(True)
        self.mask_visibility.stateChanged.connect(self.toggle_masks_visibility)
        self.toolbar.addWidget(self.mask_visibility)

        self.sync_views_checkbox = QCheckBox("Đồng bộ Zoom/Pan")
        self.sync_views_checkbox.setChecked(True)
        self.sync_views_checkbox.stateChanged.connect(self.toggle_sync_views)
        self.toolbar.addWidget(self.sync_views_checkbox)

        #  THÊM: Flip Y coordinate toggle
        self.flip_y_checkbox = QCheckBox("Flip Y (ITK-SNAP)")
        self.flip_y_checkbox.setChecked(self.flip_y_coordinates)
        self.flip_y_checkbox.stateChanged.connect(self.toggle_flip_y)
        self.toolbar.addWidget(self.flip_y_checkbox)

        # Brush size
        self.brush_size_control = QSpinBox()
        self.brush_size_control.setRange(1, 50)
        self.brush_size_control.setValue(self.brush_size)
        self.brush_size_control.valueChanged.connect(self.change_brush_size)
        self.toolbar.addWidget(QLabel("Brush:"))
        self.toolbar.addWidget(self.brush_size_control)

        # THÊM: Orientation info button
        self.orientation_btn = QPushButton("Kiểm tra Orientation")
        self.orientation_btn.clicked.connect(self.show_orientation_info)
        self.toolbar.addWidget(self.orientation_btn)

    def create_edit_toolbar(self):
        self.edit_toolbar = QToolBar()
        self.edit_toolbar.setWindowTitle("Chỉnh sửa Tổn thương")
        self.addToolBar(Qt.TopToolBarArea, self.edit_toolbar)

        self.edit_lesions_btn = QPushButton("Chỉnh sửa Tổn thương")
        self.edit_lesions_btn.clicked.connect(self.toggle_edit_lesions_mode)
        self.edit_lesions_btn.setCheckable(True)
        self.edit_lesions_btn.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; font-weight: bold; }"
                                            "QPushButton:checked { background-color: #f44336; }")
        self.edit_toolbar.addWidget(self.edit_lesions_btn)

        edit_group = QWidget()
        edit_layout = QHBoxLayout(edit_group)

        # Lesion selector
        edit_layout.addWidget(QLabel("Chọn tổn thương:"))
        self.lesion_selector = QComboBox()
        self.lesion_selector.setMinimumWidth(250)
        self.lesion_selector.currentIndexChanged.connect(
            self.select_lesion_for_editing)
        edit_layout.addWidget(self.lesion_selector)

        # Lesion threshold
        edit_layout.addWidget(QLabel("Ngưỡng sửa:"))
        self.lesion_threshold_control = QDoubleSpinBox()
        self.lesion_threshold_control.setRange(10, 60)
        self.lesion_threshold_control.setValue(self.lesion_edit_threshold)
        self.lesion_threshold_control.setSuffix("%")
        self.lesion_threshold_control.valueChanged.connect(
            self.update_lesion_threshold)
        edit_layout.addWidget(self.lesion_threshold_control)

        # Action buttons
        self.apply_lesion_threshold_btn = QPushButton(" Refine Tổn thương")
        self.apply_lesion_threshold_btn.clicked.connect(
            self.apply_threshold_to_selected_lesion)
        self.apply_lesion_threshold_btn.setStyleSheet(
            "QPushButton { background-color: #2196F3; color: white; }")
        edit_layout.addWidget(self.apply_lesion_threshold_btn)

        self.delete_lesion_btn = QPushButton("Xóa Tổn thương")
        self.delete_lesion_btn.clicked.connect(self.delete_selected_lesion)
        self.delete_lesion_btn.setStyleSheet(
            "QPushButton { background-color: #ff9800; color: white; }")
        edit_layout.addWidget(self.delete_lesion_btn)

        self.mark_edited_btn = QPushButton("Đánh dấu đã sửa")
        self.mark_edited_btn.clicked.connect(self.mark_lesion_as_edited)
        self.mark_edited_btn.setStyleSheet(
            "QPushButton { background-color: #9C27B0; color: white; }")
        edit_layout.addWidget(self.mark_edited_btn)

        self.sync_changes_btn = QPushButton(" Đồng bộ Thay đổi")
        self.sync_changes_btn.clicked.connect(self.sync_edit_changes)
        self.sync_changes_btn.setStyleSheet(
            "QPushButton { background-color: #4CAF50; color: white; font-weight: bold; }")
        edit_layout.addWidget(self.sync_changes_btn)

        self.cancel_edit_btn = QPushButton(" Hủy Thay đổi")
        self.cancel_edit_btn.clicked.connect(self.cancel_edit_changes)
        self.cancel_edit_btn.setStyleSheet(
            "QPushButton { background-color: #f44336; color: white; }")
        edit_layout.addWidget(self.cancel_edit_btn)

        self.edit_toolbar.addWidget(edit_group)
        self.edit_toolbar.setVisible(False)

    def create_central_widget(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        self.ct_view = QWidget()
        self.suv_view = QWidget()
        self.ct_view.setMouseTracking(True)
        self.suv_view.setMouseTracking(True)
        self.ct_view.setMinimumSize(500, 500)
        self.suv_view.setMinimumSize(500, 500)
        self.ct_view.setFocusPolicy(Qt.StrongFocus)
        self.suv_view.setFocusPolicy(Qt.StrongFocus)

        # Wire events
        self.ct_view.mousePressEvent = lambda event: self.view_mouse_press_event(
            event, "CT")
        self.ct_view.mouseMoveEvent = lambda event: self.view_mouse_move_event(
            event, "CT")
        self.ct_view.mouseReleaseEvent = lambda event: self.view_mouse_release_event(
            event, "CT")
        self.ct_view.wheelEvent = lambda event: self.view_wheel_event(
            event, "CT")
        self.ct_view.paintEvent = lambda event: self.view_paint_event(
            event, "CT")

        self.suv_view.mousePressEvent = lambda event: self.view_mouse_press_event(
            event, "SUV")
        self.suv_view.mouseMoveEvent = lambda event: self.view_mouse_move_event(
            event, "SUV")
        self.suv_view.mouseReleaseEvent = lambda event: self.view_mouse_release_event(
            event, "SUV")
        self.suv_view.wheelEvent = lambda event: self.view_wheel_event(
            event, "SUV")
        self.suv_view.paintEvent = lambda event: self.view_paint_event(
            event, "SUV")

        self.ct_view_data['widget'] = self.ct_view
        self.suv_view_data['widget'] = self.suv_view

        main_layout.addWidget(self.ct_view)
        main_layout.addWidget(self.suv_view)

    def create_status_bar(self):
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

    def create_dock_widgets(self):
        # CT Contrast
        ct_dock = QDockWidget("CT Contrast", self)
        ct_widget = QWidget()
        ct_layout = QFormLayout(ct_widget)

        self.ct_min_edit = QLineEdit("-150")
        self.ct_max_edit = QLineEdit("250")
        self.ct_level_edit = QLineEdit("50")
        self.ct_window_edit = QLineEdit("150")
        for edit in [self.ct_min_edit, self.ct_max_edit, self.ct_level_edit, self.ct_window_edit]:
            edit.editingFinished.connect(self.update_ct_contrast)
        ct_layout.addRow("Minimum:", self.ct_min_edit)
        ct_layout.addRow("Maximum:", self.ct_max_edit)
        ct_layout.addRow("Level:", self.ct_level_edit)
        ct_layout.addRow("Window:", self.ct_window_edit)
        ct_dock.setWidget(ct_widget)
        self.addDockWidget(Qt.LeftDockWidgetArea, ct_dock)

        # SUV Contrast
        suv_dock = QDockWidget("SUV Contrast", self)
        suv_widget = QWidget()
        suv_layout = QFormLayout(suv_widget)
        self.suv_min_edit = QLineEdit("0")
        self.suv_max_edit = QLineEdit("15")
        self.suv_level_edit = QLineEdit("7.5")
        self.suv_window_edit = QLineEdit("15")
        for edit in [self.suv_min_edit, self.suv_max_edit, self.suv_level_edit, self.suv_window_edit]:
            edit.editingFinished.connect(self.update_suv_contrast)
        suv_layout.addRow("Minimum:", self.suv_min_edit)
        suv_layout.addRow("Maximum:", self.suv_max_edit)
        suv_layout.addRow("Level:", self.suv_level_edit)
        suv_layout.addRow("Window:", self.suv_window_edit)
        suv_dock.setWidget(suv_widget)
        self.addDockWidget(Qt.RightDockWidgetArea, suv_dock)

    # -------------------------------
    # COORDINATE SYSTEM FIXES
    # -------------------------------
    def get_active_mask(self):
        """Return the mask currently được hiển thị (ưu tiên mask đang chỉnh sửa)."""
        if self.editing_mode and self.edit_mask_array is not None:
            return self.edit_mask_array
        return self.mask_array

    def map_to_image_coords(self, pos, view_type):
        """Map mouse position to image coordinates - ĐÃ SỬA LỖI FLIP Y"""
        view_data = self.ct_view_data if view_type == "CT" else self.suv_view_data
        if view_data['image_array'] is None:
            return QPoint(-1, -1)

        img_w = int(view_data['original_size'].x() * view_data['zoom_factor'])
        img_h = int(view_data['original_size'].y() * view_data['zoom_factor'])

        display_rect = QRect(
            view_data['widget'].width() // 2 - img_w // 2 +
            view_data['pan_offset'].x(),
            view_data['widget'].height() // 2 - img_h // 2 +
            view_data['pan_offset'].y(),
            img_w, img_h
        )

        if not display_rect.contains(pos):
            return QPoint(-1, -1)

        rel_x = (pos.x() - display_rect.x()) / display_rect.width()
        rel_y = (pos.y() - display_rect.y()) / display_rect.height()

        rel_x = max(0.0, min(1.0, rel_x))
        rel_y = max(0.0, min(1.0, rel_y))

        img_x = int(rel_x * (view_data['original_size'].x() - 1))

        effective_flip_y = self.flip_y_coordinates
        try:
            flip_y_count = sum(1 for t in getattr(self, 'transform_history', [])
                               if isinstance(t, dict) and t.get('type') == 'flip_y')
            if (flip_y_count % 2) == 1:
                effective_flip_y = not effective_flip_y
        except Exception:
            pass

        if effective_flip_y:
            img_y = int((1.0 - rel_y) * (view_data['original_size'].y() - 1))
        else:
            img_y = int(rel_y * (view_data['original_size'].y() - 1))

        return QPoint(img_x, img_y)

    def toggle_flip_y(self, state):
        """Toggle flip Y coordinates to match ITK-SNAP"""
        self.flip_y_coordinates = state == Qt.Checked
        self.status_bar.showMessage(
            f"Flip Y coordinates: {'ON' if self.flip_y_coordinates else 'OFF'}")
        self.ct_view.update()
        self.suv_view.update()

    def update_refine_threshold(self, value):
        """Update global refine threshold"""
        self.refine_threshold_percent = value
        if self.suv_array is not None:
            suv_max = np.max(self.suv_array)
            # Thông báo cả ngưỡng hiệu dụng (bị chặn ở 15% để giữ viền)
            effective_percent = min(value, 15.0)
            threshold_abs = suv_max * effective_percent / 100.0
            self.status_bar.showMessage(
                f"Ngưỡng Refine: {value}% (dùng {effective_percent}% hiệu dụng, ≥ {threshold_abs:.2f})")

    def export_3d_mask(self):
        """Export 3D mask to file - ĐÃ SỬA LỖI ORIENTATION"""
        if self.mask_array is None or np.sum(self.mask_array) == 0:
            QMessageBox.warning(
                self, "Cảnh báo", "Không có mask nào để export")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self, "Lưu mask 3D", "", "NIfTI files (*.nii *.nii.gz)")

        if file_path:
            try:

                export_mask_array = self.mask_array.copy()

                if hasattr(self, 'transform_history'):
                    for transform in reversed(self.transform_history):
                        if transform['type'] == 'flip_y':
                            export_mask_array = np.flip(
                                export_mask_array, axis=1)
                        elif transform['type'] == 'flip_x':
                            export_mask_array = np.flip(
                                export_mask_array, axis=2)
                        elif transform['type'] == 'rotate_90':
                            # Đảo ngược rotation
                            k = 4 - transform['k']  # 1->3, 3->1
                            export_mask_array = np.rot90(
                                export_mask_array, k=k, axes=(1, 2))

                # Tạo mask image từ array
                mask_image = sitk.GetImageFromArray(export_mask_array)

                #  SỬ DỤNG METADATA TỪ ẢNH GỐC CHƯA CHỈNH SỬA
                if hasattr(self, 'ct_original'):
                    source_image = self.ct_original
                elif self.ct_image is not None:
                    source_image = self.ct_image
                else:
                    source_image = None

                if source_image is not None:
                    mask_image.CopyInformation(source_image)

                    # Đảm bảo direction matrix chính xác
                    mask_image.SetDirection(source_image.GetDirection())
                    mask_image.SetOrigin(source_image.GetOrigin())
                    mask_image.SetSpacing(source_image.GetSpacing())

                sitk.WriteImage(mask_image, file_path)

                # Hiển thị dialog chọn orientation
                self.show_orientation_choice_dialog(file_path)

            except Exception as e:
                QMessageBox.critical(
                    self, "Lỗi", f"Không thể lưu mask 3D: {str(e)}")

    def show_orientation_choice_dialog(self, file_path):
        """Hiển thị dialog chọn orientation phù hợp"""
        dialog = QDialog(self)
        dialog.setWindowTitle("Chọn Orientation cho Export")
        dialog.setGeometry(200, 200, 500, 300)

        layout = QVBoxLayout(dialog)

        label = QLabel(
            "CHỌN ORIENTATION PHÙ HỢP:\n\n"
            "• ITK-SNAP: Sử dụng file vừa export\n"
            "• 3D Slicer & phần mềm khác: Chọn 'Standard Orientation'\n"
            "• Nếu vẫn lỗi: Thử các tùy chọn khác"
        )
        layout.addWidget(label)

        # Các tùy chọn export
        btn_original = QPushButton(" File đã export (cho ITK-SNAP)")
        btn_standard = QPushButton("Export Standard Orientation")
        btn_ras = QPushButton(" Export RAS Orientation")
        btn_lps = QPushButton(" Export LPS Orientation")

        btn_original.clicked.connect(
            lambda: self.finish_export(dialog, file_path, "original"))
        btn_standard.clicked.connect(
            lambda: self.export_standard_orientation(file_path, dialog))
        btn_ras.clicked.connect(
            lambda: self.export_ras_orientation(file_path, dialog))
        btn_lps.clicked.connect(
            lambda: self.export_lps_orientation(file_path, dialog))

        layout.addWidget(btn_original)
        layout.addWidget(btn_standard)
        layout.addWidget(btn_ras)
        layout.addWidget(btn_lps)

        dialog.exec_()

    def export_standard_orientation(self, original_path, dialog):
        """Export với standard orientation"""
        try:
            base_path = original_path.replace(
                '.nii', '_standard.nii').replace('.nii.gz', '_standard.nii.gz')
            mask_image = sitk.ReadImage(original_path)

            #  CHUYỂN ĐỔI SANG STANDARD ORIENTATION
            mask_image = sitk.DICOMOrient(
                mask_image, "LPS")  # Standard trong y tế

            sitk.WriteImage(mask_image, base_path)
            self.finish_export(dialog, base_path, "standard")

        except Exception as e:
            QMessageBox.critical(
                self, "Lỗi", f"Không thể export standard orientation: {str(e)}")

    def export_ras_orientation(self, original_path, dialog):
        """Export với RAS orientation (ITK-SNAP)"""
        try:
            base_path = original_path.replace(
                '.nii', '_ras.nii').replace('.nii.gz', '_ras.nii.gz')
            mask_image = sitk.ReadImage(original_path)

            # RAS orientation cho ITK-SNAP
            mask_image = sitk.DICOMOrient(mask_image, "RAS")

            sitk.WriteImage(mask_image, base_path)
            self.finish_export(dialog, base_path, "ras")

        except Exception as e:
            QMessageBox.critical(
                self, "Lỗi", f"Không thể export RAS orientation: {str(e)}")

    def export_lps_orientation(self, original_path, dialog):
        """Export với LPS orientation (3D Slicer)"""
        try:
            base_path = original_path.replace(
                '.nii', '_lps.nii').replace('.nii.gz', '_lps.nii.gz')
            mask_image = sitk.ReadImage(original_path)

            
            mask_image = sitk.DICOMOrient(mask_image, "LPS")

            sitk.WriteImage(mask_image, base_path)
            self.finish_export(dialog, base_path, "lps")

        except Exception as e:
            QMessageBox.critical(
                self, "Lỗi", f"Không thể export LPS orientation: {str(e)}")

    def finish_export(self, dialog, file_path, orientation_type):
        """Hoàn tất export và hiển thị thông tin"""
        dialog.accept()

        unique_labels = np.unique(self.mask_array)
        unique_labels = unique_labels[unique_labels > 0]
        num_labels = len(unique_labels)

        orientation_names = {
            "original": "Original",
            "standard": "Standard (LPS)",
            "ras": "RAS (ITK-SNAP)",
            "lps": "LPS (3D Slicer)"
        }

        message = (f"ĐÃ LƯU MASK 3D THÀNH CÔNG!\n"
                   f"File: {file_path}\n"
                   f"Số tổn thương: {num_labels}\n"
                   f"Orientation: {orientation_names[orientation_type]}\n\n"
                   f" MẸO: Nếu orientation chưa đúng, thử export với orientation khác")

        QMessageBox.information(self, "Xuất mask thành công", message)
        self.status_bar.showMessage(
            f"Đã xuất mask 3D với {num_labels} tổn thương - {orientation_names[orientation_type]}")

    def check_image_orientation(self, image):
        """Kiểm tra orientation của ảnh"""
        if image is None:
            return "Unknown"

        try:
            direction = image.GetDirection()
            origin = image.GetOrigin()
            spacing = image.GetSpacing()

            orientation_info = f"Direction: {direction}\nOrigin: {origin}\nSpacing: {spacing}"
            return orientation_info
        except:
            return "Cannot get orientation info"

    def show_orientation_info(self):
        """Hiển thị thông tin orientation"""
        if self.ct_image is None:
            return

        ct_info = self.check_image_orientation(
            self.ct_original if hasattr(self, 'ct_original') else self.ct_image)
        current_info = self.check_image_orientation(self.ct_image)

        info_text = f"ORIENTATION INFORMATION:\n\n"
        info_text += f"ORIGINAL CT:\n{ct_info}\n\n"
        info_text += f"CURRENT CT:\n{current_info}\n\n"
        info_text += f"FLIPPED: {getattr(self, 'original_flipped', False)}"
        info_text += f"\nTRANSFORM HISTORY: {len(self.transform_history)} transforms"

        QMessageBox.information(self, "Orientation Info", info_text)

    # -------------------------------
    # SLICE NAVIGATION METHODS
    # -------------------------------
    def process_slice_change(self):
        """Process pending slice changes from mouse wheel"""
        if self.pending_slice_delta == 0:
            self.slice_timer.stop()
            return
        if self.ct_array is not None and len(self.ct_array.shape) == 3:
            delta = 1 if self.pending_slice_delta > 0 else -1
            self.pending_slice_delta = 0
            new_idx = max(
                0, min(self.ct_array.shape[0] - 1, self.current_slice_idx + delta))
            if new_idx != self.current_slice_idx:
                self.current_slice_idx = new_idx
                self.slice_control.setValue(self.current_slice_idx)

    def change_slice_delta(self, delta):
        """Change slice with delta from mouse wheel"""
        self.pending_slice_delta += (1 if delta > 0 else -1)
        if not self.slice_timer.isActive():
            self.slice_timer.start()

    def change_slice(self, slice_idx):
        """Change slice programmatically"""
        if self.ct_array is not None and len(self.ct_array.shape) == 3:
            self.current_slice_idx = slice_idx
            self.show_current_slice()

    def show_current_slice(self):
        """Show current slice in both views"""
        self.clear_click_info()

        if self.ct_array is None:
            return

        # Determine which mask to display
        if self.editing_mode and self.edit_mask_array is not None:
            display_mask = self.edit_mask_array
        else:
            display_mask = self.mask_array

        if len(self.ct_array.shape) == 3:
            ct_slice = self.ct_array[self.current_slice_idx]
            suv_slice = self.suv_array[self.current_slice_idx]
            self.ct_view_data['original_size'] = QPoint(
                ct_slice.shape[1], ct_slice.shape[0])
            self.suv_view_data['original_size'] = QPoint(
                suv_slice.shape[1], suv_slice.shape[0])
            self.ct_view_data['image_array'] = ct_slice
            self.suv_view_data['image_array'] = suv_slice
            current_mask = display_mask[self.current_slice_idx].copy(
            ) if display_mask is not None else np.zeros_like(ct_slice, dtype=np.uint8)
            self.ct_view_data['mask'] = current_mask
            self.suv_view_data['mask'] = current_mask.copy()

            status_msg = f"Slice {self.current_slice_idx}/{self.ct_array.shape[0]-1}"
            if self.editing_mode:
                status_msg += " [CHẾ ĐỘ CHỈNH SỬA]"
            self.status_bar.showMessage(status_msg)
        else:
            self.ct_view_data['original_size'] = QPoint(
                self.ct_array.shape[1], self.ct_array.shape[0])
            self.suv_view_data['original_size'] = QPoint(
                self.suv_array.shape[1], self.suv_array.shape[0])
            self.ct_view_data['image_array'] = self.ct_array
            self.suv_view_data['image_array'] = self.suv_array
            self.ct_view_data['mask'] = display_mask.copy(
            ) if display_mask is not None else np.zeros_like(self.ct_array, dtype=np.uint8)
            self.suv_view_data['mask'] = self.ct_view_data['mask'].copy()

        self.ct_view.update()
        self.suv_view.update()

    # -------------------------------
    # MOUSE AND INTERACTION
    # -------------------------------
    def view_mouse_press_event(self, event, view_type):
        view_data = self.ct_view_data if view_type == "CT" else self.suv_view_data

        if event.button() == Qt.LeftButton and view_data['image_array'] is not None:
            img_point = self.map_to_image_coords(event.pos(), view_type)

            if (0 <= img_point.x() < view_data['image_array'].shape[1] and
                    0 <= img_point.y() < view_data['image_array'].shape[0]):

                # Khi tắt brush, dùng click để xem thông số label
                if not self.brush_enabled:
                    if view_data['mask'] is not None:
                        label_value = int(
                            view_data['mask'][img_point.y(), img_point.x()])
                        if label_value > 0:
                            self.show_label_stats(
                                label_value, img_point, view_type)
                            view_data['widget'].update()
                    return

                # Editing mode with selected lesion
                if self.editing_mode and self.selected_lesion_label is not None:
                    self.drawing = True
                    self.last_point = img_point
                    self.manual_editing_done = True
                    self.base_refine_mask = None
                    self.edit_changes_made = True

                    if self.edit_mask_array is not None:
                        target_label = 0 if self.eraser_mode else self.selected_lesion_label
                        if not self.eraser_mode:
                            self.edited_lesions.add(self.selected_lesion_label)
                        self.apply_brush(self.edit_mask_array,
                                         img_point, target_label)
                        view_data['mask'] = self.edit_mask_array[self.current_slice_idx].copy(
                        )
                        view_data['widget'].update()
                    return

                # Start drawing
                self.drawing = True
                self.last_point = img_point
                current_mask_array = self.edit_mask_array if self.editing_mode else self.mask_array

                if current_mask_array is not None:
                    target_label = 0 if self.eraser_mode else self.current_label
                    self.apply_brush(current_mask_array,
                                     img_point, target_label)
                    self.base_refine_mask = None
                    view_data['mask'] = current_mask_array[self.current_slice_idx].copy()
                    if self.editing_mode:
                        self.edit_changes_made = True
                    view_data['widget'].update()

        elif event.button() == Qt.RightButton:
            self.pan_start = event.pos()
            self.clear_click_info()
            view_data['widget'].update()

    def apply_brush(self, mask_array, img_point, label):
        """Apply brush to mask array at given position"""
        y, x = img_point.y(), img_point.x()
        brush_radius = self.brush_size // 2

        y_indices, x_indices = np.ogrid[-brush_radius:brush_radius +
                                        1, -brush_radius:brush_radius+1]
        brush_mask = x_indices*x_indices + y_indices * \
            y_indices <= brush_radius*brush_radius

        if mask_array.ndim == 3:
            y_start = max(0, y - brush_radius)
            y_end = min(mask_array.shape[1], y + brush_radius + 1)
            x_start = max(0, x - brush_radius)
            x_end = min(mask_array.shape[2], x + brush_radius + 1)
        else:
            y_start = max(0, y - brush_radius)
            y_end = min(mask_array.shape[0], y + brush_radius + 1)
            x_start = max(0, x - brush_radius)
            x_end = min(mask_array.shape[1], x + brush_radius + 1)

        mask_y_start = max(0, brush_radius - y)
        mask_y_end = mask_y_start + (y_end - y_start)
        mask_x_start = max(0, brush_radius - x)
        mask_x_end = mask_x_start + (x_end - x_start)

        slice_mask = brush_mask[mask_y_start:mask_y_end,
                                mask_x_start:mask_x_end]
        if mask_array.ndim == 3:
            mask_array[self.current_slice_idx, y_start:y_end,
                       x_start:x_end][slice_mask] = label
        else:
            mask_array[y_start:y_end, x_start:x_end][slice_mask] = label
        self.manual_editing_done = True
        self.base_refine_mask = None

    def view_mouse_move_event(self, event, view_type):
        view_data = self.ct_view_data if view_type == "CT" else self.suv_view_data
        img_point = self.map_to_image_coords(event.pos(), view_type)
        img_x, img_y = img_point.x(), img_point.y()

        # Drawing
        if self.drawing and view_data['image_array'] is not None and view_data['mask'] is not None:
            current_point = self.map_to_image_coords(event.pos(), view_type)
            if self.is_point_valid(current_point, view_data):
                self.draw_line(self.last_point, current_point, view_type)
                self.last_point = current_point
            # Khi đang vẽ, bỏ qua hiển thị thông số để giảm lag
            return

        # Panning
        elif event.buttons() & Qt.RightButton:
            delta = event.pos() - self.pan_start
            view_data['pan_offset'] += delta
            self.pan_start = event.pos()
            if self.sync_views:
                other_view = self.suv_view_data if view_type == "CT" else self.ct_view_data
                other_view['pan_offset'] = view_data['pan_offset']
                other_view['widget'].update()
            view_data['widget'].update()

        # Update status and hover stats (run after drawing so dùng mask mới nhất)
        status_text = f"{view_type}: X={img_x}, Y={img_y}"
        show_stats = False
        if view_data['mask'] is not None and img_x >= 0 and img_y >= 0 and not self.brush_enabled:
            if (0 <= img_x < view_data['mask'].shape[1] and
                    0 <= img_y < view_data['mask'].shape[0]):
                hover_label = int(view_data['mask'][img_y, img_x])
                if hover_label > 0:
                    self.show_label_stats(hover_label, img_point, view_type)
                    show_stats = True
                else:
                    self.clear_click_info()
            else:
                self.clear_click_info()

        if self.ct_array is not None and self.suv_array is not None:
            if (0 <= img_x < self.ct_array.shape[-1] and 0 <= img_y < self.ct_array.shape[-2]):
                if len(self.ct_array.shape) == 3:
                    ct_val = self.ct_array[self.current_slice_idx,
                                           img_y, img_x]
                    suv_val = self.suv_array[self.current_slice_idx,
                                             img_y, img_x]
                else:
                    ct_val = self.ct_array[img_y, img_x]
                    suv_val = self.suv_array[img_y, img_x]
                status_text += f" | CT: {ct_val:.1f}, SUV: {suv_val:.2f}"

        if self.editing_mode:
            status_text += " | [CHẾ ĐỘ CHỈNH SỬA]"

        if not show_stats:
            self.status_bar.showMessage(status_text)

    def is_point_valid(self, point, view_data):
        """Check if point is within image bounds"""
        return (0 <= point.x() < view_data['mask'].shape[1] and
                0 <= point.y() < view_data['mask'].shape[0] and
                0 <= self.last_point.x() < view_data['mask'].shape[1] and
                0 <= self.last_point.y() < view_data['mask'].shape[0])

    def draw_line(self, start_point, end_point, view_type):
        """Draw line between two points"""
        if not self.brush_enabled:
            return
        view_data = self.ct_view_data if view_type == "CT" else self.suv_view_data

        pil_mask = Image.new(
            'L', (view_data['mask'].shape[1], view_data['mask'].shape[0]), 0)
        draw = ImageDraw.Draw(pil_mask)
        draw.line([(start_point.x(), start_point.y()), (end_point.x(), end_point.y())],
                  fill=255, width=self.brush_size)
        draw.ellipse([(start_point.x() - self.brush_size//2, start_point.y() - self.brush_size//2),
                      (start_point.x() + self.brush_size//2, start_point.y() + self.brush_size//2)],
                     fill=255)
        draw.ellipse([(end_point.x() - self.brush_size//2, end_point.y() - self.brush_size//2),
                      (end_point.x() + self.brush_size//2, end_point.y() + self.brush_size//2)],
                     fill=255)
        temp_mask = np.array(pil_mask) > 0

        current_mask_array = self.edit_mask_array if self.editing_mode else self.mask_array

        if current_mask_array is not None:
            if self.eraser_mode:
                target_label = 0
            else:
                target_label = (self.selected_lesion_label if (self.editing_mode and
                                self.selected_lesion_label is not None) else self.current_label)

            if current_mask_array.ndim == 3:
                current_mask_array[self.current_slice_idx][temp_mask] = target_label
                view_data['mask'] = current_mask_array[self.current_slice_idx].copy()
            else:
                current_mask_array[temp_mask] = target_label
                view_data['mask'] = current_mask_array.copy()

            if self.editing_mode:
                if not self.eraser_mode and self.selected_lesion_label is not None:
                    self.edited_lesions.add(self.selected_lesion_label)
                self.manual_editing_done = True
                self.base_refine_mask = None
                self.edit_changes_made = True
            else:
                self.manual_editing_done = True
                self.base_refine_mask = None

            view_data['widget'].update()

    def view_mouse_release_event(self, event, view_type):
        if event.button() == Qt.LeftButton:
            self.drawing = False

    def view_wheel_event(self, event, view_type):
        view_data = self.ct_view_data if view_type == "CT" else self.suv_view_data
        if QApplication.keyboardModifiers() & Qt.ControlModifier:
            # Zoom
            zoom_in = event.angleDelta().y() > 0
            old_zoom = view_data['zoom_factor']
            view_data['zoom_factor'] *= 1.1 if zoom_in else 1/1.1
            view_data['zoom_factor'] = max(
                0.1, min(10.0, view_data['zoom_factor']))

            pos = event.pos()
            rel_x = (pos.x() - view_data['widget'].width()/2 - view_data['pan_offset'].x()) / (
                view_data['original_size'].x() * old_zoom)
            rel_y = (pos.y() - view_data['widget'].height()/2 - view_data['pan_offset'].y()) / (
                view_data['original_size'].y() * old_zoom)

            new_width = view_data['original_size'].x() * \
                view_data['zoom_factor']
            new_height = view_data['original_size'].y() * \
                view_data['zoom_factor']

            view_data['pan_offset'].setX(
                int(pos.x() - view_data['widget'].width()/2 - rel_x * new_width))
            view_data['pan_offset'].setY(
                int(pos.y() - view_data['widget'].height()/2 - rel_y * new_height))

            if self.sync_views:
                other_view_data = self.ct_view_data if view_type == "SUV" else self.suv_view_data
                other_view_data['zoom_factor'] = view_data['zoom_factor']
                other_view_data['pan_offset'] = view_data['pan_offset']
                other_view_data['widget'].update()
            view_data['widget'].update()
        else:
            # Change slice
            self.change_slice_delta(event.angleDelta().y())

    def keyPressEvent(self, event):
        """Toggle mask visibility with 'S' key"""
        if event.key() == Qt.Key_S:
            current = self.mask_visibility.isChecked()
            self.mask_visibility.setChecked(not current)
            self.toggle_masks_visibility()
            state_text = "Hiện mask" if not current else "Ẩn mask"
            self.status_bar.showMessage(f"{state_text} (phím S)")
            return
        super().keyPressEvent(event)

    # -------------------------------
    # PAINTING AND DISPLAY
    # -------------------------------
    def view_paint_event(self, event, view_type):
        view_data = self.ct_view_data if view_type == "CT" else self.suv_view_data
        painter = QPainter(view_data['widget'])
        painter.fillRect(view_data['widget'].rect(), Qt.black)

        if view_data['image_array'] is None:
            painter.drawText(view_data['widget'].rect(
            ), Qt.AlignCenter, f"Chưa tải ảnh {view_type}")
            return

        img_w = int(view_data['original_size'].x() * view_data['zoom_factor'])
        img_h = int(view_data['original_size'].y() * view_data['zoom_factor'])

        display_array = self.get_display_array(view_type)
        if display_array is None:
            return

        # Convert to QImage
        if view_type == "SUV":
            height, width, _ = display_array.shape
            bytes_per_line = 3 * width
            q_img = QImage(display_array.data, width, height,
                           bytes_per_line, QImage.Format_RGB888)
        else:
            height, width = display_array.shape
            bytes_per_line = width
            q_img = QImage(display_array.data, width, height,
                           bytes_per_line, QImage.Format_Grayscale8)

        display_rect = QRect(
            view_data['widget'].width() // 2 - img_w // 2 +
            view_data['pan_offset'].x(),
            view_data['widget'].height() // 2 - img_h // 2 +
            view_data['pan_offset'].y(),
            img_w, img_h
        )

        # Draw image (resized if needed)
        if view_data['zoom_factor'] != 1.0:
            interpolation = cv2.INTER_LINEAR if view_data['zoom_factor'] > 1.0 else cv2.INTER_AREA
            if view_type == "SUV":
                resized = cv2.resize(
                    display_array, (img_w, img_h), interpolation=interpolation)
                q_resized = QImage(resized.data, img_w, img_h,
                                   img_w * 3, QImage.Format_RGB888)
            else:
                resized = cv2.resize(
                    display_array, (img_w, img_h), interpolation=interpolation)
                q_resized = QImage(resized.data, img_w, img_h,
                                   img_w, QImage.Format_Grayscale8)
            painter.drawImage(display_rect, q_resized)
        else:
            painter.drawImage(display_rect, q_img)

        # Draw mask overlay
        if view_data['show_mask'] and view_data['mask'] is not None and np.any(view_data['mask']):
            mask_img = np.zeros(
                (view_data['mask'].shape[0], view_data['mask'].shape[1], 4), dtype=np.uint8)
            for label, color in self.label_colors.items():
                mask_img[view_data['mask'] == label] = [
                    color.red(), color.green(), color.blue(), color.alpha()]

            # Highlight selected lesion in editing mode
            if self.editing_mode and self.selected_lesion_label is not None:
                selected_mask = (view_data['mask'] ==
                                 self.selected_lesion_label)
                border_mask = np.zeros_like(mask_img[:, :, 0], dtype=bool)
                if np.any(selected_mask):
                    dilated = ndimage.binary_dilation(
                        selected_mask, structure=np.ones((3, 3)))
                    border_mask = dilated & ~selected_mask
                    mask_img[border_mask] = [255, 255, 255, 200]

            if view_data['zoom_factor'] != 1.0:
                resized_mask = cv2.resize(
                    mask_img, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
                q_mask = QImage(resized_mask.data, img_w, img_h,
                                img_w * 4, QImage.Format_RGBA8888)
            else:
                q_mask = QImage(mask_img.data, view_data['mask'].shape[1], view_data['mask'].shape[0],
                                view_data['mask'].shape[1] * 4, QImage.Format_RGBA8888)
            painter.drawImage(display_rect, q_mask)

        # Show label statistics if clicked
        if (self.last_clicked_label is not None and self.last_clicked_position is not None and
                self.last_clicked_view == view_type and view_data['mask'] is not None):
            self.draw_label_stats(painter, view_data, display_rect)

    def get_display_array(self, view_type):
        """Convert image array to display format"""
        view_data = self.ct_view_data if view_type == "CT" else self.suv_view_data
        if view_data['image_array'] is None:
            return None
        w_min = view_data['window_level'] - view_data['window_width']/2
        w_max = view_data['window_level'] + view_data['window_width']/2
        if w_max > w_min:
            windowed = np.clip(
                (view_data['image_array'] - w_min) / (w_max - w_min), 0, 1)
        else:
            windowed = np.zeros_like(
                view_data['image_array'], dtype=np.float32)

        if view_type == "SUV":
            normalized = (windowed * 255).astype(np.uint8)
            jet = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
            display = cv2.cvtColor(jet, cv2.COLOR_BGR2RGB)
            return display
        else:
            return (windowed * 255).astype(np.uint8)

    def draw_label_stats(self, painter, view_data, display_rect):
        """Draw label statistics on the view"""
        rel_x = self.last_clicked_position.x() / view_data['original_size'].x()
        rel_y = self.last_clicked_position.y() / view_data['original_size'].y()
        display_x = display_rect.x() + int(rel_x * display_rect.width())
        display_y = display_rect.y() + int(rel_y * display_rect.height())

        label_color = self.label_colors.get(
            self.last_clicked_label, QColor(255, 255, 255))

        stats = self.get_label_statistics(self.last_clicked_label)
        if stats:
            suv_max, suv_mean, suv_peak, volume_ml, tlg = stats
            text = (f"Label {self.last_clicked_label}  \n SUVmax={suv_max:.2f}  "
                    f"\n SUVmean={suv_mean:.2f}  \n SUVpeak={suv_peak:.2f}  "
                    f"\n Thể tích={volume_ml:.2f} ml  \n TLG={tlg:.2f}")
        else:
            text = f"Label {self.last_clicked_label}"

        font = QFont()
        font.setPointSize(12)
        font.setBold(True)
        painter.setFont(font)
        text_rect = painter.fontMetrics().boundingRect(
            QRect(0, 0, 300, 2000), Qt.TextWordWrap, text)
        text_rect.moveTo(display_x - text_rect.width()//2,
                         display_y - text_rect.height() - 40)
        text_rect.adjust(-6, -4, 6, 4)

        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(0, 0, 0, 200))
        painter.drawRect(text_rect)

        painter.setPen(QPen(QColor(255, 255, 255), 2))
        painter.drawText(text_rect, Qt.AlignCenter | Qt.TextWordWrap, text)

    # -------------------------------
    # CORE FUNCTIONALITY - OPTIMIZED
    # -------------------------------
    def load_images(self):
        """Load CT and SUV images với tracking flip state"""
        self.clear_click_info()

        ct_path, _ = QFileDialog.getOpenFileName(
            self, "Chọn ảnh CT", "", "Ảnh y tế (*.nii *.nii.gz *.dcm *.dicom *.mha *.mhd)")
        if not ct_path:
            return
        suv_path, _ = QFileDialog.getOpenFileName(
            self, "Chọn ảnh SUV", "", "Ảnh y tế (*.nii *.nii.gz *.dcm *.dicom *.mha *.mhd)")
        if not suv_path:
            return

        # Ask about segmentation
        reply = QMessageBox.question(self, "Load Segmentation",
                                     "Bạn có muốn load ảnh segmentation có sẵn không?",
                                     QMessageBox.Yes | QMessageBox.No)

        seg_path = None
        if reply == QMessageBox.Yes:
            seg_path, _ = QFileDialog.getOpenFileName(
                self, "Chọn ảnh Segmentation", "", "Ảnh y tế (*.nii *.nii.gz *.dcm *.dicom *.mha *.mhd)")

        try:
            # Load images
            self.ct_image = sitk.ReadImage(ct_path)
            self.ct_array = sitk.GetArrayFromImage(
                self.ct_image).astype(np.float32)
            self.suv_image = sitk.ReadImage(suv_path)
            self.suv_array = sitk.GetArrayFromImage(
                self.suv_image).astype(np.float32)
            self.base_refine_mask = None  

           
            self.ct_original = sitk.ReadImage(ct_path)
            self.suv_original = sitk.ReadImage(suv_path)

            # Reset transform history
            self.transform_history = []

            # Ask about flip
            reply_flip = QMessageBox.question(
                self, "Flip ảnh",
                "Có flip ảnh để hiển thị đúng chiều không?\n\n"
                "Yes: Flip ảnh theo trục Y (sẽ tự động đảo ngược khi export)\n"
                "No: Giữ nguyên orientation",
                QMessageBox.Yes | QMessageBox.No
            )

            should_flip = (reply_flip == QMessageBox.Yes)
            self.original_flipped = should_flip  # 🔥 LƯU TRẠNG THÁI FLIP

            if should_flip:
                if len(self.ct_array.shape) == 3:
                    self.ct_array = np.flip(self.ct_array, axis=1)
                    self.suv_array = np.flip(self.suv_array, axis=1)
                else:
                    self.ct_array = np.flip(self.ct_array, axis=0)
                    self.suv_array = np.flip(self.suv_array, axis=0)

                # Ghi nhận transform
                self.transform_history.append({'type': 'flip_y'})

            # Load segmentation if provided
            self.loaded_segmentation = False
            if seg_path:
                self.load_segmentation_file(seg_path, should_flip)
            else:
                self.mask_array = np.zeros_like(self.ct_array, dtype=np.uint8)

            # Setup UI
            self.setup_after_load()

        except Exception as e:
            QMessageBox.critical(self, "Lỗi", f"Không thể tải ảnh: {str(e)}")

    def load_segmentation_file(self, seg_path, should_flip):
        """Load segmentation file with error handling"""
        try:
            self.seg_image = sitk.ReadImage(seg_path)
            self.seg_array = sitk.GetArrayFromImage(
                self.seg_image).astype(np.uint8)
            self.base_refine_mask = None  # reset base khi load segmentation

            if should_flip:
                if len(self.seg_array.shape) == 3:
                    self.seg_array = np.flip(self.seg_array, axis=1)
                else:
                    self.seg_array = np.flip(self.seg_array, axis=0)

            if self.seg_array.shape == self.ct_array.shape:
                self.mask_array = self.seg_array.copy()
                self.loaded_segmentation = True
                unique_labels = np.unique(self.mask_array)
                unique_labels = unique_labels[unique_labels > 0]
                self.status_bar.showMessage(
                    f"Đã load segmentation với {len(unique_labels)} tổn thương")
            else:
                QMessageBox.warning(self, "Cảnh báo",
                                    f"Kích thước segmentation không khớp. Sẽ tạo mask mới.")
                self.mask_array = np.zeros_like(self.ct_array, dtype=np.uint8)
        except Exception as seg_error:
            QMessageBox.warning(self, "Cảnh báo",
                                f"Không thể load segmentation: {str(seg_error)}. Sẽ tạo mask mới.")
            self.mask_array = np.zeros_like(self.ct_array, dtype=np.uint8)

    def setup_after_load(self):
        """Setup UI after loading images"""
        if len(self.ct_array.shape) == 3:
            self.slice_control.setRange(0, self.ct_array.shape[0] - 1)
            self.current_slice_idx = self.ct_array.shape[0] // 2
            self.slice_control.setValue(self.current_slice_idx)

            status_msg = f"Đã tải ảnh 3D với {self.ct_array.shape[0]} slices"
            if self.loaded_segmentation:
                unique_labels = np.unique(self.mask_array)
                unique_labels = unique_labels[unique_labels > 0]
                status_msg += f" + segmentation ({len(unique_labels)} tổn thương)"
            self.status_bar.showMessage(status_msg)
        else:
            status_msg = "Đã tải ảnh 2D"
            if self.loaded_segmentation:
                status_msg += " + segmentation"
            self.status_bar.showMessage(status_msg)

        # Set original size
        if len(self.ct_array.shape) == 3:
            ct_slice = self.ct_array[self.current_slice_idx]
            suv_slice = self.suv_array[self.current_slice_idx]
        else:
            ct_slice = self.ct_array
            suv_slice = self.suv_array

        self.ct_view_data['original_size'] = QPoint(
            ct_slice.shape[1], ct_slice.shape[0])
        self.suv_view_data['original_size'] = QPoint(
            suv_slice.shape[1], suv_slice.shape[0])

        self.show_current_slice()

    # -------------------------------
    # REFINE ALGORITHM - OPTIMIZED
    # -------------------------------
    def refine_all_masks(self):
        """Refine auto theo bài báo: chỉ trong vùng đã bôi, ngưỡng tuyệt đối SUV 2.5 + ngưỡng 30% SUVmax từng lesion"""
        if self.suv_array is None:
            QMessageBox.warning(self, "Lỗi", "Chưa tải ảnh SUV")
            return

        # Chỉ refine trong vùng người dùng đã bôi
        mask_source = self.edit_mask_array if self.editing_mode and self.edit_mask_array is not None else self.mask_array
        if mask_source is None:
            QMessageBox.warning(self, "Lỗi", "Chưa có mask để refine (cần bôi trước)")
            return
        if not np.any(mask_source):
            QMessageBox.warning(self, "Cảnh báo", "Chưa có mask nào được bôi để refine")
            return

        try:
            spacing = self.ct_image.GetSpacing()
            voxel_volume_mm3 = spacing[0] * spacing[1] * spacing[2]

            if self.base_refine_mask is None:
                self.base_refine_mask = (mask_source > 0)
            if self.manual_editing_done:
                self.manual_editing_done = False

            user_drawn_mask = self.base_refine_mask.copy()
            fixed_threshold = 2.5
            initial_mask = np.logical_and(user_drawn_mask, self.suv_array >= fixed_threshold)
            if not np.any(initial_mask):
                QMessageBox.warning(self, "Cảnh báo", "Không có voxel nào đạt ngưỡng SUV trong vùng đã bôi")
                return

            if initial_mask.ndim == 3:
                labeled = cc3d.connected_components(initial_mask.astype(np.uint8), connectivity=6)
                num_components = int(np.max(labeled))
            else:
                structure = np.array([[0, 1, 0],
                                      [1, 1, 1],
                                      [0, 1, 0]], dtype=int)
                labeled, num_components = ndimage.label(initial_mask, structure=structure)

            if num_components == 0:
                QMessageBox.warning(self, "Cảnh báo", "Không tách được tổn thương nào")
                return

            final_mask = np.zeros_like(self.suv_array, dtype=np.uint8)
            lesion_stats = {}
            per_lesion_percent = 41.0 / 100.0  
            lesion_threshold_cap = 7.0  # trần ngưỡng tuyệt đối để giữ voxel SUV thấp
            next_label = 1

            for comp_label in range(1, num_components + 1):
                comp_mask = (labeled == comp_label)
                voxels_before = int(np.sum(comp_mask))
                if voxels_before == 0:
                    continue

                comp_suv_max = float(np.max(self.suv_array[comp_mask]))
               
                percent_used = per_lesion_percent
                raw_threshold = comp_suv_max * percent_used
                lesion_threshold = min(raw_threshold, lesion_threshold_cap)
                refined_mask = np.logical_and(comp_mask, self.suv_array >= lesion_threshold)

                if not np.any(refined_mask):
                    refined_mask = comp_mask 

                # Đảm bảo có màu cho label mới (không giới hạn số tổn thương)
                self.ensure_label_colors(next_label)
                final_mask[refined_mask] = next_label
                voxels_after = int(np.sum(refined_mask))
                volume_before_ml = voxels_before * voxel_volume_mm3 / 1000.0
                volume_after_ml = voxels_after * voxel_volume_mm3 / 1000.0

                lesion_stats[next_label] = {
                    'voxels_before': voxels_before,
                    'voxels_after': voxels_after,
                    'volume_before_ml': volume_before_ml,
                    'volume_after_ml': volume_after_ml,
                    'suv_max': comp_suv_max,
                    'threshold_used': lesion_threshold,
                    'threshold_raw': raw_threshold,
                    'percent_used': percent_used * 100.0
                }
                next_label += 1

            if next_label == 1:
                QMessageBox.warning(self, "Cảnh báo", "Không giữ được tổn thương nào sau refine")
                return

            self.mask_array = final_mask.astype(np.uint8)
            if self.editing_mode:
                self.edit_mask_array = self.mask_array.copy()
                self.edit_changes_made = True

            self.refine_done = True
            self.show_current_slice()

            total_voxels_before = int(np.sum(initial_mask))
            total_volume_before_ml = total_voxels_before * voxel_volume_mm3 / 1000.0
            total_voxels_after = int(np.sum(self.mask_array > 0))
            total_volume_after_ml = total_voxels_after * voxel_volume_mm3 / 1000.0

            self.show_refine_results(lesion_stats, total_volume_before_ml, total_volume_after_ml,
                                     None, 41.0)

        except Exception as e:
            QMessageBox.critical(
                self, "Lỗi", f"Không thể refine mask 3D: {str(e)}")

    def post_refine_processing(self, first_refine=False):
        """Post-processing after refine"""
        if first_refine:
            # Split label 1 into connected components for first refine
            binary_mask = (self.mask_array == 1).astype(np.uint8)
            if np.sum(binary_mask) > 0:
                if self.mask_array.ndim == 3:
                    labeled_array = cc3d.connected_components(binary_mask)
                    self.mask_array = self.relabel_components(labeled_array)
                else:
                    labeled_array, ncomp = ndimage.label(binary_mask)
                    self.mask_array = self.relabel_2d_components(
                        labeled_array, ncomp)

        # Relabel all components to keep compact labels
        binary_mask_all = (self.mask_array > 0).astype(np.uint8)
        if np.sum(binary_mask_all) > 0:
            labeled_all = cc3d.connected_components(binary_mask_all)
            num_components = int(np.max(labeled_all))
            if num_components > 20:
                self.mask_array = self.keep_largest_components(
                    labeled_all, num_components)
            else:
                self.mask_array = labeled_all.astype(np.uint8)

        self.show_current_slice()

    def relabel_components(self, labeled_array):
        """Relabel connected components in 3D"""
        num_components = int(np.max(labeled_array))
        new_mask = np.zeros_like(labeled_array, dtype=np.uint8)
        next_label = 1
        for comp_label in range(1, num_components + 1):
            comp_voxels = np.sum(labeled_array == comp_label)
            if comp_voxels == 0:
                continue
            new_mask[labeled_array == comp_label] = next_label
            next_label += 1
        return new_mask

    def relabel_2d_components(self, labeled_array, ncomp):
        """Relabel connected components in 2D"""
        new_mask = np.zeros_like(labeled_array, dtype=np.uint8)
        for comp_label in range(1, ncomp + 1):
            new_mask[labeled_array == comp_label] = comp_label
        return new_mask

    def keep_largest_components(self, labeled_all, num_components):
        """Keep only the largest components up to max_labels"""
        component_sizes = []
        for comp_label in range(1, num_components + 1):
            size = np.sum(labeled_all == comp_label)
            component_sizes.append((comp_label, size))
        component_sizes.sort(key=lambda x: x[1], reverse=True)
        new_mask2 = np.zeros_like(labeled_all, dtype=np.uint8)
        for new_label, (old_label, _) in enumerate(component_sizes[:self.max_labels], 1):
            new_mask2[labeled_all == old_label] = new_label
        return new_mask2

    def remove_small_components(self, mask, min_voxels=5):
        """Remove components smaller than min_voxels from boolean mask"""
        working_mask = mask.copy()
        if not np.any(working_mask):
            return working_mask

        if working_mask.ndim == 3:
            labeled = cc3d.connected_components(working_mask.astype(np.uint8))
            num_components = int(np.max(labeled))
            for comp_label in range(1, num_components + 1):
                comp_mask = (labeled == comp_label)
                if np.sum(comp_mask) < min_voxels:
                    working_mask[comp_mask] = False
        else:
            labeled, ncomp = ndimage.label(working_mask)
            for comp_label in range(1, ncomp + 1):
                comp_mask = (labeled == comp_label)
                if np.sum(comp_mask) < min_voxels:
                    working_mask[comp_mask] = False

        return working_mask

    # -------------------------------
    # LABEL COLOR HELPERS
    # -------------------------------
    def ensure_label_colors(self, max_label):
        """Ensure we have colors defined up to max_label"""
        for lbl in range(1, max_label + 1):
            if lbl not in self.label_colors:
                self.label_colors[lbl] = self.generate_label_color(lbl)

    def generate_label_color(self, label):
        """Generate a distinct-ish color for a label (HSV spread)"""
        hue = (label * 137) % 360  # use a large prime for distribution
        sat = 200
        val = 255
        alpha = 200
        color = QColor()
        color.setHsv(hue, sat, val, alpha)
        return color

    def show_refine_results(self, lesion_stats, total_volume_before_ml, total_volume_after_ml,
                            initial_threshold_value, per_lesion_percent):
        """Show refine results in message box"""
        percentage_kept = (total_volume_after_ml / total_volume_before_ml *
                           100) if total_volume_before_ml > 0 else 0.0

        # Đếm số tổn thương an toàn theo mask hiện tại
        current_mask = self.get_active_mask() if hasattr(self, 'get_active_mask') else self.mask_array
        if current_mask is not None:
            lesion_count = int(max(0, len(np.unique(current_mask[current_mask > 0]))))
        else:
            lesion_count = len(lesion_stats)

        result_text = "KẾT QUẢ REFINE 3D (SUV ≥ 2.5 trong vùng bôi, 41% SUVmax (cap 7) từng lesion)\n" + \
            "="*55 + "\n"
        result_text += "THỐNG KÊ TỔNG QUAN:\n"
        result_text += f"• Thể tích ban đầu: {total_volume_before_ml:.2f} ml\n"
        result_text += f"• Thể tích sau refine: {total_volume_after_ml:.2f} ml\n"
        result_text += f"• Tỉ lệ giữ lại: {percentage_kept:.1f}%\n"
        result_text += f"• Số tổn thương: {lesion_count}\n"

        result_text += "\nTHAM SỐ:\n"
        if initial_threshold_value is None:
            result_text += "• Ngưỡng khởi tạo: dùng toàn bộ vùng đã bôi \n"
        else:
            result_text += f"• Ngưỡng khởi tạo: SUV ≥ {initial_threshold_value:.2f} (trong vùng đã bôi)\n"
        result_text += f"• Ngưỡng refine từng lesion: 41% SUVmax \n"

        QMessageBox.information(self, "Kết quả Refine 3D", result_text)
        self.status_bar.showMessage(
            f"Đã refine {len(lesion_stats)} labels (SUV ≥ 2.5, 41% SUVmax từng lesion)")

    # -------------------------------
    # CÁC PHƯƠNG THỨC CÒN THIẾU
    # -------------------------------
    def clear_click_info(self):
        """Clear click information"""
        self.last_clicked_label = None
        self.last_clicked_position = None
        self.last_clicked_value = None
        self.last_clicked_view = None

    def show_label_stats(self, label_value, img_point, view_type):
        """Show statistics for clicked label"""
        self.last_clicked_label = label_value
        self.last_clicked_position = img_point
        self.last_clicked_view = view_type

        # Get CT/SUV values
        if self.ct_array is not None:
            if len(self.ct_array.shape) == 3:
                ct_val = float(
                    self.ct_array[self.current_slice_idx, img_point.y(), img_point.x()])
                suv_val = float(
                    self.suv_array[self.current_slice_idx, img_point.y(), img_point.x()])
            else:
                ct_val = float(self.ct_array[img_point.y(), img_point.x()])
                suv_val = float(self.suv_array[img_point.y(), img_point.x()])
        else:
            ct_val = 0.0
            suv_val = 0.0

        self.last_clicked_value = (ct_val, suv_val)
        stats = self.get_label_statistics(label_value)

        if stats is not None:
            suv_max, suv_mean, suv_peak, volume_ml, tlg = stats
            self.status_bar.showMessage(
                f"Clicked: Label {label_value} | CT: {ct_val:.1f} HU, SUV: {suv_val:.2f} | "
                f"SUVmax={suv_max:.2f}, SUVmean={suv_mean:.2f}, SUVpeak={suv_peak:.2f}, "
                f"Vol={volume_ml:.2f} ml, TLG={tlg:.2f} | "
                f"Pos: ({img_point.x()},{img_point.y()})"
            )
        else:
            self.status_bar.showMessage(
                f"Clicked: Label {label_value} | CT: {ct_val:.1f} HU, SUV: {suv_val:.2f} | Pos: ({img_point.x()},{img_point.y()})"
            )

    def get_color_name(self, color):
        """Get color name from QColor"""
        r, g, b = color.red(), color.green(), color.blue()
        lookup = {
            (255, 0, 0): "Đỏ", (0, 255, 0): "Xanh lá", (0, 0, 255): "Xanh dương",
            (255, 255, 0): "Vàng", (255, 0, 255): "Magenta", (0, 255, 255): "Cyan",
            (255, 128, 0): "Cam", (128, 0, 255): "Tím", (0, 128, 255): "Xanh lam nhạt",
            (255, 0, 128): "Hồng", (128, 255, 0): "Xanh lá nhạt", (0, 255, 128): "Xanh ngọc",
            (255, 128, 128): "Hồng nhạt", (128, 255, 128): "Xanh lá nhạt", (128, 128, 255): "Tím nhạt",
            (192, 192, 192): "Bạc", (128, 64, 0): "Nâu", (64, 128, 0): "Xanh olive",
            (0, 128, 128): "Xanh đậm", (128, 0, 128): "Tím đậm"
        }
        return lookup.get((r, g, b), f"RGB({r},{g},{b})")

    def change_current_label(self, index):
        self.current_label = index + 1
        color = self.label_colors[self.current_label]
        color_name = self.get_color_name(color)
        self.status_bar.showMessage(
            f"Đã chọn label {self.current_label} - {color_name}")

    def change_brush_size(self, size):
        self.brush_size = size
        self.status_bar.showMessage(f"Brush size changed to: {size}px")

    def toggle_brush_enabled(self):
        self.brush_enabled = self.brush_toggle_btn.isChecked()
        if not self.brush_enabled:
            self.drawing = False
        state = "BẬT" if self.brush_enabled else "TẮT"
        self.brush_toggle_btn.setText(f"Brush {state}")
        self.status_bar.showMessage(f"Brush: {state}")

    def toggle_eraser_mode(self):
        self.eraser_mode = self.eraser_toggle_btn.isChecked()
        state = "BẬT" if self.eraser_mode else "TẮT"
        self.eraser_toggle_btn.setText(f"Tẩy {state}")
        self.status_bar.showMessage(f"Chế độ tẩy: {state}")

    def toggle_masks_visibility(self):
        show_mask = self.mask_visibility.isChecked()
        self.ct_view_data['show_mask'] = show_mask
        self.suv_view_data['show_mask'] = show_mask
        self.ct_view.update()
        self.suv_view.update()

    def toggle_sync_views(self, state):
        self.sync_views = state == Qt.Checked
        if self.sync_views:
            self.sync_view_parameters()

    def sync_view_parameters(self):
        source = self.ct_view_data
        for view in [self.ct_view_data, self.suv_view_data]:
            view['zoom_factor'] = source['zoom_factor']
            view['pan_offset'] = source['pan_offset']
        self.ct_view.update()
        self.suv_view.update()

    def update_ct_contrast(self):
        try:
            min_val = float(self.ct_min_edit.text())
            max_val = float(self.ct_max_edit.text())
            level = (min_val + max_val) / 2
            width = max_val - min_val
            self.ct_level_edit.setText(f"{level:.2f}")
            self.ct_window_edit.setText(f"{width:.2f}")
            self.ct_view_data['min_value'] = min_val
            self.ct_view_data['max_value'] = max_val
            self.ct_view_data['window_level'] = level
            self.ct_view_data['window_width'] = width
            self.ct_view.update()
        except ValueError:
            QMessageBox.warning(self, "Lỗi", "Giá trị nhập không hợp lệ")

    def update_suv_contrast(self):
        try:
            min_val = float(self.suv_min_edit.text())
            max_val = float(self.suv_max_edit.text())
            level = (min_val + max_val) / 2
            width = max_val - min_val
            self.suv_level_edit.setText(f"{level:.2f}")
            self.suv_window_edit.setText(f"{width:.2f}")
            self.suv_view_data['min_value'] = min_val
            self.suv_view_data['max_value'] = max_val
            self.suv_view_data['window_level'] = level
            self.suv_view_data['window_width'] = width
            self.suv_view.update()
        except ValueError:
            QMessageBox.warning(self, "Lỗi", "Giá trị nhập không hợp lệ")

    def calculate_volumes_statistics(self):
        """Calculate volume statistics for all lesions"""
        active_mask = self.get_active_mask()
        if active_mask is None or np.sum(active_mask) == 0:
            QMessageBox.warning(
                self, "Cảnh báo", "Không có mask nào để tính toán")
            return

        try:
            spacing = self.ct_image.GetSpacing()
            voxel_volume_mm3 = spacing[0] * spacing[1] * spacing[2]

            report = f"THỐNG KÊ Label (ml)\n" + "="*60 + "\n\n"
            for label in range(1, self.max_labels + 1):
                label_mask = (active_mask == label)
                total_voxels = np.sum(label_mask)
                if total_voxels == 0:
                    continue

                mtv_ml = total_voxels * voxel_volume_mm3 / 1000.0
                suv_values = self.suv_array[label_mask]
                suv_mean = np.mean(suv_values) if len(suv_values) > 0 else 0
                suv_std = np.std(suv_values) if len(suv_values) > 0 else 0
                suv_max = np.max(suv_values) if len(suv_values) > 0 else 0
                suv_peak = self.calculate_suvpeak_for_label(label)
                tlg = mtv_ml * suv_mean

                color = self.label_colors[label]
                color_name = self.get_color_name(color)
                edited_status = " (ĐÃ CHỈNH SỬA)" if label in self.edited_lesions else ""
                report += f"TỔN THƯƠNG {label} ({color_name}){edited_status}:\n"
                report += f"  MTV (Thể tích): {mtv_ml:.2f} ml\n"
                report += f"  SUV Mean: {suv_mean:.2f} ± {suv_std:.2f}\n"
                report += f"  SUV Max: {suv_max:.2f}\n"
                report += f"  SUV Peak: {suv_peak:.2f}\n"
                report += f"  TLG: {tlg:.2f}\n\n"

            dialog = QDialog(self)
            dialog.setWindowTitle("Volumes and Statistics")
            dialog.setGeometry(100, 100, 600, 800)
            layout = QVBoxLayout(dialog)
            text_edit = QTextEdit()
            text_edit.setPlainText(report)
            text_edit.setReadOnly(True)
            text_edit.setFontFamily("Courier")
            text_edit.setFontPointSize(9)
            layout.addWidget(text_edit)
            close_btn = QPushButton("Close")
            close_btn.clicked.connect(dialog.accept)
            layout.addWidget(close_btn)
            dialog.exec_()
        except Exception as e:
            QMessageBox.critical(
                self, "Lỗi", f"Không thể tính toán statistics: {str(e)}")

    def calculate_suvpeak_for_label(self, label):
        """Calculate SUVpeak for a label"""
        active_mask = self.get_active_mask()
        if active_mask is None or self.suv_array is None:
            return 0.0

        label_mask = (active_mask == label)
        if np.sum(label_mask) == 0:
            return 0.0

        # Simplified SUVpeak calculation
        if len(self.suv_array.shape) == 3:
            max_coords = np.unravel_index(
                np.argmax(self.suv_array * label_mask), self.suv_array.shape)
        else:
            max_coords = np.unravel_index(
                np.argmax(self.suv_array * label_mask), self.suv_array.shape)
            max_coords = (0, max_coords[0], max_coords[1])

        # Return SUVmax as simplified SUVpeak
        suv_values = self.suv_array[label_mask]
        return float(np.max(suv_values)) if len(suv_values) > 0 else 0.0

    def get_label_statistics(self, label):
        """Get statistics for a specific label"""
        active_mask = self.get_active_mask()
        if active_mask is None or self.ct_image is None:
            return None

        label_mask = (active_mask == label)
        if np.sum(label_mask) == 0:
            return None

        suv_values = self.suv_array[label_mask]
        if len(suv_values) == 0:
            return None

        suv_max = float(np.max(suv_values))
        suv_mean = float(np.mean(suv_values))
        suv_peak = self.calculate_suvpeak_for_label(label)

        spacing = self.ct_image.GetSpacing()
        voxel_volume_mm3 = spacing[0] * spacing[1] * spacing[2]
        mtv_ml = np.sum(label_mask) * voxel_volume_mm3 / 1000.0

        tlg = mtv_ml * suv_mean

        return suv_max, suv_mean, suv_peak, mtv_ml, tlg

    # [Các method cho chế độ chỉnh sửa lesion]
    def toggle_edit_lesions_mode(self):
        """Toggle edit lesions mode"""
        if self.edit_lesions_btn.isChecked():
            if self.mask_array is not None and np.any(self.mask_array > 0):
                self.edit_mask_array = self.mask_array.copy()
                self.editing_mode = True
                self.edit_changes_made = False
                self.selected_lesion_label = None
                self.edit_toolbar.setVisible(True)
                self.populate_lesion_selector()
                self.status_bar.showMessage(
                    "Đã bật chế độ chỉnh sửa tổn thương")
            else:
                self.edit_lesions_btn.setChecked(False)
                QMessageBox.warning(
                    self, "Cảnh báo", "Không có mask nào để chỉnh sửa")
        else:
            if self.edit_changes_made:
                reply = QMessageBox.question(
                    self, "Thay đổi chưa được đồng bộ",
                    "Bạn có thay đổi chưa được đồng bộ. Bạn có muốn đồng bộ thay đổi trước khi thoát?",
                    QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel
                )
                if reply == QMessageBox.Yes:
                    self.sync_edit_changes()
                elif reply == QMessageBox.Cancel:
                    self.edit_lesions_btn.setChecked(True)
                    return

            self.editing_mode = False
            self.edit_mask_array = None
            self.selected_lesion_label = None
            self.edit_changes_made = False
            self.edit_toolbar.setVisible(False)
            self.status_bar.showMessage("Đã tắt chế độ chỉnh sửa tổn thương")

        self.show_current_slice()

    def populate_lesion_selector(self):
        """Populate lesion selector combobox"""
        self.lesion_selector.clear()
        if self.edit_mask_array is None:
            return

        unique_labels = np.unique(self.edit_mask_array)
        unique_labels = unique_labels[unique_labels > 0]

        for label in unique_labels:
            voxel_count = np.sum(self.edit_mask_array == label)
            stats = self.get_label_statistics_for_edit(label)
            edited_status = " (ĐÃ SỬA)" if label in self.edited_lesions else ""

            if stats:
                suv_max, suv_mean, suv_peak, volume_ml, tlg = stats
                self.lesion_selector.addItem(
                    f"Tổn thương {label}: {volume_ml:.2f}ml, SUVmax={suv_max:.2f}{edited_status}",
                    label
                )
            else:
                self.lesion_selector.addItem(
                    f"Tổn thương {label}{edited_status}", label)

    def get_label_statistics_for_edit(self, label):
        """Get statistics for label in edit mode"""
        if self.edit_mask_array is None or self.ct_image is None:
            return None

        label_mask = (self.edit_mask_array == label)
        if np.sum(label_mask) == 0:
            return None

        suv_values = self.suv_array[label_mask]
        if len(suv_values) == 0:
            return None

        suv_max = float(np.max(suv_values))
        suv_mean = float(np.mean(suv_values))
        suv_peak = suv_max  # Simplified

        spacing = self.ct_image.GetSpacing()
        voxel_volume_mm3 = spacing[0] * spacing[1] * spacing[2]
        mtv_ml = np.sum(label_mask) * voxel_volume_mm3 / 1000.0

        tlg = mtv_ml * suv_mean

        return suv_max, suv_mean, suv_peak, mtv_ml, tlg

    def select_lesion_for_editing(self, index):
        """Select lesion for editing"""
        if index < 0 or not self.editing_mode:
            self.selected_lesion_label = None
            return

        label = self.lesion_selector.itemData(index)
        self.selected_lesion_label = label
        self.status_bar.showMessage(f"Đã chọn Tổn thương {label} để chỉnh sửa")
        self.ct_view.update()
        self.suv_view.update()

    def update_lesion_threshold(self, value):
        """Update lesion edit threshold"""
        self.lesion_edit_threshold = value
        if self.selected_lesion_label is not None:
            self.status_bar.showMessage(
                f"Ngưỡng chỉnh sửa: {value}% cho tổn thương {self.selected_lesion_label}")

    def apply_threshold_to_selected_lesion(self):
        """Apply threshold to selected lesion"""
        if self.selected_lesion_label is None:
            QMessageBox.warning(
                self, "Cảnh báo", "Vui lòng chọn một tổn thương để áp dụng ngưỡng")
            return

        if not self.editing_mode:
            QMessageBox.warning(
                self, "Cảnh báo", "Vui lòng bật chế độ chỉnh sửa trước")
            return

        try:
            label = self.selected_lesion_label
            lesion_mask = (self.edit_mask_array == label)
            if np.sum(lesion_mask) == 0:
                QMessageBox.warning(
                    self, "Cảnh báo", "Tổn thương không tồn tại")
                return

            lesion_suv_values = self.suv_array[lesion_mask]
            lesion_suv_max = np.max(lesion_suv_values) if len(
                lesion_suv_values) > 0 else 0
            threshold_abs = lesion_suv_max * self.lesion_edit_threshold / 100.0

            lesion_mask_3d = (self.edit_mask_array == label)
            refined_mask_3d = np.logical_and(
                lesion_mask_3d, self.suv_array >= threshold_abs)

            self.edit_mask_array[lesion_mask_3d] = 0
            self.edit_mask_array[refined_mask_3d] = label

            self.edited_lesions.add(label)
            self.manual_editing_done = True
            self.edit_changes_made = True

            self.show_current_slice()
            self.populate_lesion_selector()

            self.status_bar.showMessage(
                f"Đã refine Tổn thương {label} với ngưỡng {self.lesion_edit_threshold}%")

        except Exception as e:
            QMessageBox.critical(
                self, "Lỗi", f"Không thể áp dụng ngưỡng: {str(e)}")

    def delete_selected_lesion(self):
        """Delete selected lesion"""
        if self.selected_lesion_label is None:
            QMessageBox.warning(
                self, "Cảnh báo", "Vui lòng chọn một tổn thương để xóa")
            return

        if not self.editing_mode:
            QMessageBox.warning(
                self, "Cảnh báo", "Vui lòng bật chế độ chỉnh sửa trước")
            return

        reply = QMessageBox.question(
            self, "Xác nhận xóa",
            f"Bạn có chắc muốn xóa Tổn thương {self.selected_lesion_label}?",
            QMessageBox.Yes | QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            label = self.selected_lesion_label
            lesion_mask = (self.edit_mask_array == label)
            self.edit_mask_array[lesion_mask] = 0

            self.selected_lesion_label = None
            self.edited_lesions.discard(label)
            self.edit_changes_made = True

            self.show_current_slice()
            self.populate_lesion_selector()

            if self.lesion_selector.count() == 0:
                self.edit_lesions_btn.setChecked(False)
                self.toggle_edit_lesions_mode()

            self.status_bar.showMessage(f"Đã xóa Tổn thương {label}")

    def mark_lesion_as_edited(self):
        """Mark lesion as edited"""
        if self.selected_lesion_label is None:
            QMessageBox.warning(
                self, "Cảnh báo", "Vui lòng chọn một tổn thương để đánh dấu")
            return

        self.edited_lesions.add(self.selected_lesion_label)
        self.manual_editing_done = True
        self.populate_lesion_selector()
        self.status_bar.showMessage(
            f"Đã đánh dấu Tổn thương {self.selected_lesion_label} là đã chỉnh sửa")

    def sync_edit_changes(self):
        """Sync edit changes to main mask"""
        if self.edit_changes_made and self.edit_mask_array is not None:
            self.mask_array = self.edit_mask_array.copy()
            self.edit_changes_made = False
            self.show_current_slice()
            unique_labels = np.unique(self.mask_array)
            unique_labels = unique_labels[unique_labels > 0]
            self.status_bar.showMessage(
                f"Đã đồng bộ {len(unique_labels)} tổn thương từ chế độ chỉnh sửa")
            QMessageBox.information(
                self, "Thành công", "Đã đồng bộ tất cả thay đổi!")
        else:
            self.status_bar.showMessage("Không có thay đổi nào để đồng bộ")

    def cancel_edit_changes(self):
        """Cancel edit changes"""
        if self.edit_changes_made:
            reply = QMessageBox.question(
                self, "Xác nhận hủy",
                "Bạn có chắc muốn hủy tất cả thay đổi trong chế độ chỉnh sửa?",
                QMessageBox.Yes | QMessageBox.No
            )

            if reply == QMessageBox.Yes:
                if self.mask_array is not None:
                    self.edit_mask_array = self.mask_array.copy()
                self.edit_changes_made = False
                self.edited_lesions.clear()
                self.selected_lesion_label = None
                self.show_current_slice()
                self.populate_lesion_selector()
                self.status_bar.showMessage(
                    "Đã hủy tất cả thay đổi trong chế độ chỉnh sửa")
        else:
            self.status_bar.showMessage("Không có thay đổi nào để hủy")

    def load_segmentation_only(self):
        """Load segmentation only"""
        if self.ct_array is None:
            QMessageBox.warning(
                self, "Cảnh báo", "Vui lòng tải ảnh CT/SUV trước")
            return

        seg_path, _ = QFileDialog.getOpenFileName(
            self, "Chọn ảnh Segmentation", "", "Ảnh y tế (*.nii *.nii.gz *.dcm *.dicom *.mha *.mhd)")

        if not seg_path:
            return

        try:
            self.seg_image = sitk.ReadImage(seg_path)
            self.seg_array = sitk.GetArrayFromImage(
                self.seg_image).astype(np.uint8)
            self.base_refine_mask = None  # reset base khi load segmentation

            if self.seg_array.shape == self.ct_array.shape:
                self.mask_array = self.seg_array.copy()
                self.loaded_segmentation = True
                unique_labels = np.unique(self.mask_array)
                unique_labels = unique_labels[unique_labels > 0]
                self.show_current_slice()
                self.status_bar.showMessage(
                    f"Đã load segmentation với {len(unique_labels)} tổn thương")
            else:
                QMessageBox.warning(self, "Cảnh báo",
                                    f"Ảnh segmentation có kích thước không khớp với CT/SUV")

        except Exception as e:
            QMessageBox.critical(
                self, "Lỗi", f"Không thể load segmentation: {str(e)}")

    def clear_current_slice_mask(self):
        """Clear current slice mask"""
        if self.mask_array is not None:
            if self.editing_mode and self.edit_mask_array is not None:
                self.edit_mask_array[self.current_slice_idx] = 0
                self.edit_changes_made = True
                current_mask = self.edit_mask_array[self.current_slice_idx].copy(
                )
            else:
                self.mask_array[self.current_slice_idx] = 0
                current_mask = np.zeros_like(self.ct_view_data['mask'])

            self.ct_view_data['mask'] = current_mask
            self.suv_view_data['mask'] = current_mask.copy()
            self.clear_click_info()
            self.ct_view.update()
            self.suv_view.update()

            status_msg = f"Đã xóa mask của slice {self.current_slice_idx}"
            if self.editing_mode:
                status_msg += " (Thay đổi chưa được đồng bộ)"
            self.status_bar.showMessage(status_msg)

    def clear_current_label(self):
        """Clear current label from all slices"""
        mask_to_use = self.edit_mask_array if self.editing_mode else self.mask_array

        if mask_to_use is not None and np.any(mask_to_use == self.current_label):
            reply = QMessageBox.question(self, "Xác nhận",
                                         f"Bạn có chắc muốn xóa Label {self.current_label} trên TOÀN BỘ volume?",
                                         QMessageBox.Yes | QMessageBox.No)
            if reply == QMessageBox.Yes:
                lesion_mask = (mask_to_use == self.current_label)
                mask_to_use[lesion_mask] = 0
                self.clear_click_info()
                self.edited_lesions.discard(self.current_label)

                if self.editing_mode:
                    self.edit_changes_made = True

                self.show_current_slice()

                status_msg = f"Đã xóa label {self.current_label} trên toàn bộ volume"
                if self.editing_mode:
                    status_msg += " (Thay đổi chưa được đồng bộ)"
                self.status_bar.showMessage(status_msg)
        else:
            QMessageBox.information(
                self, "Thông tin", f"Label {self.current_label} không tồn tại trong volume")

    def clear_all_masks(self):
        """Clear all masks"""
        reply = QMessageBox.question(self, "Xác nhận", "Bạn có chắc chắn muốn xóa tất cả masks?",
                                     QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            if self.mask_array is not None:
                self.mask_array.fill(0)
            if self.edit_mask_array is not None:
                self.edit_mask_array.fill(0)
            self.base_refine_mask = None

            self.ct_view_data['mask'] = np.zeros_like(
                self.ct_view_data['mask'])
            self.suv_view_data['mask'] = np.zeros_like(
                self.suv_view_data['mask'])
            self.clear_click_info()
            self.edited_lesions.clear()
            self.manual_editing_done = False
            self.edit_changes_made = False
            self.ct_view.update()
            self.suv_view.update()
            self.status_bar.showMessage("Đã xóa tất cả masks")

    def calculate_current_slice_statistics(self):
        """Calculate statistics for current slice"""
        if self.ct_view_data['mask'] is None or not np.any(self.ct_view_data['mask'] > 0):
            QMessageBox.warning(
                self, "Cảnh báo", "Slice hiện tại không có mask")
            return

        try:
            spacing = self.ct_image.GetSpacing()
            pixel_area_mm2 = spacing[0] * spacing[1]
            ct_slice = self.ct_view_data['image_array']
            suv_slice = self.suv_view_data['image_array']

            report = f"THỐNG KÊ SLICE {self.current_slice_idx}\n" + \
                "="*40 + "\n\n"

            for label in range(1, self.max_labels + 1):
                label_mask = (self.ct_view_data['mask'] == label)
                voxel_count = np.sum(label_mask)
                if voxel_count == 0:
                    continue

                area_mm2 = voxel_count * pixel_area_mm2
                ct_values = ct_slice[label_mask]
                suv_values = suv_slice[label_mask]
                ct_mean = np.mean(ct_values) if len(ct_values) > 0 else 0
                ct_std = np.std(ct_values) if len(ct_values) > 0 else 0
                suv_mean = np.mean(suv_values) if len(suv_values) > 0 else 0
                suv_std = np.std(suv_values) if len(suv_values) > 0 else 0
                suv_max = np.max(suv_values) if len(suv_values) > 0 else 0

                color = self.label_colors[label]
                color_name = self.get_color_name(color)
                edited_status = " (ĐÃ CHỈNH SỬA)" if label in self.edited_lesions else ""

                report += f"TỔN THƯƠNG {label} ({color_name}){edited_status}:\n"
                report += f"  VOXELS: {voxel_count:,}\n"
                report += f"  DIỆN TÍCH: {area_mm2:.2f} mm²\n"
                report += f"  CT Mean: {ct_mean:.2f} ± {ct_std:.2f}\n"
                report += f"  SUV Mean: {suv_mean:.2f} ± {suv_std:.2f}\n"
                report += f"  SUV Max: {suv_max:.2f}\n\n"

            QMessageBox.information(self, "Thống kê Slice", report)

        except Exception as e:
            QMessageBox.critical(
                self, "Lỗi", f"Không thể tính toán statistics: {str(e)}")


def main():
    app = QApplication(sys.argv)
    window = MedicalSegmentationApp()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
