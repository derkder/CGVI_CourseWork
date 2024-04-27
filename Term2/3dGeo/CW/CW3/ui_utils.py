import open3d as o3d
import numpy as np
import sys
from meshdeform_utils import *
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, QSlider, QLabel, QHBoxLayout
from PyQt5.QtCore import Qt

class PointCloudVisualizer:
    def __init__(self, cloud):
        self.cloud = cloud
        self.vis = o3d.visualization.VisualizerWithEditing()
        self.vis.create_window()
        self.vis.add_geometry(self.cloud)

    def pick_points(self):
        self.vis.run()  # User can pick points in the Open3D window
        self.vis.destroy_window()
        return self.vis.get_picked_points()

class PointCloudApp(QMainWindow):
    def __init__(self, mesh_path):
        super().__init__()
        self.setWindowTitle("PointCloud Manipulation App")
        self.setGeometry(100, 100, 800, 600)
        
        # Load point cloud
        self.mesh = o3d.io.read_triangle_mesh(mesh_path)
        self.point_cloud = o3d.io.read_point_cloud(mesh_path)
        
        # Main widget and layout
        self.main_widget = QWidget(self)
        self.setCentralWidget(self.main_widget)
        self.layout = QVBoxLayout()
        
        # Buttons
        self.btn_select_points = QPushButton("Select Four Points", self)
        self.btn_select_points.clicked.connect(self.select_four_points)
        self.layout.addWidget(self.btn_select_points)

        self.btn_select_points = QPushButton("Select One Point", self)
        self.btn_select_points.clicked.connect(self.select_one_point)
        self.layout.addWidget(self.btn_select_points)
        
        self.btn_adjust_point = QPushButton("Adjust Point Position", self)
        self.btn_adjust_point.clicked.connect(self.adjust_point_position)
        self.layout.addWidget(self.btn_adjust_point)

        self.btn_mesh_deform_manual = QPushButton("Apply Mesh Deformation Manually", self)
        self.btn_mesh_deform_manual.clicked.connect(lambda: self.apply_mesh_deformation(True))
        self.layout.addWidget(self.btn_mesh_deform_manual)

        self.btn_mesh_deform_auto = QPushButton("Apply Mesh Deformation Automatically", self)
        self.btn_mesh_deform_auto.clicked.connect(lambda: self.apply_mesh_deformation(False))
        self.layout.addWidget(self.btn_mesh_deform_auto)

        
        # Set layout
        self.main_widget.setLayout(self.layout)
        
        # List to store selected points
        self.control_points_indices = []
        self.handle_point_index = None

        self.adjustment_window = None

        # The currently selected point for adjustment
        self.current_point_index = None

    def select_four_points(self):
        visualizer = PointCloudVisualizer(self.point_cloud)
        picked_indices = visualizer.pick_points()
        if len(picked_indices) >= 4:
            self.control_points_indices = picked_indices[:4]
            print("Selected Control Points:", np.asarray(self.control_points_indices))
        else:
            print("Not enough points selected.")

    def select_one_point(self):
        visualizer = PointCloudVisualizer(self.point_cloud)
        picked_indices = visualizer.pick_points()
        if len(picked_indices) >= 1:
            self.handle_point_index = picked_indices[0]
            print("Selected Handle Point:", np.asarray(self.handle_point_index))
        else:
            print("Not enough points selected.")

    def adjust_point_position(self):
        if not self.handle_point_index:
            print("No points selected yet.")
            return
        
        # Open a new window to adjust the position of the first selected point
        self.adjustment_window = PointAdjustmentWindow(self.point_cloud.points[self.handle_point_index])
        self.adjustment_window.show()

    def apply_mesh_deformation(self, isManually):
        if(isManually):
            if not self.control_points_indices:
                print("No control points selected yet.")
                return
            if not self.handle_point_index:
                print("No handle points selected yet.")
                return
            if not self.adjustment_window:
                print("Havent adjusted yet.")
                return
            mesh_deformation(self.control_points_indices, self.handle_point_index, self.adjustment_window.displacements, self.mesh, isManually = True)
        else:
            mesh_deformation(mesh = self.mesh, isManually = False)

class PointAdjustmentWindow(QWidget):
    def __init__(self, point):
        super().__init__()
        self.original_point = np.array(point)
        self.setWindowTitle("Adjust Point Position")
        self.setGeometry(150, 150, 300, 200)
        self.layout = QVBoxLayout()
        
        # Array to store the displacement values from sliders, multiplied by 10 for finer control
        self.displacements = [0, 0, 0]  # X, Y, Z displacements in tenths
        
        # Sliders for XYZ with labels
        self.slider_x = self.create_slider("X", 0)
        self.slider_y = self.create_slider("Y", 0)
        self.slider_z = self.create_slider("Z", 0)
        
        self.setLayout(self.layout)

    def create_slider(self, label_text, initial_value):
        hbox = QHBoxLayout()
        label = QLabel(f"{label_text} Displacement: {initial_value / 100:.2f}")  # Displaying as float
        slider = QSlider(Qt.Horizontal)
        slider.setMinimum(-100)
        slider.setMaximum(100)
        slider.setValue(initial_value)  # Start with zero displacement
        slider.valueChanged.connect(lambda value, lbl=label, idx=label_text: self.update_point_position(value, lbl, idx))
        hbox.addWidget(label)
        hbox.addWidget(slider)
        self.layout.addLayout(hbox)
        return slider

    def update_point_position(self, value, label, index):
        # Update the displacement value
        self.displacements[index_map[index]] = value / 100
        # Update label to reflect the displacement as a float
        label.setText(f"{index} Displacement: {value / 100:.2f}")  # Convert to float for display
        print(self.displacements)

index_map = {"X": 0, "Y": 1, "Z": 2}


