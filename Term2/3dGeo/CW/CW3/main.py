from ui_utils import *


# Entry point for the application
if __name__ == "__main__":
    app = QApplication(sys.argv)
    mesh_path = 'meshes/bun_zipper_res2.ply'
    main_window = PointCloudApp(mesh_path)
    main_window.show()
    sys.exit(app.exec_())
