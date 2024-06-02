import sys
import open3d as o3d
import trimesh
import vtk
import pymesh
from tkinter import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from GUI import Ui_control
from PyQt5 import QtCore, QtWidgets, QtGui
from PyQt5.QtCore import Qt, QTimer,QPointF,QObject,pyqtSignal
from PyQt5.QtGui import  QPainter, QColor, QPen, QBrush
from PyQt5.QtWidgets import QGraphicsPixmapItem, QGraphicsScene, QFileDialog,QGraphicsItem,QOpenGLWidget
import mesh
import math
import numpy as np
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
from numpy import linalg as LA


#mesh_ply = o3d.io.read_triangle_mesh("D:/model/cow-2.ply")
#mesh_ply.compute_vertex_normals()#渲染
#print(len(mesh_ply.vertices))


#o3d.visualization.draw_geometries([mesh_ply],window_name="原模型")
#转换为obj格式
#mesh=o3d.geometry.TriangleMesh()
#mesh.vertices=mesh_ply.vertices
#mesh.triangles=mesh_ply.triangles
#mesh.vertex_normals=mesh_ply.vertex_normals
#mesh.vertex_colors=mesh_ply.vertex_colors

#界面设计
class Mywindow(QtWidgets.QMainWindow,Ui_control):
    model_loaded = pyqtSignal(object)
    def __init__(self):
        super(Mywindow, self).__init__()
        self.setupUi(self)
        self.port_list.addItem("")
        self.port_list.addItem("顶点删除法")
        self.port_list.addItem("顶点聚类法")
        self.port_list.addItem("边折叠法")
        #self.getCurvature()

        # def Model_Express(self):



    def Ratio_Enter(self):

        self.ratio=float(self.text_edit.text())
        print(self.ratio)


    def Load_Model(self):
       # global mesh
       # global vertices,triangles,fileName
        global fileName
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getOpenFileName(None, "Load Model", "",
                                                  "Model Files (*.obj *.ply )", options=options)
        print(fileName)
        if fileName:
            mesh = o3d.io.read_triangle_mesh(fileName)
            #self.update_file_info()
            self.vertices = mesh.vertices
            self.triangles = mesh.triangles
            self.num_vertices = len(self.vertices)
            self.num_triangles=len(self.triangles)
            print(self.num_vertices)
            print(self.num_triangles)

            self.num_vertices_int=float(self.num_vertices)
            self.num_triangles_int = float(self.num_triangles)
            print(self.num_triangles_int)
            self.get_max_bound=mesh.get_max_bound()
            self.get_min_bound=mesh.get_min_bound()
            print(self.get_max_bound)
            print(self.get_min_bound)
            self.fileName=fileName
            #mesh.compute_vertex_normals()  # 渲染
            mesh.compute_vertex_normals()  # 渲染
            o3d.visualization.draw_geometries([mesh], window_name="原模型")

    #def update_file_info(self):
        # 获取文件大小
     #   file_size = os.path.getsize(self.fileName)
        # 转换文件大小为可读格式
     #   self.file_size_str = "{} KB".format(round(file_size / 1024, 2))
     #   self.label_file_size.setText("文件大小: {}".format(self.file_size_str))
     #   self.label_vertex_count.setText("顶点数量: {}".format(self.num_vertices))


    def Quadric_Decimation(self):
        scene = trimesh.load(self.fileName)

        # 检查场景是否包含任何几何体
        if scene.geometry:
            # 获取场景中的第一个几何体
            mesh = scene.dump(concatenate=True)
        else:
            raise ValueError("该场景不包含任何几何体")

        #mesh = trimesh.load_mesh(self.fileName)
        simplification_rate=self.ratio
        vertex_defects = trimesh.curvature.vertex_defects(mesh)


        # 计算目标顶点数
        original_vertex_count = len(mesh.vertices)
        target_vertex_count = int(original_vertex_count * (1 - simplification_rate))

        # 确定曲率阈值
        sorted_indices = np.argsort(vertex_defects)
        threshold_index = sorted_indices[-target_vertex_count]
        curvature_threshold = vertex_defects[threshold_index]

        # 选择曲率低于阈值的顶点
        low_curvature_indices = np.where(vertex_defects < curvature_threshold)[0]

        # 创建一个布尔掩码，初始值全为True
        vertex_mask = np.ones(mesh.vertices.shape[0], dtype=bool)

        # 将低曲率顶点的掩码值设为False
        vertex_mask[low_curvature_indices] = False

        # 更新顶点
        new_vertices = mesh.vertices[vertex_mask]

        # 创建一个映射，从旧顶点索引到新顶点索引
        old_to_new_index = -np.ones(mesh.vertices.shape[0], dtype=int)
        old_to_new_index[vertex_mask] = np.arange(np.sum(vertex_mask))

        # 更新面
        new_faces = []
        for face in mesh.faces:
            if all(vertex_mask[face]):
                new_faces.append(old_to_new_index[face])

        new_faces = np.array(new_faces)

        # 创建简化后的网格
        simplified_mesh = trimesh.Trimesh(vertices=new_vertices, faces=new_faces)
        print(simplified_mesh.vertices)
        # 保存简化后的网格为OBJ文件
        simplified_mesh_path = 'D:/graduation/mesh_vd.obj'
        simplified_mesh.export(simplified_mesh_path)
        # 使用open3d加载简化后的.obj文件
        simplified_mesh_o3d = o3d.io.read_triangle_mesh(simplified_mesh_path)
        simplified_mesh_o3d = o3d.t.geometry.TriangleMesh.from_legacy(simplified_mesh_o3d)

        # 填补简化后网格中的空洞
        filled_mesh = simplified_mesh_o3d.fill_holes()

        # 保存填补空洞后的网格
        simplified_mesh_filled_path = 'D:/graduation/mesh_vd_filled.obj'
        o3d.io.write_triangle_mesh(simplified_mesh_filled_path, filled_mesh.to_legacy())

        # 加载和可视化原始和简化后的.obj文件
        original_mesh_o3d = o3d.io.read_triangle_mesh(self.fileName)
        simplified_mesh_filled_o3d = o3d.io.read_triangle_mesh(simplified_mesh_filled_path)

        # 设置绘制选项
        original_mesh_o3d.compute_vertex_normals()
        simplified_mesh_filled_o3d.compute_vertex_normals()
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="Simplified Mesh", width=800, height=600)
        vis.add_geometry(simplified_mesh_filled_o3d)
        vis.run()
        vis.destroy_window()

    def Vertex_Clustering(self):
        mesh = o3d.io.read_triangle_mesh(self.fileName)
        if  max(self.get_max_bound - self.get_min_bound)>1000:
            voxel_size = max(self.get_max_bound - self.get_min_bound) *(1-self.ratio)*0.01
        else:
            voxel_size = max(self.get_max_bound - self.get_min_bound) *(1-self.ratio)
        #voxel_size = max(self.get_max_bound - self.get_min_bound) /(self.num_vertices*self.ratio)
        print(voxel_size)
        mesh_smp = mesh.simplify_vertex_clustering(
            voxel_size=voxel_size,
            contraction=o3d.geometry.SimplificationContraction.Average)
        self.vertices_smp=len(mesh_smp.vertices)
        self.triangles_smp=len(mesh_smp.triangles)
        print(self.vertices_smp)
        print(self.triangles_smp)
        obj_file_path = "D:/graduation/mesh_smp.obj"

        # 保存mesh为OBJ文件
        o3d.io.write_triangle_mesh(obj_file_path, mesh_smp)

        print(f"Mesh saved as {obj_file_path}")
        mesh_smp.compute_vertex_normals()  # 渲染
        o3d.visualization.draw_geometries([mesh_smp], window_name="顶点聚类法")

    def QEM(self):
     mesh = o3d.io.read_triangle_mesh(self.fileName)
     target_number_of_triangles=(self.num_triangles_int)*(self.ratio)
     target_number_of_triangles_int=math.floor(float(target_number_of_triangles))
     print()
     print(target_number_of_triangles_int)
     #mesh = o3d.geometry.TriangleMesh(fileName)
     mesh_td = mesh.simplify_quadric_decimation(
        target_number_of_triangles=target_number_of_triangles_int)
     self.vertices_td=len(mesh_td.vertices)
     self.triangles_td=len(mesh_td.triangles)
     print(self.vertices_td)
     print(self.triangles_td)
     obj_file_path = "D:/graduation/mesh_qem.obj"

     #保存mesh为OBJ文件
     o3d.io.write_triangle_mesh(obj_file_path, mesh_td)

     print(f"Mesh saved as {obj_file_path}")
     mesh_td.compute_vertex_normals()  #渲染
     o3d.visualization.draw_geometries([mesh_td], window_name="边折叠法")

    def Mode_Selected(self):
       # if self.fileName=='' or self.ratio=='':
        #    print("未导入模型或输入简化率")
       # else:
            mode_selected=self.port_list.currentText()
            if mode_selected=="顶点删除法":
                self.Quadric_Decimation()
            if mode_selected=="顶点聚类法":
                self.Vertex_Clustering()
            if mode_selected=="边折叠法":
                self.QEM()

    def display_model(self):
        # 读取简化后的模型
        mesh_smp = o3d.io.read_triangle_mesh("simplified_model.stl")

        # 将简化后的模型转换为 numpy 数组
        vertices = np.asarray(mesh_smp.vertices)
        triangles = np.asarray(mesh_smp.triangles)

        # 创建 QGraphicsScene
        scene = QtWidgets.QGraphicsScene()
        self.model2_display.setScene(scene)

        # 创建 QGraphicsPolygonItem
        for triangle in triangles:
            poly = QtGui.QPolygonF()
            for vertex_id in triangle:
                vertex = vertices[vertex_id]
                poly.append(QtCore.QPointF(vertex[0], vertex[1]))  # 仅考虑前两个坐标，忽略 Z 轴
            scene.addPolygon(poly, QtGui.QPen(QtCore.Qt.black), QtGui.QBrush(QtCore.Qt.gray))

    def obj_to_ply(self):
        self.ply_file_path = "D:/model/mesh.ply"


    #def Deep_Learning(self):





if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = Mywindow()
    window.setFixedSize(600, 400)
    window.show()
    sys.exit(app.exec_())










