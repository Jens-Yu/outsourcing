import trimesh
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

def simplify_mesh_by_rate(mesh, simplification_rate):
    # 计算顶点缺陷（曲率）
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

    return simplified_mesh, vertex_defects, curvature_threshold

# 加载三维模型
original_mesh_path = 'D:/Programfiles/outsourcing/mesh_simplification/Airplane1.obj'
scene = trimesh.load(original_mesh_path)

# 检查场景是否包含任何几何体
if scene.geometry:
    # 获取场景中的第一个几何体
    mesh = scene.dump(concatenate=True)
else:
    raise ValueError("该场景不包含任何几何体")

# 保存原始网格以供比较
original_mesh = mesh.copy()

# 根据简化率确定曲率阈值并简化网格
simplification_rate = 0.2  # 20%的简化率
simplified_mesh, vertex_defects, curvature_threshold = simplify_mesh_by_rate(mesh, simplification_rate)

# 保存简化后的网格为OBJ文件
simplified_mesh_path = 'D:/Programfiles/outsourcing/mesh_simplification/AirPlane1_simplified.obj'
simplified_mesh.export(simplified_mesh_path)

# 使用open3d加载简化后的.obj文件
simplified_mesh_o3d = o3d.io.read_triangle_mesh(simplified_mesh_path)
simplified_mesh_o3d = o3d.t.geometry.TriangleMesh.from_legacy(simplified_mesh_o3d)

# 填补简化后网格中的空洞
filled_mesh = simplified_mesh_o3d.fill_holes()

# 保存填补空洞后的网格
simplified_mesh_filled_path = 'D:/Programfiles/outsourcing/mesh_simplification/AirPlane1_simplified_filled.obj'
o3d.io.write_triangle_mesh(simplified_mesh_filled_path, filled_mesh.to_legacy())

# 加载和可视化原始和简化后的.obj文件
original_mesh_o3d = o3d.io.read_triangle_mesh(original_mesh_path)
simplified_mesh_filled_o3d = o3d.io.read_triangle_mesh(simplified_mesh_filled_path)

# 设置绘制选项
original_mesh_o3d.compute_vertex_normals()
simplified_mesh_filled_o3d.compute_vertex_normals()

# 创建open3d可视化窗口
vis = o3d.visualization.Visualizer()
vis.create_window(window_name="Original Mesh", width=800, height=600)
vis.add_geometry(original_mesh_o3d)
vis.run()
vis.destroy_window()

vis = o3d.visualization.Visualizer()
vis.create_window(window_name="Simplified and Filled Mesh", width=800, height=600)
vis.add_geometry(simplified_mesh_filled_o3d)
vis.run()
vis.destroy_window()

# 可视化曲率分布（可选）
plt.figure()
plt.hist(vertex_defects, bins=50)
plt.axvline(curvature_threshold, color='r', linestyle='dashed', linewidth=2)
plt.xlabel('Curvature')
plt.ylabel('Frequency')
plt.title('Curvature Distribution')
plt.show()





'''import trimesh
import numpy as np
import vedo
import matplotlib.pyplot as plt

def simplify_mesh_by_rate(mesh, simplification_rate):
    # 计算顶点缺陷（曲率）
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

    return simplified_mesh, vertex_defects, curvature_threshold

# 加载三维模型
original_mesh_path = 'D:/Programfiles/outsourcing/mesh_simplification/AirPlane0.obj'
mesh = trimesh.load_mesh(original_mesh_path)

# 保存原始网格以供比较
original_mesh = mesh.copy()

# 根据简化率确定曲率阈值并简化网格
simplification_rate = 0.1  # 10%的简化率
simplified_mesh, vertex_defects, curvature_threshold = simplify_mesh_by_rate(mesh, simplification_rate)

# 保存简化后的网格为OBJ文件
simplified_mesh_path = 'D:/Programfiles/outsourcing/mesh_simplification/AirPlane0_simplified.obj'
simplified_mesh.export(simplified_mesh_path)

# 使用vedo加载和可视化原始和简化后的.obj文件
original_mesh_vtk = vedo.load(original_mesh_path)
simplified_mesh_vtk = vedo.load(simplified_mesh_path)

# 创建一个vedo Plotter
plotter = vedo.Plotter(shape=(1, 2), title="Mesh Simplification")

# 在左侧绘制原始网格
plotter.show(original_mesh_vtk, at=0, axes=1, title="Original Mesh")

# 在右侧绘制简化后的网格
plotter.show(simplified_mesh_vtk, at=1, axes=1, title="Simplified Mesh")

# 显示图形
plotter.interactive().close()

# 可视化曲率分布（可选）
plt.figure()
plt.hist(vertex_defects, bins=50)
plt.axvline(curvature_threshold, color='r', linestyle='dashed', linewidth=2)
plt.xlabel('Curvature')
plt.ylabel('Frequency')
plt.title('Curvature Distribution')
plt.show()'''

'''


import trimesh
import numpy as np
import vedo

def simplify_mesh_by_rate(mesh, simplification_rate):
    # 计算顶点缺陷（曲率）
    vertex_defects = trimesh.curvature.vertex_defects(mesh)

    # 计算目标顶点数
    original_vertex_count = len(mesh.vertices)
    target_vertex_count = int(original_vertex_count * (1 - simplification_rate))

    # 确定曲率阈值
    sorted_indices = np.argsort(vertex_defects)
    threshold_index = sorted_indices[target_vertex_count]
    curvature_threshold = vertex_defects[threshold_index]

    # 选择曲率高于阈值的顶点
    high_curvature_indices = np.where(vertex_defects > curvature_threshold)[0]

    # 创建一个布尔掩码，初始值全为True
    vertex_mask = np.ones(mesh.vertices.shape[0], dtype=bool)

    # 将高曲率顶点的掩码值设为False
    vertex_mask[high_curvature_indices] = False

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

    return simplified_mesh, vertex_defects, curvature_threshold

# 加载三维模型
original_mesh_path = 'D:/Programfiles/outsourcing/mesh_simplification/AirPlane0.obj'
mesh = trimesh.load_mesh(original_mesh_path)

# 根据简化率确定曲率阈值并简化网格
simplification_rate = 0.1  # 50%的简化率
simplified_mesh, vertex_defects, curvature_threshold = simplify_mesh_by_rate(mesh, simplification_rate)

# 保存简化后的网格为OBJ文件
simplified_mesh_path = 'D:/Programfiles/outsourcing/mesh_simplification/AirPlane0_simplified.obj'
simplified_mesh.export(simplified_mesh_path)

# 使用vedo加载和可视化原始和简化后的.obj文件
original_mesh_vtk = vedo.load(original_mesh_path)
simplified_mesh_vtk = vedo.load(simplified_mesh_path)

# 创建一个vedo Plotter
plotter = vedo.Plotter(shape=(1, 2), title="Mesh Simplification")

# 在左侧绘制原始网格
plotter.show(original_mesh_vtk, at=0, axes=1, title="Original Mesh")

# 在右侧绘制简化后的网格
plotter.show(simplified_mesh_vtk, at=1, axes=1, title="Simplified Mesh")

# 显示图形
plotter.interactive().close()

# 可视化曲率分布（可选）
import matplotlib.pyplot as plt

plt.figure()
plt.hist(vertex_defects, bins=50)
plt.axvline(curvature_threshold, color='r', linestyle='dashed', linewidth=2)
plt.xlabel('Curvature')
plt.ylabel('Frequency')
plt.title('Curvature Distribution')
plt.show()'''
