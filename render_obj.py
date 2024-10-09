import trimesh
import os
import matplotlib.pyplot as plt
import numpy as np

def main():

    root_path = "./16k/"
    obj_name = "model.obj"
    folder_path = "1/"
    obj_path = os.path.join(root_path + folder_path + obj_name)
    print(obj_path)

    mesh = trimesh.load(obj_path)
    point = np.array([0.5157325519524518,
  -0.06133666148616426,
  0.2341906828632054])
    # point = np.array([0.03211031095297248, 0.02013190000973366, 0.0072423226751939015])
    vertex_index, vertex = find_nearest_vertex(mesh, point)
    # vertex = mesh.vertices[vertex_index]
    vertex_norm = vertex_normal(mesh, vertex_index)
    
    print("Given point: ", point)
    print("Vertex Index: ",vertex_index)
    print("Vertex Values: ", vertex)
    print("Vertex Norm: ", vertex_norm)
    # breakpoint()
    mesh.show()








def vertex_normal(mesh, vertex_index):
    # Get the faces that share this vertex
    faces = mesh.vertex_faces[vertex_index]

    # Filter out any -1 values (if the vertex is on a boundary, it might not have full face coverage)
    faces = faces[faces != -1]

    # Get the normals of the faces that share the vertex
    face_normals = mesh.face_normals[faces]

    # Compute the average normal (vertex normal)
    vertex_normal = np.mean(face_normals, axis=0)

    # Normalize the resulting normal vector
    vertex_normal /= np.linalg.norm(vertex_normal)
    print(f"Surface normal at vertex {vertex_index}: {vertex_normal}")
    return vertex_normal

def find_nearest_vertex(mesh, point):
    """
    Finds the index of the nearest vertex to a given point in a mesh.
    
    Parameters:
    - mesh: The Trimesh object
    - point: A 3D point as a numpy array (shape: (3,))
    
    Returns:
    - nearest_vertex_index: Index of the nearest vertex
    - nearest_vertex: Coordinates of the nearest vertex
    """
    # Get the vertices of the mesh
    vertices = mesh.vertices
    
    # Compute the Euclidean distance from the point to each vertex
    distances = np.linalg.norm(vertices - point, axis=1)
    
    # Find the index of the minimum distance
    nearest_vertex_index = np.argmin(distances)
    
    # Get the coordinates of the nearest vertex
    nearest_vertex = vertices[nearest_vertex_index]
    
    return nearest_vertex_index, nearest_vertex

#run main
if __name__ == "__main__":
    main()
# trim = mesh

# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')


# ax.plot_trisurf(trim.vertices[:, 0], trim.vertices[:,1], trim.vertices[:,2], triangles=trim.faces)
# plt.show()
