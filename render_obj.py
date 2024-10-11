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








def parse_yaml_to_numpy(filepath):
    """
    Parses a YAML file containing 3D points into a dictionary of lists of numpy arrays.
    
    Parameters:
    - filepath: Path to the YAML file
    
    Returns:
    - Dictionary where each key corresponds to a YAML section (e.g. 'point_normals')
      and the value is a list of numpy arrays representing 3D points.
    """
    # Open and load the YAML file
    with open(filepath, 'r') as file:
        data = yaml.safe_load(file)
    
    # Initialize a dictionary to hold the parsed points
    parsed_data = {}
    
    # Iterate over each key-value pair in the loaded YAML data
    for key, value in data.items():
        if isinstance(value, list) and isinstance(value[0], list) and len(value[0]) == 3:
            # Convert each 3D point into a numpy array
            parsed_data[key] = [np.array(point) for point in value]
        else:
            # Skip keys that don't match the 3D point structure
            print(f"Skipping non-point data: {key}")
    
    return parsed_data

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

def load_mesh(i):
    root_path = "../16k/"
    obj_name = "model.obj"
    folder_path = str(i) + "/"
    obj_path = os.path.join(root_path + folder_path + obj_name)
    print(obj_path)

    mesh = trimesh.load(obj_path)
    return mesh

def create_geometry_at_point(geometry_type, position):
    """
    Creates a geometry at a given 3D point in space using trimesh.
    
    Parameters:
    - geometry_type: A string specifying the type of geometry ('sphere', 'box', etc.)
    - position: A 3D numpy array or list specifying the (x, y, z) coordinates
    
    Returns:
    - Transformed trimesh object with geometry at the given position
    """
    # Create the geometry based on the type
    if geometry_type == 'sphere':
        geometry = trimesh.creation.icosphere(radius=0.001)
    elif geometry_type == 'box':
        geometry = trimesh.creation.box(extents=[0.002, 0.002, 0.002])
    elif geometry_type == 'cylinder':
        geometry = trimesh.creation.cylinder(radius=0.001, height=0.002)
    else:
        raise ValueError(f"Unsupported geometry type: {geometry_type}")
    
    # Create a translation matrix to move the geometry to the desired position
    translation_matrix = np.eye(4)
    translation_matrix[:3, 3] = position
    
    # Apply the translation to the geometry
    geometry.apply_transform(translation_matrix)
    
    return geometry

def create_look_at_matrix(camera_position, target_position, up_vector=[0, 1, 0]):
    """
    Manually create a 'look-at' transformation matrix.

    Parameters:
    - camera_position: The position of the camera (3D array).
    - target_position: The point the camera is looking at (3D array).
    - up_vector: The up direction vector (default is [0, 1, 0]).

    Returns:
    - A 4x4 transformation matrix.
    """
    # Convert input to numpy arrays
    camera_position = np.array(camera_position)
    target_position = np.array(target_position)
    up_vector = np.array(up_vector)

    # Forward vector: from camera to the target (this is the viewing direction)
    forward = target_position - camera_position
    forward /= np.linalg.norm(forward)  # Normalize the forward vector

    # Right vector: perpendicular to the forward and up vectors
    right = np.cross(forward, up_vector)
    right /= np.linalg.norm(right)  # Normalize the right vector

    # Recompute the true up vector as perpendicular to forward and right
    up = np.cross(right, forward)

    # Create a 4x4 view matrix (look-at matrix)
    view_matrix = np.eye(4)
    view_matrix[:3, 0] = right  # X-axis
    view_matrix[:3, 1] = up     # Y-axis
    view_matrix[:3, 2] = -forward  # Z-axis (negative because forward is the opposite of the camera direction)
    view_matrix[:3, 3] = camera_position  # Camera position
#     print(view_matrix)


    return view_matrix

def get_camera_transform(mesh, point_index, distance=0.1):
    """
    Focuses the camera on a specific point on the mesh, positioning the camera normal
    to the surface of the mesh at that point, and ensuring the point is centered in the view.

    Parameters:
    - scene: A Trimesh scene object that contains the mesh.
    - mesh: The Trimesh mesh object.
    - point_index: The index of the point on the mesh (vertex index).
    - distance: The distance of the camera from the point along the normal vector (default is 5.0).

    Returns:
    - updated_scene: The scene with the camera focused on the specified point.
    """
    # Get the vertex coordinates at the specified index
    point = mesh.vertices[point_index]
    
    # Get the surface normal at the point (vertex normal)
    normal = mesh.vertex_normals[point_index]
    
    # Calculate the camera position by moving along the normal direction
    camera_position = point + normal * distance
    
    # Create a 'look-at' matrix to orient the camera to face the point
    camera_transform = create_look_at_matrix(camera_position, point)
    
    return camera_transform
    

#run main
if __name__ == "__main__":
    main()
# trim = mesh

# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')


# ax.plot_trisurf(trim.vertices[:, 0], trim.vertices[:,1], trim.vertices[:,2], triangles=trim.faces)
# plt.show()
