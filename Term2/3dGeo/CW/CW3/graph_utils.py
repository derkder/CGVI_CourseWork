import heapq
import numpy as np
import open3d as o3d

class PointCloudGraph:
    def __init__(self, num_vertices, vertices):
        self.num_vertices = num_vertices
        self.vertices = vertices
        self.adjacency_matrix = np.zeros((num_vertices, num_vertices), dtype=int)

    def add_edge(self, v1, v2):
        if 0 <= v1 < self.num_vertices and 0 <= v2 < self.num_vertices:
            self.adjacency_matrix[v1, v2] = 1
            self.adjacency_matrix[v2, v1] = 1

    def sub_graph(self, indices):
        self.vertices = self.vertices[indices, :]
        self.adjacency_matrix = self.adjacency_matrix[indices, :][:, indices]
        self.num_vertices = len(self.vertices)

    def a_star(self, start, end):
        # Heuristic function: Euclidean distance
        heuristic = lambda x, y: np.linalg.norm(self.vertices[x] - self.vertices[y])

        # Priority queue for A* algorithm
        open_set = []
        heapq.heappush(open_set, (0 + heuristic(start, end), start, [start]))  # (cost + heuristic, current vertex, path)
        
        # Costs to reach each vertex
        g_score = {vertex: float('inf') for vertex in range(self.num_vertices)}
        g_score[start] = 0
        
        while open_set:
            _, current, path = heapq.heappop(open_set)
            
            if current == end:
                return path
            
            neighbors = np.where(self.adjacency_matrix[current] > 0)[0]
            for neighbor in neighbors:
                tentative_g_score = g_score[current] + 1  # Each edge has a weight of 1
                
                if tentative_g_score < g_score[neighbor]:
                    g_score[neighbor] = tentative_g_score
                    f_score = tentative_g_score + heuristic(neighbor, end)
                    heapq.heappush(open_set, (f_score, neighbor, path + [neighbor]))
        
        return []  # Return empty list if no path is found

    def dilate_boundary(self, boundary_pts, dilation_factor):
        # Simple dilation by adding immediate neighbors
        dilated_boundary = set(boundary_pts)
        for _ in range(dilation_factor):
            current_boundary = list(dilated_boundary)
            for pt in current_boundary:
                dilated_boundary.update(np.where(self.adjacency_matrix[pt] == 1)[0])
        return list(dilated_boundary)

    def edit_selection(self, boundary_pts, handle_pt):
        queue = [handle_pt]
        editable_vertices = set()
        while len(queue) != 0:
            temp = queue.pop(0)
            editable_vertices.add(temp)
            if temp not in boundary_pts:
                neighbors = np.where(self.adjacency_matrix[temp] == 1)[0].tolist()
                for nodeid in neighbors:
                    if nodeid not in editable_vertices:
                        queue.append(nodeid)
        return list(editable_vertices)


def find_neighbors(adjacency_matrix, node_index):
    return np.where(adjacency_matrix[node_index] == 1)[0].tolist()

def display_inlier_outlier(point_cloud, indices):
    inliers = point_cloud.select_by_index(indices)
    outliers = point_cloud.select_by_index(indices, invert=True)
    
    # Coloring
    inliers.paint_uniform_color([1, 0, 0])  # Red inliers
    outliers.paint_uniform_color([0, 1, 0])  # Green outliers
    
    # Visualizing
    o3d.visualization.draw_geometries([inliers, outliers])
