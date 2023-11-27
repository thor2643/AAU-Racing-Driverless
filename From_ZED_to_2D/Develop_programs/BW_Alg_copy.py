# BowyerWatson algorithm for Delaunay triangulation

import math

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

class Triangle:
    def __init__(self, p1, p2, p3):
        self.vertices = [p1, p2, p3]

    def calculate_circumcircle_center(self):
        # Calculate the circumcircle center and radius for the triangle
        p1, p2, p3 = self.vertices
        d = 2 * (p1.x * (p2.y - p3.y) + p2.x * (p3.y - p1.y) + p3.x * (p1.y - p2.y))
        
        if abs(d) < 1e-10:  # Add a small epsilon value to avoid division by zero
            return None  # You can return None or some other appropriate value in this case
        
        ux = ((p1.x ** 2 + p1.y ** 2) * (p2.y - p3.y) + (p2.x ** 2 + p2.y ** 2) * (p3.y - p1.y) + (p3.x ** 2 + p3.y ** 2) * (p1.y - p2.y)) / d
        uy = ((p1.x ** 2 + p1.y ** 2) * (p3.x - p2.x) + (p2.x ** 2 + p2.y ** 2) * (p1.x - p3.x) + (p3.x ** 2 + p3.y ** 2) * (p2.x - p1.x)) / d
        return Point(ux, uy)

    def in_circumcircle(self, p):
        # Check if a point p is inside the circumcircle of the triangle
        center = self.calculate_circumcircle_center()
        radius = math.sqrt((p.x - center.x) ** 2 + (p.y - center.y) ** 2)
        d = math.sqrt((self.vertices[0].x - center.x) ** 2 + (self.vertices[0].y - center.y) ** 2)
        return radius < d

def bowyer_watson(pointList):
    # Create a super-triangle that contains all points
    max_val = max(max(p.x, p.y) for p in pointList)
    super_triangle = [Point(3 * max_val, 0), Point(0, 3 * max_val), Point(-3 * max_val, -3 * max_val)]
    triangulation = [Triangle(*super_triangle)]

    for point in pointList:
        bad_triangles = []

        # Find all triangles that are no longer valid due to the insertion
        for triangle in triangulation:
            if triangle.in_circumcircle(point):
                bad_triangles.append(triangle)

        polygon = []

        # Find the boundary of the polygonal hole
        for triangle in bad_triangles:
            for edge in triangle.vertices:
                shared = False
                for other_triangle in bad_triangles:
                    if other_triangle != triangle and edge in other_triangle.vertices:
                        shared = True
                        break
                if not shared:
                    polygon.append(edge)

        # Remove the bad triangles from the triangulation
        for triangle in bad_triangles:
            triangulation.remove(triangle)

        # Retriangulate the polygonal hole
        for edge in polygon:
            new_tri = Triangle(edge, point, edge)
            triangulation.append(new_tri)

    # Remove triangles that contain any vertex from the original super-triangle
    triangulation = [tri for tri in triangulation if not any(v in super_triangle for v in tri.vertices)]

    return triangulation
