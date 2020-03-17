'''
    This module defines the geometric primitives and operations necessary
    for Ray casting.

'''
import numpy as np

class Vec:
    ''' Provides a base class of vector with cartesian coordinates
        and common operations implemented.
    '''
    def __init__(self, x, y, z):
        '''Stores three floating-point coordinates in a python list
            belonging to the vector instance.
        '''
        self.coords = [x, y, z]

    def __add__(self, other):
        return Vec(self[0] + other[0], self[1] + other[1], self[2] + other[2])

    def __sub__(self, other):
        return Vec(self[0] - other[0], self[1] - other[1], self[2] - other[2])

    def __mul__(self, constant):
        return Vec(*[constant*coord for coord in self.coords])

    def __truediv__(self, constant):
        return Vec(*[coord/constant for coord in self.coords])
    
    def dot(self, other):
        ''' Implements the dot product '''
        return self[0]*other[0] + self[1]*other[1] + self[2]*other[2]

    def cross(self, other):
        ''' Implements the cartesian cross product '''
        x = self[1]*other[2] - self[2]*other[1]
        y = self[2]*other[0] - self[0]*other[2]
        z = self[0]*other[1] - self[1]*other[0]
        return Vec(x,y,z)

    def __getitem__(self, key):
        ''' Provides easy access to vector coordinates '''
        if key == "x" or key == 0:
            return self.coords[0]

        elif key == "y" or key == 1:
            return self.coords[1]

        elif key == "z" or key == 2:
            return self.coords[2]

    def __str__(self):
        return f"<class Vec, coords: {self.coords}>"

    def mag(self):
        '''Returns magnitude of the vector '''
        return (self.dot(self))**0.5

    def unit(self):
        '''Returns the unit vector sharing direction with this vector'''
        mag = self.mag()
        coords = [coord/mag for coord in self.coords]
        return Vec(*coords) # the * unpacks coords list as the parameters

class NullVec(Vec):
    '''Defines the "null" vector as a vector with zero length and no
        associated direction. The purpose of this class is to provide
        graceful behavior when operating on zero vectors.
    '''
    def __init__(self):
        self.coords = None

    def __add__(self, other):
        return other

    def __sub__(self, other):
        return Vec(-other[0], -other[1], -other[2])
        
    def __mult__(self, other):
        return NullVec()
    
    def dot(self, other):
        ''' Returns null vector'''
        return NullVec()

    def cross(self, other):
        ''' Returns null vector'''
        return NullVec()

    def __getitem__(self, key):
        ''' Provides easy access to vector coordinates '''
        return 0

    def __str__(self):
        return f"<class NullVec, coords: [0, 0, 0]>"

    def mag(self):
        '''Returns magnitude of the vector '''
        return 0

    def unit(self):
        '''Returns null vector'''
        return NullVec()

    
class Plane:
    '''Class definition for an infinite plane descrived by its normal
        vector and any point in the plane.

    '''
    def __init__(self, norm, point):
        ''' Expects the norm and point in the plane to be given as Vec
            objects.
        '''
        self.norm = norm.unit()
        self.D = self.norm.dot(point)

    def __str__(self):
        return f"<class Plane, {self.norm.coords[0]:.3}x + {self.norm.coords[1]:.3}y + {self.norm.coords[2]:.3}z - {self.D:.3} = 0>"

    def Intersect(self, ray):
        '''Calculates the distance and point of intersect of a Ray with
            a plane.

            Returns a tuple: (isIntersect, distance, Vec)
        '''
        denominator = self.norm.dot(ray.direction)
        isIntersect = bool(denominator)
        if isIntersect:
            distance = (self.norm.dot(ray.origin)+self.D)/denominator
            return isIntersect, distance, (ray.origin + ray.direction*distance)
        else:
            return isIntersect, 0, NullVec()

    def Reflection(self, direction):
        ''' Calculates the reflected direction using the plane's normal vector.

            Expects the direction as unit length Vec object.
        '''
        norm = self.norm
        # check direction vector approaches the plane
        return direction - norm*(2*norm.dot(direction))


class Tri:
    '''Class definition for triangle composed of three (3) vertices
        stored as a list of vectors.
    '''
    def __init__(self, vertices):
        ''' Expects vertices list containing the points in counter-clockwise
            order as viewed from the "outside" direction (so that the normal also
            points in this direction).
        '''
        self.vertices = vertices

    def __str__(self):
        return f"<class Tri, Vertices:\n\t{self.vertices[0]},\n\t{self.vertices[1]},\n\t{self.vertices[2]}>"

    def norm(self):
        edge1 = self.vertices[1]-self.vertices[0]
        edge2 = self.vertices[2]-self.vertices[0]
        return edge1.cross(edge2).unit()

    def Intersect(self, ray):
        ''' Calculates the distance and point of intersect of a Ray with
            a triangle. First the intersection with the plane is calculated,
            then the barycentric coordinates are calculated to check if the
            point falls within the triangle.

            Returns a tuple: (isIntersect, distance, Vec)

            Note: A future iteration could explore the Möller-Trumbore or 
            other algorithms for triangle intersection.
        '''
        origin = self.vertices[0] 
        #Calculate edge vectors
        edge1 = self.vertices[1]-self.vertices[0]
        edge2 = self.vertices[2]-self.vertices[0]

        #find intersection point using linear algebra
        tempMat = np.array([edge1.coords, edge2.coords, ray.direction.coords]).T
        tempVec = np.array((origin-ray.origin).coords)
        try:
            tempVec = np.linalg.inv(tempMat).dot(tempVec)
            distance = tempVec[2]
            beta = tempVec[0]
            gamma = tempVec[1]
            isIntersect = gamma + beta < 1
            return isIntersect, distance, (ray.origin + ray.direction*distance)
        except np.linalg.LinAlgError as err:
            if 'Singular matrix' in str(err):
                return False, 0, NullVec()

    def Reflection(self, direction):
        ''' Calculates the reflected direction using the plane's normal vector.

            Expects the direction as unit length Vec object.
        '''
        norm = self.norm()
        # vector and norm must always point in opposite directions
        return direction - norm*(2*norm.dot(direction))