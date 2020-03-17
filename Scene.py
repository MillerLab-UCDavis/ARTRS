'''
    This module defines basic interactions with the scene and acts
    as an interface with blender.

    It can not be run as a standalone script; proper use involves
    the blender open-source 3D modeling program.
    Syntax is:
        blender -b -p Scene.py
    
    or with scene data:
        blender myscene.blend -b -p Scene.py
'''

# import bpy

# print(f"Objects: {bpy.data.objects}")

# for mesh in bpy.data.meshes:
#     print(f"Mesh: {mesh}")
#     numVertices = len(mesh.vertices)
#     for index in range(numVertices):
#         print(f"Vertex {index}: {list(mesh.vertices[index].co)}")

import numpy as np
import scipy.io.wavfile as wavfile
import Geometry

PROP_SPEED = 343.0 #speed of sound (m/s) at room temp and 1atm

class Scene:
    def __init__(self, fileName = "output.wav", sources = None, tris = None, receiver = None):
        self.fileName = fileName #output wav file
        self.sources = []
        self.tris = []
        self.receivers = []
        self.sampRate = 0

    def addSource(self, source):
        self.sources.append(source)
        self.sampRate = max(self.sampRate, source.sampRate)

    def addSurface(self, surface):
        self.tris.append(surface)

    def addSurfaces(self, surfaces):
        for surface in surfaces:
            self.tris.append(surface)

    def addReceiver(self, receiver):
        self.receivers.append(receiver)

    def Trace(self, numRaysAzimuth=50, numRaysPolar=50):
        data = [] #list to store each "channel's" output data
        channels = 0
        raysList = []
        maxSampLength = 0
        #Calculate rays to trace
        for polIndex in np.arange(numRaysPolar):
            for azIndex in np.arange(numRaysAzimuth):
                polarAngle = polIndex * np.pi / numRaysPolar
                azimuthAngle = azIndex * np.pi / numRaysAzimuth
                cylCoord = np.sin(azimuthAngle)
                raysList.append(Geometry.Vec(cylCoord*np.cos(polarAngle),
                                cylCoord*np.sin(polarAngle),
                                np.cos(azimuthAngle)))
        #Calculate the signal as heard from each receiver
        for receiver in self.receivers:
            data.append(np.array([0], dtype="int16"))

            for direction in raysList:
                ray = Ray(direction, origin=receiver.location)
                rayData = ray.Trace(self)
                data[channels] = addSources(data[channels], rayData)

            # print(f"Max received amplitude: {np.max(data[channels])}")
            maxSampLength = max(maxSampLength, len(data[channels]))
            channels += 1

        for index, channel in enumerate(data):
            data[index] = np.pad(channel, pad_width=[0, maxSampLength-len(channel)])
        data = np.array(data, dtype="int16")
        return data

    def Save(self, data):
        # print("Data type: ", data.dtype)
        wavfile.write(self.fileName, self.sampRate, data)
        return True

class Source:
    '''Class definition of a sound source modeled as a small sphere from
        which the signal originates.

    '''
    def __init__(self, location, fileName = "click.wav"):
        '''Expects a fileName in the local folder, and a location stored
            as a vector.
        '''
        self.fileName = fileName
        self.location = location
        #radius may eventually be calculated from strength of the signal
        self.radius = 0.05 # set to 5cm for now
        data = wavfile.read(fileName, "rb")
        self.sampRate = data[0]
        self.signal = data[1][:,0] #using only left channel for now
        # print(f"Max source amplitude: {np.max(self.signal)}")
        # print(f"Type: {self.signal.dtype}")

    def Delay(self, seconds):
        sampDelay = int(round(self.sampRate*seconds))
        self.signal = np.pad(self.signal, pad_width=[sampDelay,0])
        return self

    def Intersect(self, ray):
        '''Calculates the distance and point of intersect of a Ray with
            a sound source modeled as a small sphere.

            Returns a tuple: (isIntersect, distance, Vec)
        '''
        #use a quadratic equation to determine point of intersection
        tempVec = ray.origin - self.location
        quadA = ray.direction.dot(ray.direction)
        quadB = ray.direction.dot(tempVec)
        quadC = tempVec.dot(tempVec)-self.radius**2
        discriminant = quadB**2-quadA*quadC
        if discriminant > 0:
            discriminant = discriminant**0.5
            #smallest solution
            distance = (-quadB-discriminant)/quadA
            if distance < 0:
                #greatest solution
                distance = (-quadB+discriminant)/quadA
            return True, distance, (ray.origin + ray.direction*distance)
        elif discriminant == 0:
            distance = -quadB/quadA
            return True, distance, (ray.origin + ray.direction*distance)
        else:
            return False, 0, Geometry.NullVec()

class Receiver:
    '''Class definition of a sound source modeled as a small sphere from
        which the signal originates.

    '''
    def __init__(self, location, name):
        '''Expects a file name for the signal received, and a location for
            the receiver.
        '''
        self.name = name
        self.location = location
        self.sounds = []




class Ray:
    ''' A class which describes rays intended to be cast from the receiver
        to a source. Rays have an origin and a direction towards which they
        will propagate. 
    '''
    def __init__(self, direction, origin = None, distance = 0):
        '''Expects two vectors: one to describe the origin (or tail)
            pf the vector, and another to describe the direction of 
            the head for the vector.

            Omitting the origin parameter has the behavior of declaring
            a ray which begins at the origin of the coordinate system.
        '''
        self.direction = direction.unit()
        self.origin = Geometry.NullVec() if origin is None else origin
        self.distance = distance

    def Trace(self, Scene):
        ''' Calculates the sound data for this ray finding intersections
            with each triangle in Scene. 
        '''
        # print(f"{self.distance}")

        nearDistance = float("inf")
        nearIntersect = Geometry.NullVec()
        nearThing = None
        hasReflection = False
        rayData = np.array([0], dtype="int16") #empty data condition
        # sourceDirections = None

        #Direct path to sources
        for source in Scene.sources:
            (isIntersect, srcDist, srcIntersect) = source.Intersect(self)
            del srcIntersect #For now I'm ignoring the intersection point
            if isIntersect:
                # print(self)
                delayTime = srcDist / PROP_SPEED
                delaySamples = int(round(delayTime*source.sampRate))
                # print(f"Delay = {delaySamples}, {type(delaySamples)}")
                srcData = np.pad(source.signal, pad_width=[delaySamples,0])
                rayData = addSources(rayData, srcData)


        #Intersect with all objects in the scene
        for thing in Scene.tris:
            (isIntersect, distance, intersection) = thing.Intersect(self)
            if isIntersect:
                #This is an attempt to reduce branching for optimization purposes
                ##not sure if it actually helps.
                updateDistance = int(nearDistance > distance and distance > 0)
                nearDistance = [nearDistance, distance][updateDistance]
                nearIntersect = [nearIntersect, intersection][updateDistance]
                nearThing = [nearThing, thing][updateDistance]
                
        #a recursive call to Trace should be performed
        hasReflection = nearDistance + self.distance < 120 #TODO: Find new calculation for terminating the recursion
        if hasReflection:
            direction = nearThing.Reflection(self.direction)
            reflectedRay = Ray(direction, origin=nearIntersect, distance=self.distance+nearDistance)
            rayData = addSources(rayData, reflectedRay.Trace(Scene))
        
        return rayData
            
    def __str__(self):
        return f"<class Ray, Origin: {self.origin},\tDirection: {self.direction},\tDistance:{self.distance}>"


def addSources(data1, data2):
    length1 = len(data1)
    length2 = len(data2)
    if length1 < length2:
        data1 = np.pad(data1, pad_width=[0,length2-length1])
    elif length2 < length1:
        data2 = np.pad(data2, pad_width=[0,length1-length2])
    return data1+data2

