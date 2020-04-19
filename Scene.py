'''
    This module provides a low-level programmatic interface
    for defining and ray tracing scenes based on geometric
    primitives. 
    
    The module is designed to minimized the operations
    in each function so the user can decide when actions
    need to be taken. As such, declaring 
'''

import numpy as np
import scipy.io.wavfile as wavfile
import Geometry
#import numba
import multiprocessing as mp
from itertools import product

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

    def Trace(self, numRaysAzimuth=128, numRaysPolar=128, duration=5):
        '''Calculates the signal received by each receiver in the scene.
            
            The numRaysAzimuthal determines the number of divisions for
            the 2pi radians around the z-axis (with zero at the x-axis),
            while numRaysPolar defines the divisions in the pi radians
            away from the z-axis. Duration defines the total length in
            seconds desired from the output
        '''
        numChannels = len(self.receivers)
        maxSampLength = self.sampRate*duration
        data = np.zeros((numChannels, maxSampLength), dtype='float32') # initialize space in memory
        directions = []
        #Calculate all directions to trace
        for polIndex in np.arange(numRaysPolar):
            for azIndex in np.arange(numRaysAzimuth):
                polarAngle = polIndex * np.pi / numRaysPolar
                azimuthAngle = azIndex * 2*np.pi / numRaysAzimuth
                cylCoord = np.sin(polarAngle)
                directions.append(Geometry.Vec(cylCoord*np.cos(azimuthAngle),
                                cylCoord*np.sin(azimuthAngle),
                                np.cos(polarAngle)))

        #Calculate the signal as heard from each receiver
        for channel, receiver in enumerate(self.receivers):
            print(f"Tracing channel {channel}...")
            with mp.Pool() as pool:
                results = pool.starmap(traceDirection,product(directions,[self],[receiver], [maxSampLength]))

            for result in results:
                data[channel][:len(result)] += result
                del result
            del results

            data[channel] /= np.max(data[channel])
            print(f"Finished channel {channel}.")


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
        self.signal = data[1][:,0].astype("float32") #using only left channel for now
        self.signal /= np.max(self.signal)

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
            #use smallest solution by default
            distance = (-quadB-discriminant)/quadA
            if distance < 0:
                #use greatest solution when smallest is negative
                distance = (-quadB+discriminant)/quadA
            return True, distance, (ray.origin + ray.direction*distance)
        elif discriminant == 0:
            distance = -quadB/quadA
            return True, distance, (ray.origin + ray.direction*distance)
        else:
            return False, 0, Geometry.NullVec()

class Receiver:
    '''Class definition of a sound receiver modeled as a point
        which a signal can be detected, and representing a single
        channel in the output of a Scene.

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
            the head for the vector. Distance parameter indicates the
            distanced traveled before reaching the ray origin.

            Omitting the origin parameter declares a ray which begins
            at the origin of the coordinate system.
        '''
        self.direction = direction.unit()
        self.origin = Geometry.NullVec() if origin is None else origin
        self.distance = distance

    def Trace(self, Scene, numSamples = None, rayData=None, attenuation=1.0):
        ''' Calculates the sound data for this ray finding intersections
            with each triangle in Scene. 
        '''
        numSamples = 1 if numSamples is None else numSamples
        rayData = np.zeros((numSamples), dtype="float32") if rayData is None else rayData
        nearDistance = float("inf") #default nearest distance
        nearIntersect = Geometry.NullVec()
        nearThing = None
        hasReflection = False

        #Direct path to sources
        for source in Scene.sources:
            (isIntersect, srcDist, srcIntersect) = source.Intersect(self)
            del srcIntersect #For now I'm ignoring the intersection point
            if isIntersect and srcDist > 0:
                delayTime = srcDist / PROP_SPEED
                delaySamples = int(round(delayTime*source.sampRate))
                srcLen = len(source.signal)
                rayData[delaySamples:min(delaySamples+srcLen,numSamples)] += source.signal[:numSamples-delaySamples]


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
            reflectedRay.Trace(Scene, numSamples=numSamples, rayData=rayData, attenuation=0.6)
            del reflectedRay
        
        #enforce reflection attenuation
        if attenuation != 1.0:
            rayData *= attenuation
            
        return rayData
            
    def __str__(self):
        return f"<class Ray, Origin: {self.origin},\tDirection: {self.direction},\tDistance:{self.distance}>"


def traceDirection(direction, scene, receiver, numSamples):
    return Ray(direction, origin=receiver.location).Trace(scene, numSamples=numSamples)