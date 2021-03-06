'''
    This module provides a low-level programmatic interface
    for defining and ray tracing scenes based on geometric
    primitives. 
'''

import numpy as np
import soundfile as sf

import multiprocessing as mp
from itertools import product
import Geometry

PROP_SPEED = 343.0 #speed of sound (m/s) at room temp and 1atm
REFL_COEFF = 1-0.03 #estimated for 500Hz using random-incidence absorption
MAX_PATH_LEN = 50 #spreading losses amount to ~60dB after 
ATM_ATTEN = 0.6e-3 #atmospheric attenuation at roughly room temp and 50% humidity
EULER = np.exp(1)


class Scene:
    def __init__(self, fileName = "output.wav", sources = None, tris = None, receiver = None):
        self.fileName = fileName #output wav file
        self.sources = []
        self.tris = []
        self.receivers = []
        self.sampRate = 0

    def addSource(self, source):
        self.sources.append(source)
        self.sampRate = max(self.sampRate, source.sampRate) #TODO: handle resampling for mismatched sources

    def addSources(self, sources):
        for source in sources:
            self.sources.append(source)
            self.sampRate = max(self.sampRate, source.sampRate)

    def addSurface(self, surface):
        self.tris.append(surface)

    def addSurfaces(self, surfaces):
        for surface in surfaces:
            self.tris.append(surface)

    def addReceiver(self, receiver):
        self.receivers.append(receiver)

    def addReceivers(self, receivers):
        for receiver in receivers:
            self.receivers.append(receiver)

    def clear(self):
        '''Clears sources and receivers for scene reuse'''
        self.sources = []
        self.receivers = []

    def Trace(self, numRaysAzimuth=128, numRaysPolar=128, duration=5, memChunk=int(5e9)):
        '''Calculates the signal received by each receiver in the scene.
            
            The numRaysAzimuthal determines the number of divisions for
            the 2pi radians around the z-axis (with zero at the x-axis),
            while numRaysPolar defines the divisions in the pi radians
            away from the z-axis.
            
            Duration defines the total length in seconds desired for
            the output data.

            memChunk roughly represents the number of bytes of data that
            are calculated before the results of each ray traced are added
            together, and helps to divide the work so that the size of the
            working set doesn't exceed the total system memory. The 
            default size of memory chunks is 5GB.
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
        totalRays = len(directions)
        
        #Calculate the signal as heard from each receiver
        for channel, receiver in enumerate(self.receivers):
            print(f"Tracing channel {channel}...")
            with mp.Pool() as pool:
                workers = []
                args = list(product(directions,[self],[receiver], [maxSampLength]))
                totalBytes = 4*totalRays*maxSampLength
                numChunks = totalBytes//memChunk + 1
                chunkSize = totalRays//(numChunks-1) #true size
                #split into multiple asynchronous calls so that they get processed along the way
                print(f"Queueing {numChunks} chunks")
                for chunk in range(numChunks):
                    workers.append(pool.starmap_async(traceDirection, args[chunk*chunkSize:(chunk+1)*chunkSize]))

                raysDone = 0
                print("Progress:\t[--------------------]\t0%", end="\r", flush=True)
                for i in range(numChunks):
                    chunk = workers.pop(0).get()
                    for rayData in chunk:
                        data[channel][:len(rayData)] += rayData
                        del rayData
                        raysDone += 1
                        progress = "Progress:\t["
                        percent = raysDone*100/totalRays
                        barLen = int(percent//5)
                        progress += "#"*barLen
                        progress += "-"*(20-barLen)
                        progress +=f"]  {raysDone}/{totalRays} rays ({percent:.1f}%) chunk {i+1} out of {numChunks}"
                        print(progress, end="\r", flush=True)
                    del chunk

            data[channel] /= np.max(data[channel])
            print(f"\nFinished channel {channel}.")


        return data

    def Save(self, data):
        sf.write(self.fileName, data.T, self.sampRate)
        return True

    def __str__(self):
        description = f"<class Scene, file name: {self.fileName},\tsample rate: >\n"
        description += "<Sources>\n"
        for source in self.sources:
            description += f"\t{source}\n"
        description += "</Sources>\n"
        description += "<Receivers>\n"
        for receiver in self.receivers:
            description += f"\t{receiver}\n"
        description += "</Receivers>\n"
        description += "<Tris>\n"
        for tri in self.tris:
            description += f"{tri}\n"
        description += "</Tris>\n"
        description += "</Scene>"
        return description

class Source:
    '''Class definition of a sound source modeled as a small sphere from
        which the signal originates.

    '''
    def __init__(self, location = Geometry.NullVec(), name = "click.wav", data=False):
        '''Expects a fileName in the local folder, and a location stored
            as a vector. Alternatively, the data can be passed in as a
            (signal, sampleRate) tuple. 
        '''
        #radius may eventually be calculated from strength of the signal
        self.radius = 0.05 # set to 5cm for now
        self.name = name
        self.location = location
        if not data:
            #check for file matching name
            try:
                data, self.sampRate = sf.read(name)
                self.signal = data.T[0].astype("float32") #using only left channel for now
                self.signal /= np.max(self.signal)
            except:
                print(f"No file matching '{name}' found; source data will be left uninitialized")
                self.sampRate = None
                self.signal = None
        else:
            #Initialize with data tuple
            self.signal, self.sampRate = data
            self.signal = self.signal.astype("float32")
            self.signal /= np.max(self.signal)

    def __str__(self):
        return f"<class Source, Name: {self.name},\tLocation: {self.location},\tSample rate: {self.sampRate}>"

    def Save(self, directory=""):
        name = self.name if self.name[-4:] == ".wav" else self.name+".wav"
        sf.write(directory+name, self.signal, self.sampRate)
        return True

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

    def __str__(self):
        return f"<class Receiver, Name: {self.name}, Location: {self.location}>"


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

    def __str__(self):
        return f"<class Ray, Origin: {self.origin},\tDirection: {self.direction},\tDistance:{self.distance}>"

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
            if isIntersect and srcDist > 0 and srcDist + self.distance < MAX_PATH_LEN :
                totDist = srcDist + self.distance
                delayTime = srcDist / PROP_SPEED
                delaySamples = int(round(delayTime*source.sampRate))
                srcLen = len(source.signal)
                signal = ((EULER**(-ATM_ATTEN*totDist))**0.5)*source.signal[:numSamples-delaySamples]
                rayData[delaySamples:min(delaySamples+srcLen,numSamples)] += signal
                del signal


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
        hasReflection = nearDistance + self.distance < MAX_PATH_LEN #TODO: Find better calculation for terminating the recursion
        if hasReflection:
            direction = nearThing.Reflection(self.direction)
            totDist = nearDistance + self.distance
            reflectedRay = Ray(direction, origin=nearIntersect, distance=self.distance+nearDistance)
            totAtten = REFL_COEFF*((EULER**(-ATM_ATTEN*totDist))**0.5)
            reflectedRay.Trace(Scene, numSamples=numSamples, rayData=rayData, attenuation=totAtten)
            del reflectedRay
        
        #enforce reflection attenuation
        if attenuation != 1.0:
            rayData *= attenuation
            
        return rayData
            

class RectRoom(Scene):
    def __init__(self, width, length, height, fileName = "output.wav"):
        '''The most basic, pre-defined Scene representing a closed
            rectangular room. Width defines the x-dimension, length
            defines the y-dimension, and height defines the z-dimension.
            The room is defined within the positive cartesian quadrant.
        '''
        self.width = width
        self.length = length
        self.height = height
        #Define corners of room
        vert1 = Geometry.NullVec()
        vert2 = Geometry.Vec(0, 0, height)
        vert3 = Geometry.Vec(width, 0, height)
        vert4 = Geometry.Vec(width, 0, 0)
        vert5 = Geometry.Vec(0, length, 0)
        vert6 = Geometry.Vec(0, length, height)
        vert7 = Geometry.Vec(width, length, height)
        vert8 = Geometry.Vec(width, length, 0)
        #Call super class to initialize member variables and functions
        Scene.__init__(self, fileName=fileName)
        #Close wall
        self.addSurfaces([Geometry.Tri([vert1, vert2, vert4]),
                        Geometry.Tri([vert2, vert3, vert4])])
        #Farthest wall
        self.addSurfaces([Geometry.Tri([vert5, vert6, vert8]),
                        Geometry.Tri([vert6, vert7, vert8])])
        #Left wall
        self.addSurfaces([Geometry.Tri([vert1, vert2, vert5]),
                        Geometry.Tri([vert2, vert6, vert5])])
        #Right wall
        self.addSurfaces([Geometry.Tri([vert4, vert3, vert8]),
                        Geometry.Tri([vert3, vert7, vert8])])
        #Floor
        self.addSurfaces([Geometry.Tri([vert1, vert5, vert4]),
                        Geometry.Tri([vert5, vert8, vert4])])
        #Roof
        self.addSurfaces([Geometry.Tri([vert2, vert6, vert3]),
                        Geometry.Tri([vert6, vert7, vert3])])

    def createPositions(self, numPositions, padding=0.25):
        '''Returns a number of vectors with random locations within
            the Scene (with some padding in meters from each wall).
        '''
        coords = np.random.random_sample(size=(numPositions, 3))
        coords[:,0] *= self.width-2*padding
        coords[:,0] += padding
        coords[:,1] *= self.length-2*padding
        coords[:,1] += padding
        #height coordinate decided according to human height distribution
        coords[:,2] = [int(value) for value in coords[:,2]]
        coords[:,2] = [[np.random.normal(1.63, 0.07), np.random.normal(1.75, 0.075)][int(value)] for value in coords[:,2]]
        return [Geometry.Vec(*row) for row in coords]

def traceDirection(direction, scene, receiver, numSamples):
    '''Helper function which can handle the parallel execution of rays'''
    return Ray(direction, origin=receiver.location).Trace(scene, numSamples=numSamples)
