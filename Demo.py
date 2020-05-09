import Scene
from Geometry import Vec, NullVec, Plane, Tri
import time
import sys

origin = NullVec()

xhat = Vec(1,0,0)
yhat = Vec(0,1,0)
zhat = Vec(0,0,1)

## Small example room

rectRoom = Scene.Scene(fileName = "rectRoom2.wav")
#ears separated by about 8cm (overestimate)
rectRoom.addReceiver(Scene.Receiver(Vec(-0.08,0,1.75), "left_ear"))
rectRoom.addReceiver(Scene.Receiver(Vec(0.08,0,1.75), "right_ear"))

#click 10 meters straight ahead ~1ft above ground
rectRoom.addSource(Scene.Source(Vec(0,10,0.3), name="click.wav"))
#click 12 meters ahead, 1.5 meters left of center, and 2m above ground
rectRoom.addSource(Scene.Source(Vec(-1.5,12,2), name="click.wav").Delay(2))
#click 11 meters ahead, 3 meters right of center, 1m above ground
rectRoom.addSource(Scene.Source(Vec(3,11,1), name="click.wav").Delay(4))

#Left wall
rectRoom.addSurfaces([Tri([Vec(-5,-5,0), Vec(-5,15,0), Vec(-5,-5,3)]),
                    Tri([Vec(-5,15,0), Vec(-5,15,3), Vec(-5,-5,3)])
                    ])
#Right wall
rectRoom.addSurfaces([Tri([Vec(5,-5,0), Vec(5,15,0), Vec(5,-5,3)]),
                    Tri([Vec(5,15,0), Vec(5,15,3), Vec(5,-5,3)])
                    ])
#Front wall
rectRoom.addSurfaces([Tri([Vec(-5,15,0), Vec(-5,15,3), Vec(5,15,0)]),
                    Tri([Vec(-5,15,3), Vec(5,15,3), Vec(5,15,0)])
                    ])
#Back wall
rectRoom.addSurfaces([Tri([Vec(-5,-5,0), Vec(-5,-5,3), Vec(5,-5,0)]),
                    Tri([Vec(-5,-5,3), Vec(5,-5,3), Vec(5,-5,0)])
                    ])
#Roof
rectRoom.addSurfaces([Tri([Vec(-5,-5,3), Vec(-5,15,3), Vec(5,-5,3)]),
                    Tri([Vec(-5,15,3), Vec(5,15,3), Vec(5,-5,3)])
                    ])
#Floor
rectRoom.addSurfaces([Tri([Vec(-5,-5,0), Vec(-5,15,0), Vec(5,-5,0)]),
                    Tri([Vec(-5,15,0), Vec(5,15,0), Vec(5,-5,0)])
                    ])

if __name__ == "__main__":
    argc = len(sys.argv)
    if argc == 1:
        numAzimuth = 512
        numPolar = 512
    elif argc == 2:
        numAzimuth = int(sys.argv[1])
        numPolar = int(sys.argv[1])
    elif argc == 3:
        numAzimuth = int(sys.argv[1])
        numPolar = int(sys.argv[2])
    print(f"Resolution ({numAzimuth},{numPolar})")
    startTime = time.time()
    traceData = rectRoom.Trace(numRaysAzimuth=numAzimuth, numRaysPolar=numPolar)
    totalTime = time.time()-startTime
    print(f"Done in {totalTime} seconds.")

    print(traceData.shape)

    rectRoom.Save(traceData.T)



