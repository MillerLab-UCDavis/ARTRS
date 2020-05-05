import Scene
from Scene import Ray
from Geometry import Vec, NullVec, Plane, Tri
import time

origin = NullVec()

xhat = Vec(1,0,0)
yhat = Vec(0,1,0)
zhat = Vec(0,0,1)

# planeX = Plane(xhat, xhat)
# planeY = Plane(yhat, yhat*4)
# planeZ = Plane(zhat, zhat)

# planeXN = Plane(origin-xhat, origin-xhat)
# planeYN = Plane(origin-yhat, origin-yhat)
# planeZN = Plane(origin-zhat, origin-zhat)



# xyVec = Vec(3,4,0)
# xyPlane = Plane(xyVec, xyVec)
# result = xyPlane.Intersect(Ray(xyVec))
# print(f"Tilted plane intersect: {result[2]},\n distance: {result[1]}")

# xyPlane = Plane(origin-xyVec, xyVec)
# result = xyPlane.Intersect(Ray(xyVec))
# print(f"Tilted plane 2 intersect: {result[2]},\n distance: {result[1]}")

# xyVec = Vec(3,0,4)
# xyPlane = Plane(xyVec, origin-xyVec)
# result = xyPlane.Intersect(Ray(xyVec))
# print(f"Tilted plane 3 intersect: {result[2]},\n distance: {result[1]}")

# xyVec = Vec(4,3,0)
# triangle = Tri([Vec(3,0,0), Vec(3,0,12), Vec(0,4,0)])
# result = triangle.Intersect(Ray(xyVec))
# print(f"\nTriangle intersect: {result[2]},\n\n distance: {result[1]}")

# triPlane = Plane(triangle.norm(),triangle.vertices[0])
# result = triPlane.Intersect(Ray(xyVec))
# print(f"\nTri-plane intersect: {result[2]},\n\n distance: {result[1]}")


## Small example room

rectRoom = Scene.Scene(fileName = "rectRoom2.wav")
#ears separated by about 8cm (overestimate)
rectRoom.addReceiver(Scene.Receiver(Vec(-0.08,0,1.75), "left_ear"))
rectRoom.addReceiver(Scene.Receiver(Vec(0.08,0,1.75), "right_ear"))

#click 10 meters straight ahead ~1ft above ground
rectRoom.addSource(Scene.Source(Vec(0,10,0.3), fileName="click.wav"))
#click 12 meters ahead, 1.5 meters left of center, and 2m above ground
rectRoom.addSource(Scene.Source(Vec(-1.5,12,2), fileName="click.wav").Delay(2))
#click 11 meters ahead, 3 meters right of center, 1m above ground
rectRoom.addSource(Scene.Source(Vec(3,11,1), fileName="click.wav").Delay(4))

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
    startTime = time.time()
    traceData = rectRoom.Trace(numRaysAzimuth=152, numRaysPolar=152)
    totalTime = time.time()-startTime
    print(f"Done in {totalTime} seconds.")

    print(traceData.shape)

    rectRoom.Save(traceData.T)



