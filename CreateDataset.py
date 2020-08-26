import tensorflow as tf
import pandas as pd
import numpy as np
import soundfile as sf
import os, sys, glob, random

import Geometry as Geo
import Scene

MEMCHUNK = int(os.getenv("MEMCHUNK", "5000000000"))

def createReceivers(steering, origin, numRecs = 8, spacing = 0.0186):
    '''Creates a linear microphone array normal in the xy-plane
        to the given steering vector. Returns a list of receivers
        centered around the origin. Default parameters are based
        on the microphone array used in:

        M. H. Anderson et al., “Towards mobile gaze-directed
        beamforming: a novel neuro-technology for hearing loss,”
        in 2018 40th Annual International Conference of the IEEE
        Engineering in Medicine and Biology Society (EMBC), Jul.
        2018, pp. 5806–5809, doi: 10.1109/EMBC.2018.8513566.
    '''
    direction = Geo.Vec(steering["y"], -steering["x"], 0).unit()
    arrayLength = direction*(numRecs*spacing*0.5)
    receivers = [Scene.Receiver(origin+direction*index-arrayLength, name=f"channel {index}") for index in range(numRecs)]
    return receivers


def createDataset(corpus, scene, numMixtures, numSpeakers, duration = 60, numRecs = 8, spacing = 0.0186, resolution=(1024, 512)):
    '''Creates a mixed dataset from an isolated speech corpus by
        randomly placing the sources in the scene with a randomly
        placed linear microphone array and then ray-tracing the scene.
        The mixed data will be stored in the TensorFlow TFRecords
        format in a folder with the same name as the corpus, but with
        "-mix" appended to it.

        numMixtures refers to the number of mixes to create, while 
        numSpeakers refers to the number of unique speakers in each
        mix.
    '''
    outPath = corpus.directory+"-mix"
    sceneName = scene.fileName + f"-N{numSpeakers}-D{duration}"
    if not os.path.exists(outPath):
        os.makedirs(outPath)
    #check for existing mixture data
    numExisting = len(glob.glob(outPath+f"/{sceneName}*"))

    for index in range(numMixtures):
        print(f"\nStarting mixture {index+1} of {numMixtures}")
        #Add speakers to the scene
        speakers = corpus.getSpeakers(numSpeakers)
        locations = scene.createPositions(numSpeakers)
        sources = corpus.getSources(speakers, locations, duration)
        scene.addSources(sources)
        #Add microphones to the scene
        micOrigin = scene.createPositions(1, padding=numRecs*spacing*0.5)[0]
        steerAngles = [srcDirection-micOrigin for srcDirection in locations]
        micArray = createReceivers(steerAngles[0], micOrigin, numRecs=numRecs, spacing=spacing)
        scene.addReceivers(micArray)
        #Label and Trace the scene
        scene.fileName = f"{outPath}/{sceneName}--{index+numExisting}.wav"
        traceData = scene.Trace(numRaysAzimuth=resolution[0], numRaysPolar=resolution[1], duration=duration, memChunk=MEMCHUNK)
        serializeTrace = tf.io.serialize_tensor(traceData)
        scene.Save(traceData)
        #Save as TFRecords file
        exampleMessage = {
		    "Scene": tf.train.Feature(bytes_list=tf.train.BytesList(value=[bytes(scene.fileName, "utf8")])),
            "sampleRate": tf.train.Feature(int64_list=tf.train.Int64List(value=[scene.sampRate])),
            "numRecs": tf.train.Feature(int64_list=tf.train.Int64List(value=[numRecs])),
            "micArrOrigin": tf.train.Feature(float_list=tf.train.FloatList(value=micOrigin.coords)),
            "micSteer":  tf.train.Feature(float_list=tf.train.FloatList(value=steerAngles[0].coords))
        }
        for spkNum, person in enumerate(sources):
            exampleMessage["speaker"+str(spkNum+1)] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[bytes(person.name, "utf8")]))
            exampleMessage["location"+str(spkNum+1)] = tf.train.Feature(float_list=tf.train.FloatList(value=person.signal))
            exampleMessage["signal"+str(spkNum+1)] = tf.train.Feature(float_list=tf.train.FloatList(value=person.location.coords))
            person.Save(f"{outPath}/")
        exampleMessage["traceData"] = tf.train.Feature(int64_list=tf.train.Int64List(value=serializeTrace.numpy()))
        exampleObj = tf.train.Example(features=tf.train.Features(feature=exampleMessage))
        with tf.io.TFRecordWriter(scene.fileName[:-4]+".tfrecord") as writer:
            writer.write(exampleObj.SerializeToString())

        scene.clear()



class LibriSpeech:
    parent = "../Dataset/LibriSpeech/"
    def __init__(self, directory):
        self.directory = LibriSpeech.parent + directory

        self.speakers = pd.read_table(f"{LibriSpeech.parent}SPEAKERS.TXT", sep="|",header=0, comment=";")
        self.speakers.rename(columns=lambda x: x.strip(), inplace=True)
        self.speakers["SUBSET"] = self.speakers["SUBSET"].str.strip()
        self.speakers = self.speakers[self.speakers["SUBSET"] == directory]
        self.speakers.reset_index(inplace=True, drop=True)

        self.consumed = pd.DataFrame(columns=self.speakers.columns)

    def getSpeakers(self, numSpeak):
        try:
            subSamp = random.sample(range(len(self.speakers)), numSpeak)
        except ValueError:
            print("Out of unique speakers; resetting corpus.")
            self.reset()
            subSamp = random.sample(range(len(self.speakers)), numSpeak)
        finally:
            people = self.speakers.iloc[subSamp]
            self.speakers.drop(index=subSamp, inplace=True)
            self.speakers.reset_index(inplace=True, drop=True)
            self.consumed = self.consumed.append(people, ignore_index=True)
            return people

    def getSources(self, people, locations, duration):
        '''Accepts DataFrame of people, and returns a source
            with a random location in the corresponding to each person a.'''
        sources = []
        for index, person in enumerate(people["ID"]):
            fullPath = f"{self.directory}/{person}"
            chapters = os.listdir(fullPath)
            book = random.choice(chapters)
            sentences = [clip.replace("\\","/") for clip in sorted(glob.glob(fullPath+f"/{book}/*.flac"))]
            name = f"s{person}-b{book}-d"

            line = sentences.pop(0)
            currDur = sf.info(line).duration
            data, sampRate = sf.read(line)
            lastNum = "0000"
            for line in sentences:
                if currDur < duration:
                    currDur += sf.info(line).duration
                    data = np.append(data, sf.read(line)[0])
                    lastNum = line[-9:-5] #isolate the line number from filename
                else:
                    break
            sources.append(Scene.Source(location=locations[index], name=name+lastNum, data=(data[:sampRate*duration], sampRate)))
            del data
        return sources
    
    def reset(self):
        self.speakers = self.speakers.append(self.consumed, ignore_index=True)
        self.consumed = pd.DataFrame(columns=self.speakers.columns)

if __name__ == "__main__":
    argc = len(sys.argv)
    smallRoom = Scene.RectRoom(width=3, length=4, height=3, fileName="SmallRoom")
    corpus = LibriSpeech("dev-clean" if argc == 1 else sys.argv[1])
    createDataset(corpus, smallRoom, 60, 3, resolution=(640, 320))
