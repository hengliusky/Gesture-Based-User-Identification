import numpy as np
import os
import math
#load the subdirectory of source data set as the user ID
def getSubDir(path):
    dirname=list()
    for file in os.listdir(path):
        dirname.append(file)
    return dirname

def gaussianSmooth(gestureBuf):
    gaussianBuf=np.zeros(gestureBuf.shape)
    gaussianBuf[0:3]=gestureBuf[0:3]

    x=np.ones(5)
    f=np.ones(5)
    for i in range(2,len(gestureBuf)-2):
        for j in range(25):
            px=gestureBuf[i-2:i+3,j*3]
            py=gestureBuf[i-2:i+3,j*3+1]
            pz=gestureBuf[i-2:i+3,j*3+2]
            d1 = ((px[0] - px[1]) * (px[0] - px[1]) + (py[0] - py[1]) * (py[0] - py[1]) + (pz[0] - pz[1]) * (pz[0] - pz[1])) ** 0.5
            d2 = ((px[1] - px[2]) * (px[1] - px[2]) + (py[1] - py[2]) * (py[1] - py[2]) + (pz[1] - pz[2]) * (pz[1] - pz[2])) ** 0.5
            d3 = ((px[2] - px[3]) * (px[2] - px[3]) + (py[2] - py[3]) * (py[2] - py[3]) + (pz[2] - pz[3]) * (pz[2] - pz[3])) ** 0.5
            d4 = ((px[3] - px[4]) * (px[3] - px[4]) + (py[3] - py[4]) * (py[3] - py[4]) + (pz[3] - pz[4]) * (pz[3] - pz[4])) ** 0.5
            u=(d1+d2+d3+d4)/4.0
            s=max((d1+d2),(d3+d4))
            for row in range(5):
                x[row]=((px[2]-px[row])*(px[2]-px[row])+(py[2]-py[row])*(py[2]-py[row])+(pz[2]-pz[row])*(pz[2]-pz[row]))**0.5
            for row in range(5):
                buf=-(x[row]-u)*(x[row]-u)/(2*s*s)
                f[row]=(math.e**buf)/((2*3.1415*s)**0.5)
            gaussianBuf[i][j * 3] = (f[0] * px[0] + f[1] * px[1] + f[2] * px[2] + f[3] * px[3] + f[4] * px[4]) / (f[0]+f[1]+f[2]+f[3]+f[4])
            gaussianBuf[i][j * 3 + 1] = (f[0] * py[0] + f[1] * py[1] + f[2] * py[2] + f[3] * py[3] + f[4] * py[4]) / (f[0]+f[1]+f[2]+f[3]+f[4])
            gaussianBuf[i][j * 3 + 2] = (f[0] * pz[0] + f[1] * pz[1] + f[2] * pz[2] + f[3] * pz[3] + f[4] * pz[4]) / (f[0]+f[1]+f[2]+f[3]+f[4])
    gaussianBuf[len(gestureBuf)-2:len(gestureBuf)]=gestureBuf[len(gestureBuf)-2:len(gestureBuf)]
    return gaussianBuf

def jointNorm(gaussianBuf):
    normGesture=np.ones((gaussianBuf.shape[0],75))
    for i in range(gaussianBuf.shape[0]):
        spine_base = gaussianBuf[i][0:3]
        spine_shoulder = gaussianBuf[i][60:63]
        d = ((spine_shoulder[0]-spine_base[0])*(spine_shoulder[0]-spine_base[0]) + (spine_shoulder[1]-spine_base[1])*(spine_shoulder[1]-spine_base[1]) + (spine_shoulder[2]-spine_base[2])*(spine_shoulder[2]-spine_base[2]))**0.5
        handIndex = 0
        for j in range(25):
            #if j==5 or j==6 or j==7 or j==9 or j==10 or j==11 or j==21 or j==22 or j==23 or j==24:
                normGesture[i][handIndex * 3] = (gaussianBuf[i][j*3] - spine_base[0])/d
                normGesture[i][handIndex * 3 + 1] = (gaussianBuf[i][j * 3 + 1] - spine_base[1]) / d
                normGesture[i][handIndex * 3 + 2] = (gaussianBuf[i][j * 3 + 2] - spine_base[2]) / d
                handIndex += 1
    return normGesture

def dataReduction(gestureBuf, normSize = 65):
    gestureLength = gestureBuf.shape[0]
    featureSize = gestureBuf.shape[1]
    normGesture = np.zeros((normSize,featureSize))

    for i in range(normSize):
        count = int(float(i) / float(normSize) * float(gestureLength))
        zerobuf = np.zeros(featureSize)
        normGesture[i] = gestureBuf[count]

    normSize = normSize * featureSize
    gesture = normGesture.reshape(normSize, order='C')
    return gesture


def getData(dataPath, labelPath):
    timeData = np.loadtxt(labelPath, dtype = int)
    skeletonData = np.loadtxt(dataPath, dtype = float)
    gestureBuf1 = skeletonData[timeData[0,0]:timeData[0,1]]
    gestureBuf2 = skeletonData[timeData[1, 0]:timeData[1, 1]]
    gestureBuf3 = skeletonData[timeData[2, 0]:timeData[2, 1]]
    gestureBuf1 = jointNorm(gaussianSmooth(gestureBuf1))
    gestureBuf2 = jointNorm(gaussianSmooth(gestureBuf2))
    gestureBuf3 = jointNorm(gaussianSmooth(gestureBuf3))
    gestureBuf1 = dataReduction(gestureBuf1)
    gestureBuf2 = dataReduction(gestureBuf2)
    gestureBuf3 = dataReduction(gestureBuf3)
    return gestureBuf1, gestureBuf2, gestureBuf3

#65
if __name__ == '__main__':
    dataPath = "SourceDataPath"
    userName = getSubDir(dataPath)
    userNum = len(userName)
    templateNum = 20
    trainGesture_V = open("./LabeledData/Train/Gesture_V/skeleton.txt", 'w')
    trainGesture_Label_V = open("./LabeledData/Train/Gesture_V/userID.txt", 'w')
    testGesture_V = open("./LabeledData/Test/Gesture_V/skeleton.txt", 'w')
    testGesture_Label_V = open("./LabeledData/Test/Gesture_V/userID.txt", 'w')
    trainGesture_O = open("./LabeledData/Train/Gesture_O/skeleton.txt", 'w')
    trainGesture_Label_O = open("./LabeledData/Train/Gesture_O/userID.txt", 'w')
    testGesture_O = open("./LabeledData/Test/Gesture_O/skeleton.txt", 'w')
    testGesture_Label_O = open("./LabeledData/Test/Gesture_O/userID.txt", 'w')
    trainGesture_Clapping = open("./LabeledData/Train/Gesture_Clapping/skeleton.txt", 'w')
    trainGesture_Label_Clapping = open("./LabeledData/Train/Gesture_Clapping/userID.txt", 'w')
    testGesture_Clapping = open("./LabeledData/Test/Gesture_Clapping/skeleton.txt", 'w')
    testGesture_Label_Clapping = open("./LabeledData/Test/Gesture_Clapping/userID.txt", 'w')
    for userID in range(userNum):
        randArr = np.arange(20)
        np.random.shuffle(randArr)
        print(userID)
        for templateID in range(templateNum):
            skeletonPath = dataPath + "/" + userName[userID] + "/" + str(randArr[templateID]) + "/" + "body3DData.txt"
            labelPath = dataPath + "/" + userName[userID] + "/" + str(randArr[templateID]) + "/" + "label.txt"
            gestureRightHand, gestureLeftHand, gestureBothHand = getData(skeletonPath,labelPath)
            if templateID < 6 :
                testGesture_Label_V.write(str(userID))
                testGesture_Label_V.write('\n')
                for col in range(len(gestureRightHand)):
                    testGesture_V.write(str(gestureRightHand[col]))
                    testGesture_V.write('\t')
                testGesture_V.write('\n')
                testGesture_Label_O.write(str(userID))
                testGesture_Label_O.write('\n')
                for col in range(len(gestureLeftHand)):
                    testGesture_O.write(str(gestureLeftHand[col]))
                    testGesture_O.write('\t')
                testGesture_O.write('\n')
                testGesture_Label_Clapping.write(str(userID))
                testGesture_Label_Clapping.write('\n')
                for col in range(len(gestureBothHand)):
                    testGesture_Clapping.write(str(gestureBothHand[col]))
                    testGesture_Clapping.write('\t')
                testGesture_Clapping.write('\n')
            else :
                trainGesture_Label_V.write(str(userID))
                trainGesture_Label_V.write('\n')
                for col in range(len(gestureRightHand)):
                    trainGesture_V.write(str(gestureRightHand[col]))
                    trainGesture_V.write('\t')
                trainGesture_V.write('\n')
                trainGesture_Label_O.write(str(userID))
                trainGesture_Label_O.write('\n')
                for col in range(len(gestureLeftHand)):
                    trainGesture_O.write(str(gestureLeftHand[col]))
                    trainGesture_O.write('\t')
                trainGesture_O.write('\n')
                trainGesture_Label_Clapping.write(str(userID))
                trainGesture_Label_Clapping.write('\n')
                for col in range(len(gestureBothHand)):
                    trainGesture_Clapping.write(str(gestureBothHand[col]))
                    trainGesture_Clapping.write('\t')
                trainGesture_Clapping.write('\n')
    print("finished!")
