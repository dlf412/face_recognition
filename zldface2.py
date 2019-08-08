#!/usr/bin/env python
# encoding:utf-8

import io
import sys
from ctypes import *

import cv2
import os
from PIL import Image
import numpy as np
pyVer = sys.version_info[0]
import platform

system_name = platform.system()

if system_name == u"Linux":
    APPID = b"Fqa9LM1ww4qcT58MWYjfkDq8DPeycb76t4jhK7Eu9QNY"
    SDKKEY = b"3yXBYs6VGfvC1voKfxB4c2jmbn4eoPQEjLaBYbnZhFEB"
    face_dll = "libarcsoft_face.so"
    face_engine_dll = "libarcsoft_face_engine.so"
    asf_install_file = ".asf_install.dat"

if system_name == u"Windows":
    APPID = b"Fqa9LM1ww4qcT58MWYjfkDq8DPeycb76t4jhK7Eu9QNY"
    SDKKEY = b"3yXBYs6VGfvC1voKfxB4c2jmjtdpGFYYgUp9Qk8wYCNb"
    face_dll = "libarcsoft_face.dll"
    face_engine_dll = "libarcsoft_face_engine.dll"
    asf_install_file = "asf_install.dat"

c_ubyte_p = POINTER(c_uint8)
c_void_pp = POINTER(c_void_p)

ASF_OP_0_ONLY = 1			# 0, 0, ...
ASF_OP_90_ONLY = 2			# 90, 90, ...
ASF_OP_270_ONLY = 3			# 270, 270, ...
ASF_OP_180_ONLY = 4			# 180, 180, ...
ASF_OP_0_HIGHER_EXT = 5		# 0, 90, 270, 180, 0, 90, 270, 180, ...

ASF_NONE = 0
ASF_FACE_DETECT = 1
ASF_FACERECOGNITION = 4
ASF_AGE = 8
ASF_GENDER = 16
ASF_FACE3DANGLE = 32

ASF_DETECT_MODE_VIDEO = 0		# Video模式，一般用于多帧连续检测
ASF_DETECT_MODE_IMAGE =	0xFFFFFFFF		# Image模式，一般用于静态图的单次检测

ASVL_PAF_RGB24_B8G8R8=0x201
ASVL_PAF_RGB32_B8G8R8A8=0x302

class Rect(Structure):
    # _pack_ = 1
    _fields_ = [("left", c_int32),
                ("top", c_int32),
                ("right", c_int32),
                ("bottom", c_int32)]

class ASF_MultiFaceInfo(Structure):
    _fields_ = [
        ("faceRect", POINTER(Rect)),  # 人脸框信息
        ("faceOrient", POINTER(c_int32)),  # 输入图像的角度，可以参考 ArcFaceCompare_OrientCode
        ("faceNum", c_int32)
    ]

class ASF_SingleFaceInfo(Structure):
    _fields_ = [
        ("faceRect", Rect),  # 人脸框信息
        ("faceOrient", c_int32),  # 输入图像的角度，可以参考 ArcFaceCompare_OrientCode
    ]

class ASF_FaceFeature(Structure):
    _fields_ = [
        ("feature", c_ubyte_p),   # 人脸特征信息
        ("featureSize", c_int32)  # 人脸特征信息长度
    ]

class BGRFile(object):
    def __init__(self, filePath=None, array=None, buffer=None):
        if filePath:
            oldimg = Image.open(filePath)
            self.width = oldimg.width & 0xFFFFFFFC
            self.height = oldimg.height
            if (self.width != oldimg.width):
                crop_area = (0, 0, self.width, self.height)
                img = oldimg.crop(crop_area)
            else:
                img = oldimg
            BMP_bytes = io.BytesIO()
            img.transpose(Image.FLIP_TOP_BOTTOM).convert('RGB').save(BMP_bytes, format='BMP')
            bgr_buffer = bytes(BMP_bytes.getvalue()[54:])
            self.imgdata = (c_ubyte * len(bgr_buffer)).from_buffer_copy(bgr_buffer)
        elif array is not None:
            h, w, m = array.shape
            self.width = ((w - 1) & 0xFFFFFFFC) + 4
            self.height = ((h - 1) & 0xFFFFFFFE) + 2
            array = cv2.resize(array, (self.width, self.height), interpolation=cv2.INTER_AREA)
            raw_data = array.tobytes()
            self.imgdata = (c_ubyte * len(raw_data)).from_buffer_copy(raw_data)
        elif buffer:
            nparr = np.fromstring(buffer, np.uint8)
            array = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if array is not None:
                h, w, m = array.shape
                self.width = ((w - 1) & 0xFFFFFFFC) + 4
                self.height = ((h - 1) & 0xFFFFFFFE) + 2
                array = cv2.resize(array, (self.width, self.height), interpolation=cv2.INTER_AREA)
                raw_data = array.tobytes()
                self.imgdata = (c_ubyte * len(raw_data)).from_buffer_copy(raw_data)
            else:
                raise Exception("BRFFile convert buffer to array failed")
        else:
            raise Exception("Not Supportted BGRFile Format Input")



class FaceEngine(object):
    def __init__(self, dll_dir='./', scale=16, facenum=1):
        # self._workdir = dll_dir
        # 工作目录切换到_workdir
        try:
            run_dir = os.getcwd()
            os.chdir(dll_dir)
            self._dll = CDLL(face_engine_dll)
            # 激活
            self._dll.ASFActivation.restype = c_long
            self._dll.ASFActivation.argtypes = [
                c_char_p,
                c_char_p
            ]

            # 初始化引擎
            self._dll.ASFInitEngine.restype = c_long
            self._dll.ASFInitEngine.args = [
                c_long,  # [in]	AF_DETECT_MODE_VIDEO 视频模式：适用于摄像头预览，视频文件识别
                         # AF_DETECT_MODE_IMAGE 图片模式：适用于静态图片的识别；
                c_int32,  # [in]	检测脸部的角度优先值，参考 ArcFaceCompare_OrientPriority
                c_int32,  # [in] 用于数值化表示的最小人脸尺寸，该尺寸代表人脸尺寸相对于图片长边的占比。
                          # 图像数据尺寸为1280x720，设置nscale为8，则检测到的最小人脸长边为1280/8 = 160	 例如，用户想检测到的最小人脸尺寸是图片长边的 1/8，则scaleVal设置为 8
                          # video 模式有效值范围[2,16], Image 模式有效值范围[2,32],推荐值为 16
                c_int32, # [in] 最大需要检测的人脸个数
                c_int32, # [in] 用户选择需要检测的功能组合，可单个或多个  # ASF_FACE_DETECT | ASF_FACERECOGNITION | ASF_AGE | ASF_GENDER | ASF_FACE3DANGLE;
                c_void_pp,  # [out] 初始化返回的引擎handle
            ]

            # 检测人脸
            self._dll.ASFDetectFaces.restype = c_long
            self._dll.ASFDetectFaces.argtypes = [
                c_void_p, # handle
                c_int32,  # width
                c_int32,  # height
                c_int32,  # format
                c_ubyte_p, # imgData
                POINTER(ASF_MultiFaceInfo)
            ]

            # 提取特征
            self._dll.ASFFaceFeatureExtract.restype = c_long
            self._dll.ASFFaceFeatureExtract.argtypes = [
                c_void_p,  # handle
                c_int32,  # width
                c_int32,  # height
                c_int32,  # format
                c_ubyte_p,  # imgData
                POINTER(ASF_SingleFaceInfo),
                POINTER(ASF_FaceFeature)
            ]

            self._dll.ASFFaceFeatureCompare.restype = c_long
            self._dll.ASFFaceFeatureCompare.argtypes = [
                c_void_p,  # handle
                POINTER(ASF_FaceFeature),  # width
                POINTER(ASF_FaceFeature),
                POINTER(c_float)
            ]

            self._dll.ASFUninitEngine.restype = c_long
            self._dll.ASFUninitEngine.argtypes = [
                c_void_p
            ]

            self._handler = c_void_p(None)
            # 激活
            if not os.path.exists(asf_install_file):
                res = self._dll.ASFActivation(c_char_p(APPID), c_char_p(SDKKEY))
                if res != 0 and res != 90114:  # 90114已激活
                    raise Exception("sdk ASFActivation error, res:%d" % res)
            self._initEngine(scale=scale, facenum=facenum)
        finally:
            os.chdir(run_dir)

    def _initEngine(self, model=c_long(ASF_DETECT_MODE_IMAGE), op=ASF_OP_0_ONLY,
                  scale=16, facenum=1, mask=(ASF_FACE_DETECT | ASF_FACERECOGNITION)):

        resp = self._dll.ASFInitEngine(model, op, scale, facenum, mask, byref(self._handler))
        if resp != 0:
            raise Exception("sdk ASFInitEngine error, res:%d" % resp)

    def df_from_file(self, imgfile):
        try:
            img = BGRFile(imgfile)
        except Exception:
            raise Exception("not a image")
        faces = ASF_MultiFaceInfo()
        resp = self._dll.ASFDetectFaces(self._handler, img.width, img.height, ASVL_PAF_RGB24_B8G8R8,
                                 img.imgdata, byref(faces))
        if resp != 0:
            raise Exception("sdk ASFDetectFaces error, res:%d" % resp)

        return [ASF_SingleFaceInfo(faceRect=faces.faceRect[i],
                                   faceOrient=faces.faceOrient[i]) for i in range(faces.faceNum)], img

    def df_from_buffer(self, buffer):
        img = BGRFile(buffer=buffer)
        faces = ASF_MultiFaceInfo()
        resp = self._dll.ASFDetectFaces(self._handler, img.width, img.height, ASVL_PAF_RGB24_B8G8R8,
                                        img.imgdata, byref(faces))
        if resp != 0:
            raise Exception("sdk ASFDetectFaces error, res:%d" % resp)

        return [ASF_SingleFaceInfo(faceRect=faces.faceRect[i],
                                   faceOrient=faces.faceOrient[i]) for i in range(faces.faceNum)], img

    def df_from_array(self, array):
        img = BGRFile(array=array)
        faces = ASF_MultiFaceInfo()
        resp = self._dll.ASFDetectFaces(self._handler, img.width, img.height, ASVL_PAF_RGB24_B8G8R8,
                                        img.imgdata, byref(faces))
        if resp != 0:
            raise Exception("sdk ASFDetectFaces error, res:%d" % resp)

        return [ASF_SingleFaceInfo(faceRect=faces.faceRect[i],
                                   faceOrient=faces.faceOrient[i]) for i in range(faces.faceNum)], img

    def extractFRFeature(self, img, face):
        feature = ASF_FaceFeature()
        resp = self._dll.ASFFaceFeatureExtract(self._handler, img.width, img.height, ASVL_PAF_RGB24_B8G8R8,
                                               img.imgdata, byref(face),
                                               byref(feature))
        if resp != 0:
            raise Exception("sdk ASFFaceFeatureExtract error, res:%d" % resp)
        if pyVer == 3:
            return bytes([feature.feature[i] for i in range(feature.featureSize)])
        else:
            return ''.join([chr(feature.feature[i]) for i in range(feature.featureSize)])

    def faceMatch(self, feature1, feature2):
        model1 = ASF_FaceFeature(feature= (c_ubyte * len(feature1)).from_buffer_copy(feature1),
                                 featureSize=len(feature1))
        model2 = ASF_FaceFeature(feature=(c_ubyte * len(feature2)).from_buffer_copy(feature2),
                                 featureSize=len(feature2))
        score = c_float(0.0)
        resp = self._dll.ASFFaceFeatureCompare(self._handler, byref(model1), byref(model2), byref(score))
        if resp != 0:
            raise Exception("sdk ASFFaceFeatureCompare error, res:%d" % resp)
        return score.value

    def __del__(self):
        self._dll.ASFUninitEngine(self._handler)


if __name__ == '__main__':
    engine = FaceEngine()
    import time

    start = time.clock()
    faces1, imgdata1 = engine.df_from_file("1504239052.15_face.jpg") # "zp.bmp")
    faces2, imgdata2 = engine.df_from_file("36250219840410.jpg")
    # print "detect 2 image cost %f" % (time.clock() - start)
    #
    # print faces1[0].faceRect.top, faces1[0].faceRect.left, faces1[0].faceRect.bottom, faces1[0].faceRect.right
    # print faces2[0].faceRect.top, faces2[0].faceRect.left, faces2[0].faceRect.bottom, faces2[0].faceRect.right

    start = time.clock()
    if faces1:
        #print faces1.faceRect, faces1.faceOrient
        feature1 = engine.extractFRFeature(imgdata1, faces1[0])

    if faces2:
        #print faces2.faceRect, faces2.faceOrient
        feature2 = engine.extractFRFeature(imgdata2, faces2[0])
    # print "extract 2 image cost %f" % (time.clock() - start)

    start = time.clock()
    # print engine.faceMatch(feature1, feature2)
    # print "match cost %f" %(time.clock() - start)


    faces3, imgdata3 = engine.df_from_file("agent.jpg")
    assert faces3 == []

    try:
        face4, imgdata4 = engine.df_from_file("zldface.py")
    except Exception as err:
        print (str(err))

