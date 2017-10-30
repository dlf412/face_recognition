#!/usr/bin/env python
#encoding:utf-8

from ctypes import *
#from PIL import Image
import numpy as np
import cv2
import time

import sys
pyVer = sys.version_info[0]

import platform

system_name = platform.system()

if system_name == u"Linux":
    APPID = b"Your APPID"
    FD_SDKKEY = b"******"
    FT_SDKKEY = b"******"
    FR_SDKKEY = b"******"
    detect_dll = "libarcsoft_fsdk_face_detection.so"
    recognition_dll = "libarcsoft_fsdk_face_recognition.so"

if system_name == u"Windows":
    APPID = b"Your APPID"
    FD_SDKKEY = b"******"
    FT_SDKKEY = b"******"
    FR_SDKKEY = b"******"
    detect_dll = "libarcsoft_fsdk_face_detection.dll"
    recognition_dll = "libarcsoft_fsdk_face_recognition.dll"

class Rect(Structure):
    #_pack_ = 1
    _fields_ = [("left", c_int32),
                ("top", c_int32),
                ("right", c_int32),
                ("bottom", c_int32)]

class AFR_FSDK_FACEINPUT(Structure):
    _fields_ = [
        ("rcFace", Rect),
        ("lOrient", c_int32)
    ]

class AFR_FSDK_FACEMODEL(Structure):
    _fields_ = [
        ("pbFeature", POINTER(c_ubyte)),
        ("lFeatureSize", c_int32)
    ]

class Aft_fsdk_faceres(Structure):
    _fields_ = [
        ("nFace", c_int32),
        ("lfaceOrient", c_int32),
        ("rcFace", POINTER(Rect))
    ]

class ASVLOFFSCREEN(Structure):
    _fields_ = [
        ("u32PixelArrayFormat", c_uint32),
        ("i32Width", c_int32),
        ("i32Height", c_int32),
        ("ppu8Plane", (POINTER(c_uint8))*4),
        ("pi32Pitch", c_int32*4)
    ]

class AFD_FSDK_FACERES(Structure):
    #_pack_ = 1
    _fields_ = [
        ("nFace", c_int32),
        ("rcFace", POINTER(Rect)),
        ("lfaceOrient", POINTER(c_int32))
    ]

class FREngine(object):
    WORKBUF_SIZE_FR = 40 * 1024 * 1024
    def __init__(self):
        self._fr_dll = CDLL(recognition_dll)
        self._pWorkMem = create_string_buffer(self.WORKBUF_SIZE_FR)
        self._handler = c_void_p(None)

        self._fr_dll.AFR_FSDK_ExtractFRFeature.restype = c_int32
        self._fr_dll.AFR_FSDK_ExtractFRFeature.argtypes = [c_void_p, POINTER(ASVLOFFSCREEN),
                                                           POINTER(AFR_FSDK_FACEINPUT), POINTER(AFR_FSDK_FACEMODEL)]

        ret = self._fr_dll.AFR_FSDK_InitialEngine(APPID, FR_SDKKEY,
                                     byref(self._pWorkMem),
                                     self.WORKBUF_SIZE_FR,
                                     byref(self._handler)
                                     )
        if ret != 0:
            raise Exception("Init FEngine failed, errorcode:{}".format(ret))

    def extractFRFeature(self, ImgData, FaceRes):
        face_model = AFR_FSDK_FACEMODEL()
        ret = self._fr_dll.AFR_FSDK_ExtractFRFeature(self._handler, byref(ImgData),
                                                     byref(FaceRes), byref(face_model))

        if ret != 0:
            raise Exception("extract face Feature failed, errorcode:{}".format(ret))

        if pyVer == 3:
            return bytes([face_model.pbFeature[i] for i in range(face_model.lFeatureSize)])
        else:
            return ''.join([chr(face_model.pbFeature[i]) for i in range(face_model.lFeatureSize)])


    def faceMatch(self, face_model1, face_model2):
        model1 = AFR_FSDK_FACEMODEL(
            pbFeature=cast((c_ubyte * len(face_model1)).from_buffer_copy(face_model1), POINTER(c_ubyte)),
            lFeatureSize=len(face_model1))

        model2 = AFR_FSDK_FACEMODEL(
            pbFeature=cast((c_ubyte * len(face_model2)).from_buffer_copy(face_model2), POINTER(c_ubyte)),
            lFeatureSize=len(face_model2))

        SimilScore = c_float(0)
        ret = self._fr_dll.AFR_FSDK_FacePairMatching(self._handler, byref(model1),
                                                     byref(model2), byref(SimilScore))
        if ret != 0:
            raise Exception("match face Feature failed, errorcode:{}".format(ret))
        return SimilScore.value

    def __del__(self):
        self._fr_dll.AFR_FSDK_UninitialEngine(self._handler)
        if self._pWorkMem:
            del self._pWorkMem

class FDEngine(object):
    WORKBUF_SIZE_FD = 40*1024*1024
    ASVL_PAF_RGB24_B8G8R8 = 0x201
    ASVL_PAF_RGB32_B8G8R8A8 = 0x302
    AFD_FSDK_OPF_0_HIGHER_EXT = 0x5

    @staticmethod
    def resize2even(array):
        h, w, _ = array.shape
        if h % 2 == 1:
            h += 1
        if w % 2 == 1:
            w += 1
        return cv2.resize(array, (w, h),interpolation=cv2.INTER_AREA)


    def __init__(self):
        self._fd_dll = CDLL(detect_dll)
        self._pWorkMem = create_string_buffer(self.WORKBUF_SIZE_FD)
        self._handler = c_void_p(None)
        ret = self._fd_dll.AFD_FSDK_InitialFaceEngine(APPID, FD_SDKKEY,
                                                 byref(self._pWorkMem),
                                                 self.WORKBUF_SIZE_FD,
                                                 byref(self._handler),
                                                 self.AFD_FSDK_OPF_0_HIGHER_EXT,
                                                 16, 12)
        if ret != 0:
            raise Exception("Init FDEngine failed, errorcode:{}".format(ret))

    def df_from_file(self, imgfile):
        #im = Image.open(imgfile, 'rb')
        #b, g, r = im.split()
        #im = Image.merge("RGB", (r,g,b))
        #raw_data = im.tobytes
        #array = np.asarray(im)
        #Image.
        array = cv2.imread(imgfile)
        if array is not None:
            return self.df_from_array(array)
        else:
            raise Exception("{} not a image file or not exists".format(imgfile))

    def df_from_array(self, array):
        pImgData = ASVLOFFSCREEN()
        h, w, m = array.shape
        if m == 3:
            pImgData.u32PixelArrayFormat = self.ASVL_PAF_RGB24_B8G8R8
        elif m == 4:
            pImgData.u32PixelArrayFormat = self.ASVL_PAF_RGB32_B8G8R8A8

        if h%2 or w%2:
            array = self.resize2even(array)
        raw_data = array.tobytes()
        pbuffer = (c_ubyte * len(raw_data)).from_buffer_copy(raw_data)
        pImgData.i32Width = w
        pImgData.i32Height = h
        pImgData.ppu8Plane[0] = cast(pbuffer, POINTER(c_uint8))
        #pImgData.ppu8Plane[0] = cast(pointer(pbuffer), POINTER(c_uint8))
        pImgData.pi32Pitch[0] = w*3
        pFacers = POINTER(AFD_FSDK_FACERES)()
        ret = self._fd_dll.AFD_FSDK_StillImageFaceDetection(self._handler, byref(pImgData), byref(pFacers))
        if ret != 0:
            raise Exception("detect face faield, errorcode:{}".format(ret))
        faceres = pFacers.contents
        retFacers = []
        for i in range(faceres.nFace):
            face = AFR_FSDK_FACEINPUT()
            face.rcFace = faceres.rcFace[i]
            face.lOrient = faceres.lfaceOrient[i]
            retFacers.append(face)
        return retFacers, pImgData

    def __del__(self):
        self._fd_dll.AFD_FSDK_UninitialFaceEngine(self._handler)
        if self._pWorkMem:
            del self._pWorkMem

if __name__ == '__main__':
    
    ref = sys.argv[1]
    sample = sys.argv[2]

    fd_eng = FDEngine()
    fr_eng = FREngine()

    face_model1 = face_model2 = None

    import time
    start = time.clock()
    faceres, ImgData = fd_eng.df_from_file(ref)
    print "detect face cost:{}".format(time.clock() - start)
    for face in faceres:
        print face.rcFace.left, face.rcFace.top, face.rcFace.right, face.rcFace.bottom
        start = time.clock()
        face_model1 = fr_eng.extractFRFeature(ImgData, face)
        print "extract feature cost:{}".format(time.clock() - start)

    faceres2, ImgData2 = fd_eng.df_from_file(sample)
    for face in faceres2:
        print face.rcFace.left, face.rcFace.top, face.rcFace.right, face.rcFace.bottom
        face_model2 = fr_eng.extractFRFeature(ImgData2, face)

    if face_model1 and face_model2:
        start = time.clock()
        print fr_eng.faceMatch(face_model1, face_model2)
        print "match features cost:{}".format(time.clock() - start)
    else:
        print "facemode1: {}, facemodel2:{}".format(face_model1, face_model2)

