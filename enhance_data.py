import numpy as np
import cv2
import os
#import argparse


def Histogram_segmentation(img):  # adaptive
    cr = 0
    final_plus = img.copy()  # 是否0
    img_max = img.max()
    img_min = img.min()
    S = (img_max-img_min)*cr
    T1 = S
    T2 = img_max-S
    idx1 = img < T1
    img[idx1] = 0
    idx2 = img > T2
    img[idx2] = 0
    # 得到濾過後該channel最大值及最小值
    img_max = img.max()
    img_min = img[img > 0].min()
    a, b = img.shape
    img_strech = np.zeros((a, b))
    img_strech[img > 0] = (img[img > 0]-img_min)/(img_max-img_min)*255
    img_strech[idx1] = final_plus[idx1]  # 是否0
    img_strech[idx2] = final_plus[idx2]  # 是否0
    img_strech = img_strech.astype('uint8')
    return img_strech


def enhance_image(image):
    (B, G, R) = cv2.split(image)
    img_g_strech, img_r_strech, img_b_strech = Histogram_segmentation(
        G), Histogram_segmentation(R), Histogram_segmentation(B)
    img_enhance = cv2.merge([img_b_strech, img_g_strech, img_r_strech])
    # print("OK")
    return img_enhance


if __name__ == "__main__":
    # 構造參數解析器
    ##ap = argparse.ArgumentParser()
    ##ap.add_argument("-i", "--image", required=True, help="Path to the image")
    ##args = vars(ap.parse_args())

    dataPath = "C:/Users/Sally_Lab/Desktop/Liton/test2/"
    datasavePath = "C:\\Users\\Sally_Lab\\Desktop\\Liton\\test22\\"
    # 遞迴列出所有子目錄與檔案
    for root, dirs, files in os.walk(dataPath):
        print("root", root)
        print("dirs", dirs)
        #print("files", files)
        if root != dataPath:
            root = root + "/"
            for file in files:
                #print("file", file)
                image = cv2.imread(
                    root + file)
                img_enhance = enhance_image(image)
                cv2.imwrite(
                    root + file, img_enhance)
    '''
    FileList = os.listdir(dataPath)
    for file in FileList:
        print(file)
        # 讀取圖片
        
        image = cv2.imread(
            dataPath+file)
        ##image = cv2.imread(args["image"])
        # print("go")
        img_enhance = enhance_image(image)
        # print("111")
        # cv2.imshow(img_enhance)
        # print("222")
        cv2.imwrite(
            datasavePath + file, img_enhance)
        print("333")

'''
