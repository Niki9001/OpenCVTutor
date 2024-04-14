
import cv2
import numpy as np




# def process_license_plate(image):
#     # 高斯去噪
#     image = cv2.GaussianBlur(image, (3, 3), 0)
#     # 灰度处理
#     gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
#     # 自适应阈值处理
#    # thresh_image = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
#    #                                               cv2.THRESH_BINARY, 11, 2)
#     ret, thresh_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_OTSU)
#     # 去除一些小的白点和进行形态学操作
#     #kernelX = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 5))
#     #kernelY = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 5))
#     # 膨胀，腐蚀
#     #thresh_image = cv2.dilate(thresh_image1, kernelX)
#     #thresh_image = cv2.erode(thresh_image, kernelX)
#     # 腐蚀，膨胀
#     #thresh_image = cv2.erode(thresh_image, kernelY)
#     #thresh_image = cv2.dilate(thresh_image, kernelY)
#
#     # 使用cv2.RETR_TREE模式寻找轮廓
#     #contours, hierarchy = cv2.findContours(thresh_image.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#
#     #filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 400]
#
#     # 绘制轮廓
#     #cv2.drawContours(image, filtered_contours, -1, (255, 0, 0), 2)
#     return thresh_image

# import cv2
# import numpy as np
#
#
def process_license_plate(image):
    # 高斯去噪
    image_blurred = cv2.GaussianBlur(image, (1, 1), 0)

    # 灰度处理
    gray_image = cv2.cvtColor(image_blurred, cv2.COLOR_RGB2GRAY)

    # Canny边缘检测
   # edges = cv2.Canny(gray_image, 100, 200)

    # 形态学操作：先膨胀再腐蚀，增强图像中的结构元素
    #kernel = np.ones((3, 3), np.uint8)
    #edges_dilated = cv2.dilate(edges, kernel, iterations=1)
    #edges_processed = cv2.erode(edges_dilated, kernel, iterations=1)

    # 再次使用自适应阈值处理，以确保边缘检测后的图像有清晰的二值化效果
    #thresh_image = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
     #                                    cv2.THRESH_BINARY, 11, 2)
    #ret, thresh_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_OTSU)
    ret, thresh_image = cv2.threshold(gray_image, 180, 255, cv2.THRESH_BINARY)

    return thresh_image

