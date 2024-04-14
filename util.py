import string
import easyocr
import cv2
import numpy as np

# Initialize the OCR reader
reader = easyocr.Reader(['en'], gpu=False)

# Mapping dictionaries for character conversion
dict_char_to_int = {'O': '0',
                    'I': '1',
                    'J': '3',
                    'A': '4',
                    'G': '6',
                    'S': '5'}

dict_int_to_char = {'0': 'O',
                    '1': 'I',
                    '3': 'J',
                    '4': 'A',
                    '6': 'G',
                    '5': 'S'}


def write_csv(results, output_path):
    """
    Write the results to a CSV file.

    Args:
        results (dict): Dictionary containing the results.
        output_path (str): Path to the output CSV file.
    """
    with open(output_path, 'w') as f:
        f.write('{},{},{},{},{},{},{}\n'.format('frame_nmr', 'car_id', 'car_bbox',
                                                'license_plate_bbox', 'license_plate_bbox_score', 'license_number',
                                                'license_number_score'))

        for frame_nmr in results.keys():
            for car_id in results[frame_nmr].keys():
                print(results[frame_nmr][car_id])
                if 'car' in results[frame_nmr][car_id].keys() and \
                   'license_plate' in results[frame_nmr][car_id].keys() and \
                   'text' in results[frame_nmr][car_id]['license_plate'].keys():
                    f.write('{},{},{},{},{},{},{}\n'.format(frame_nmr,
                                                            car_id,
                                                            '[{} {} {} {}]'.format(
                                                                results[frame_nmr][car_id]['car']['bbox'][0],
                                                                results[frame_nmr][car_id]['car']['bbox'][1],
                                                                results[frame_nmr][car_id]['car']['bbox'][2],
                                                                results[frame_nmr][car_id]['car']['bbox'][3]),
                                                            '[{} {} {} {}]'.format(
                                                                results[frame_nmr][car_id]['license_plate']['bbox'][0],
                                                                results[frame_nmr][car_id]['license_plate']['bbox'][1],
                                                                results[frame_nmr][car_id]['license_plate']['bbox'][2],
                                                                results[frame_nmr][car_id]['license_plate']['bbox'][3]),
                                                            results[frame_nmr][car_id]['license_plate']['bbox_score'],
                                                            results[frame_nmr][car_id]['license_plate']['text'],
                                                            results[frame_nmr][car_id]['license_plate']['text_score'])
                            )
        f.close()


def license_complies_format(text):
    """
    Check if the license plate text complies with the required format.

    Args:
        text (str): License plate text.

    Returns:
        bool: True if the license plate complies with the format, False otherwise.
    """


    if (text[0] in string.ascii_uppercase or text[0] in dict_int_to_char.keys()) and \
       (text[1] in string.ascii_uppercase or text[1] in dict_int_to_char.keys()) and \
       (text[2] in string.ascii_uppercase or text[1] in dict_int_to_char.keys()) and \
       (text[3] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[3] in dict_char_to_int.keys()) and \
       (text[4] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[4] in dict_char_to_int.keys()) and \
       (text[5] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[5] in dict_char_to_int.keys()):
        return True
    else:
        return False


def format_license(text):
    """
    Format the license plate text by converting characters using the mapping dictionaries.

    Args:
        text (str): License plate text.

    Returns:
        str: Formatted license plate text.
    """
    license_plate_ = ''
    combined_mapping = {'O': '0', 'I': '1'}
    for char in text:
        if char in combined_mapping:
            license_plate_ += combined_mapping[char]
        else:
            license_plate_ += char

    return text


# def read_license_plate(license_plate_crop):
#     """
#     Read the license plate text from the given cropped image. Only return results
#     that are at least 6 characters long, with the first three characters being letters
#     (with 'O' converted to 'D') and the last three being digits. Ignore any characters
#     between the letters and digits.
#
#     Args:
#         license_plate_crop (PIL.Image.Image): Cropped image containing the license plate.
#
#     Returns:
#         tuple: Tuple containing the formatted license plate text and its confidence score,
#                or None, None if no valid plate text is found.
#     """
#     detections = reader.readtext(license_plate_crop)
#
#     for detection in detections:
#         bbox, text, score = detection
#
#         # Convert text to uppercase and remove all spaces and non-alphanumeric characters
#         filtered_text = ''.join(filter(str.isalnum, text.upper()))
#
#         # Only proceed if the filtered text is longer than 6 characters
#         if len(filtered_text) > 6:
#             # Convert 'O' to 'D' in the first three letters
#             letters = ''.join([c if c != 'O' else 'D' for c in filtered_text[:3] if c.isalpha()])
#
#             # Extract the last three digits
#             digits = ''.join([c for c in filtered_text[-3:] if c.isdigit()])
#
#             # Only return the result if we have exactly 3 letters followed by 3 digits
#             if len(letters) == 3 and len(digits) == 3:
#                 return letters + digits, score
#
#     return None, None
def read_license_plate(license_plate_crop):
    """
    Read the license plate text from the given cropped image, only return results
    that are exactly 6 characters long and consist of letters and digits.

    Args:
        license_plate_crop (PIL.Image.Image): Cropped image containing the license plate.

    Returns:
        tuple: Tuple containing the formatted license plate text and its confidence score.
    """

    detections = reader.readtext(license_plate_crop)

    for detection in detections:
        bbox, text, score = detection

        # Convert text to uppercase and remove all spaces and non-alphanumeric characters
        filtered_text = ''.join(filter(str.isalnum, text.upper()))

        # Check if the filtered text is exactly 6 characters long
        if len(filtered_text) ==6 :
            return filtered_text, score

    return None, None



def apply_perspective_transform(license_plate_crop, x1, y1, x2, y2):
    """
    对裁剪后的车牌图像应用透视变换。
    """
    # 确定源点：检测到的车牌的四个角点
    src_points = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)

    # 确定目标点：一个标准大小的矩形
    # 假设我们希望最终的车牌图像是200x100像素
    width, height = 200, int(200 * (y2 - y1) / (x2 - x1))
    dst_points = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)

    # 计算透视变换矩阵
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)

    # 应用透视变换
    license_plate_transformed = cv2.warpPerspective(license_plate_crop, matrix, (width, height), flags=cv2.INTER_CUBIC)

    return license_plate_transformed


# def read_license_plate(license_plate_crop):
#     """
#     Read the license plate text from the given cropped image, only return results
#     that are exactly 6 characters long and consist of letters and digits.
#
#     Args:
#         license_plate_crop (PIL.Image.Image): Cropped image containing the license plate.
#
#     Returns:
#         tuple: Tuple containing the formatted license plate text and its confidence score.
#     """
#
#     detections = reader.readtext(license_plate_crop)
#
#     for detection in detections:
#         bbox, text, score = detection
#
#         # Convert text to uppercase and remove all spaces and non-alphanumeric characters
#         filtered_text = ''.join(filter(str.isalnum, text.upper()))
#
#         # Check if the filtered text is exactly 6 characters long
#         if len(filtered_text) > 6 :
#             return filtered_text, score
#
#     return None, None


def get_car(license_plate, vehicle_track_ids):
    """
    Retrieve the vehicle coordinates and ID based on the license plate coordinates.

    Args:
        license_plate (tuple): Tuple containing the coordinates of the license plate (x1, y1, x2, y2, score, class_id).
        vehicle_track_ids (list): List of vehicle track IDs and their corresponding coordinates.

    Returns:
        tuple: Tuple containing the vehicle coordinates (x1, y1, x2, y2) and ID.
    """
    x1, y1, x2, y2, score, class_id = license_plate

    foundIt = False
    for j in range(len(vehicle_track_ids)):
        xcar1, ycar1, xcar2, ycar2, car_id = vehicle_track_ids[j]

        if x1 > xcar1 and y1 > ycar1 and x2 < xcar2 and y2 < ycar2:
            car_indx = j
            foundIt = True
            break

    if foundIt:
        return vehicle_track_ids[car_indx]

    return -1, -1, -1, -1, -1
