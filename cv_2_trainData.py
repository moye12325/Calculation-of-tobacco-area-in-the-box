import cv2
import numpy as np
import os

def calculate_area(image, lower_hsv, upper_hsv, min_contour_area, scale=1.0):
    """
    计算烟丝的总面积并绘制结果，同时生成掩码图像。
    :param image: 输入图像
    :param lower_hsv: HSV 范围下限
    :param upper_hsv: HSV 范围上限
    :param min_contour_area: 最小有效面积（像素单位）
    :param scale: 每像素实际长度，单位为 mm/像素
    :return: 绘制结果的图像、总面积、掩码图像
    """
    # 转换为灰度图像和 HSV
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 拉普拉斯变换
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    laplacian_abs = cv2.convertScaleAbs(laplacian)

    # 阈值分割
    _, binary_laplacian = cv2.threshold(laplacian_abs, 20, 255, cv2.THRESH_BINARY)

    # 基于颜色范围分割
    binary_hsv = cv2.inRange(hsv, lower_hsv, upper_hsv)

    # 结合结果
    combined_binary = cv2.bitwise_and(binary_laplacian, binary_hsv)

    # 形态学操作优化结果
    kernel = np.ones((3, 3), np.uint8)
    combined_cleaned = cv2.morphologyEx(combined_binary, cv2.MORPH_CLOSE, kernel, iterations=2)

    # 查找轮廓
    contours, _ = cv2.findContours(combined_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 初始化总面积
    total_area = 0.0

    # 创建副本图像用于绘制轮廓
    image_with_contours = image.copy()

    # 创建空白掩码图像（初始化为全零）
    mask = np.zeros(image.shape[:2], dtype=np.uint8)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > min_contour_area:  # 过滤掉小面积噪声
            total_area += area
            # 绘制轮廓
            cv2.drawContours(image_with_contours, [cnt], -1, (0, 255, 0), 2)  # 绿色轮廓
            # 在轮廓中心显示面积值
            M = cv2.moments(cnt)
            if M["m00"] != 0:  # 防止除以零
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                cv2.putText(image_with_contours, f"{area:.1f}", (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

            # 填充掩码图像上的轮廓区域为1
            cv2.drawContours(mask, [cnt], -1, 255, thickness=cv2.FILLED)

    # 计算实际面积（平方毫米）
    total_actual_area = total_area * (scale ** 2)

    # 显示总面积
    cv2.putText(image_with_contours, f"Total Area: {total_actual_area:.2f} mm^2",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    return image_with_contours, total_actual_area, mask


def process_images_in_directory(directory_path):
    """
    处理指定文件夹中的所有图像文件，生成对应的掩码和结果。
    :param directory_path: 图像文件夹路径
    """
    # 获取所有图像文件
    image_files = [f for f in os.listdir(directory_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    # 设置HSV范围和最小轮廓面积
    lower_hsv = np.array([9, 50, 50])  # 烟丝颜色的 HSV 下限
    upper_hsv = np.array([25, 255, 255])  # 烟丝颜色的 HSV 上限
    min_contour_area = 500  # 最小有效面积（像素单位）
    pixel_scale = 0.1  # 每像素的实际长度，单位：mm/像素

    # 遍历所有图像文件并处理
    for image_file in image_files:
        image_path = os.path.join(directory_path, image_file)
        image = cv2.imread(image_path)

        if image is None:
            print(f"无法读取图像: {image_file}")
            continue

        # 缩放图像以加快处理速度
        scale_factor = 1
        image = cv2.resize(image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)

        # 计算面积和生成掩码
        result_image, total_area, mask = calculate_area(image, lower_hsv, upper_hsv, min_contour_area, pixel_scale)

        # 打印总面积
        print(f"处理图像: {image_file}")
        print(f"烟丝总面积（像素单位）: {total_area / (pixel_scale ** 2):.2f} 像素")
        print(f"烟丝总实际面积: {total_area:.2f} 平方毫米")

        # 保存掩码图像和原始图像
        base_filename = os.path.splitext(image_file)[0]
        # cv2.imwrite(f'output/{base_filename}_input.jpg', image)
        cv2.imwrite(f'output/{base_filename}_mask.png', mask)
        # cv2.imwrite(f'output/{base_filename}_result.jpg', result_image)

        # 显示结果
        # show_resized_window(f"Original Image - {image_file}", image, 800, 600)
        # show_resized_window(f"Result Image - {image_file}", result_image, 800, 600)
        # cv2.waitKey(0)

    cv2.destroyAllWindows()


def show_resized_window(window_name, img, width, height):
    """显示调整大小的窗口"""
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, width, height)
    cv2.imshow(window_name, img)


# 主程序
if __name__ == "__main__":
    # 设定文件夹路径
    directory_path = 'qualityData'  # 修改为你实际的图像数据文件夹路径

    # 检查文件夹是否存在
    if not os.path.exists(directory_path):
        print(f"文件夹 {directory_path} 不存在。")
        exit()

    # 创建输出文件夹
    if not os.path.exists('output'):
        os.makedirs('output')

    # 处理文件夹中的所有图像
    process_images_in_directory(directory_path)
