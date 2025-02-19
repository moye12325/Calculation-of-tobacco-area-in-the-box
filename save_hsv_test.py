import cv2
import numpy as np
import os

def calculate_area(image, lower_hsv, upper_hsv, min_contour_area, scale=1.0):
    """
    计算烟丝的总面积并绘制结果
    :param image: 输入图像
    :param lower_hsv: HSV 范围下限
    :param upper_hsv: HSV 范围上限
    :param min_contour_area: 最小有效面积（像素单位）
    :param scale: 每像素实际长度，单位为 mm/像素
    :return: 绘制结果的图像，总面积
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

    # 计算实际面积（平方毫米）
    total_actual_area = total_area * (scale ** 2)

    # 显示总面积
    cv2.putText(image_with_contours, f"Total Area: {total_actual_area:.2f} mm^2",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    return image_with_contours, total_actual_area

# 主程序
if __name__ == "__main__":
    import os

    # 图像文件夹路径
    img_folder = './imgs'

    # 获取文件夹中所有图片文件
    img_files = [f for f in os.listdir(img_folder) if f.endswith(('.jpg', '.png', '.jpeg'))]

    # HSV 范围和最小轮廓面积
    lower_hsv = np.array([9, 50, 50])  # 烟丝颜色的 HSV 下限
    upper_hsv = np.array([25, 255, 255])  # 烟丝颜色的 HSV 上限
    min_contour_area = 500  # 最小有效面积（像素单位）

    # 每像素的实际长度（例如 0.1 mm/像素）
    pixel_scale = 0.1  # 单位：mm/像素

    # 前缀
    prefix = 'processed_'

    # 创建processed文件夹（如果不存在）
    output_folder = os.path.join(img_folder, 'processed')
    os.makedirs(output_folder, exist_ok=True)

    # 循环处理每个图像
    for img_file in img_files:
        # 读取图像
        image_path = os.path.join(img_folder, img_file)
        image = cv2.imread(image_path)
        if image is None:
            print(f"无法读取图像 {img_file}，请检查路径。")
            continue

        # 缩放图像以加快处理速度
        scale_factor = 0.5
        image = cv2.resize(image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)

        # 计算面积
        result_image, total_area = calculate_area(image, lower_hsv, upper_hsv, min_contour_area, pixel_scale)

        # 打印总面积
        print(f"图像 {img_file} 的烟丝总面积（像素单位）: {total_area / (pixel_scale ** 2):.2f} 像素")
        print(f"图像 {img_file} 的烟丝总实际面积: {total_area:.2f} 平方毫米")

        # 保存处理后的图像到processed文件夹
        output_path = os.path.join(output_folder, prefix + img_file)
        cv2.imwrite(output_path, result_image)

    print("处理完成。")