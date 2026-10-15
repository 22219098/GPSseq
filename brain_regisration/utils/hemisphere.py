import cv2
import numpy as np
import matplotlib.pyplot as plt


def process_brain_image_pipeline(
    image,
    binary_threshold=10,
    canny_threshold1=50,
    canny_threshold2=150,
    hough_threshold=55,
    min_line_length=30,
    max_line_gap=5,
    parallel_offset=10,
    mirror_threshold=10,
    align_threshold=15,
    margin=2,
    target_distance=1,
    visualize=True
):
    """
    综合流程：检测对称轴 -> 镜像对齐 -> 缩放图像

    返回：
        final_image: 缩放后的图像
        symmetry_line_resized: 缩放后的对称轴线段坐标
        centroid_resized: 缩放后的质心坐标
        (left_shift, right_shift): 对齐时左右偏移量
    """
    # Step 1: 检测对称轴和质心
    symmetry_line, hemisphere = detect_symmetry_axis_and_hemisphere_side(
        image=image,
        binary_threshold=binary_threshold,
        canny_threshold1=canny_threshold1,
        canny_threshold2=canny_threshold2,
        hough_threshold=hough_threshold,
        min_line_length=min_line_length,
        max_line_gap=max_line_gap,
        parallel_offset=parallel_offset
    )

    px1, py1, px2, py2 = symmetry_line

    # Step 2: 镜像对齐
    aligned, (adj_px1, adj_py1, adj_px2, adj_py2), (cx_new, cy_new) = mirror_and_align_half_brain(
        image=image,
        px1=px1, py1=py1, px2=px2, py2=py2,
        threshold=mirror_threshold,
        align_threshold=align_threshold,
        margin=margin,
        target_distance=target_distance
    )
    # 可视化
    if visualize:
        px1_r, py1_r, px2_r, py2_r = (adj_px1, adj_py1, adj_px2, adj_py2)
        cx_r, cy_r = (cx_new, cy_new)

        plt.figure(figsize=(6, 6))
        plt.imshow(aligned, cmap='gray')
        plt.plot([px1_r, px2_r], [py1_r, py2_r], 'r-', linewidth=2, label='Symmetry Axis')
        plt.plot(cx_r, cy_r, 'bo', label='Centroid')
        plt.legend()
        plt.title('Final Processed Image (Aligned & Resized)')
        plt.axis('off')
        plt.tight_layout()
        plt.show()

    return aligned, (adj_px1, adj_py1, adj_px2, adj_py2), (cx_new, cy_new), hemisphere




def detect_symmetry_axis_and_hemisphere_side(
    image,
    binary_threshold=10,
    canny_threshold1=50,
    canny_threshold2=150,
    hough_threshold=55,
    min_line_length=30,
    max_line_gap=5,
    parallel_offset=10
):
    """
    从图像中检测分割线并生成对称轴，同时根据质心判断左右半脑。

    返回：
    - image: 原始灰度图
    - binary: 二值图
    - selected_line: 边界线段 (x1, y1, x2, y2)
    - symmetry_line: 对称轴线段 (px1, py1, px2, py2)
    - centroid: 质心坐标 (cx, cy)
    - hemisphere: 'left' or 'right'
    """

    # --- Step 1: 图像预处理 ---
    _, binary = cv2.threshold(image, binary_threshold, 1, cv2.THRESH_BINARY)
    
    # --- Step 2: 计算质心 ---
    M = cv2.moments(binary.astype(np.uint8))
    if M['m00'] == 0:
        raise ValueError("二值图像为空，无法计算质心")
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    
    # Step 3: 形态学操作增强结构
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (12, 12))  # 稍大一点效果更明显
    binary_cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
    binary_cleaned = cv2.erode(binary_cleaned, kernel, iterations=1)

    # Step 4: 边缘检测（增强后）
    edges = cv2.Canny((binary_cleaned * 255).astype(np.uint8), canny_threshold1, canny_threshold2)

    # --- Step 5: 检测直线 ---
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, hough_threshold,
                            minLineLength=min_line_length, maxLineGap=max_line_gap)
    if lines is None or len(lines) == 0:
        raise ValueError("未检测到任何线段")

    selected_line = None
    max_len = 0
    for line in lines:
        x1, y1, x2, y2 = line[0]
        length = np.hypot(x2 - x1, y2 - y1)
        if length > max_len:
            max_len = length
            selected_line = (x1, y1, x2, y2)

    # --- Step 6: 构造偏移对称轴（自动选择“远离质心”方向） ---
    def get_parallel_line(x1, y1, x2, y2, offset=30, direction='auto', centroid=None):
        dx, dy = x2 - x1, y2 - y1
        length = np.hypot(dx, dy)
        nx, ny = -dy / length, dx / length  # 法向量（垂直方向）

        # 如果是自动判断，就比较两侧与质心的距离
        if direction == 'auto' and centroid is not None:
            cx, cy = centroid

            # 构造两条线（正负偏移）
            line1 = (x1 + offset * nx, y1 + offset * ny, x2 + offset * nx, y2 + offset * ny)
            line2 = (x1 - offset * nx, y1 - offset * ny, x2 - offset * nx, y2 - offset * ny)

            # 各自中心点
            mx1, my1 = (line1[0] + line1[2]) / 2, (line1[1] + line1[3]) / 2
            mx2, my2 = (line2[0] + line2[2]) / 2, (line2[1] + line2[3]) / 2

            # 比较质心到平行线的中心距离
            dist1 = (cx - mx1)**2 + (cy - my1)**2
            dist2 = (cx - mx2)**2 + (cy - my2)**2

            selected_line = line1 if dist1 > dist2 else line2
            return tuple(map(int, selected_line))

        # 默认偏移方式
        return (
            int(x1 + offset * nx),
            int(y1 + offset * ny),
            int(x2 + offset * nx),
            int(y2 + offset * ny)
        )
    px1, py1, px2, py2 = get_parallel_line(*selected_line, offset=parallel_offset, direction='auto', centroid=(cx, cy))

        # --- Step 7: 判断垂足与象限 ---
    def foot_of_perpendicular(x0, y0, x1, y1, x2, y2):
        dx, dy = x2 - x1, y2 - y1
        if dx == dy == 0:
            return x1, y1
        t = ((x0 - x1) * dx + (y0 - y1) * dy) / (dx*dx + dy*dy)
        fx = x1 + t * dx
        fy = y1 + t * dy
        return fx, fy

    fx, fy = foot_of_perpendicular(cx, cy, px1, py1, px2, py2)
    dx = fx - cx
    dy = fy - cy

    # --- Step 8: 判定左右半脑 ---
    if dx > 0 and dy < 0:
        quadrant = 1
    elif dx < 0 and dy < 0:
        quadrant = 2
    elif dx < 0 and dy > 0:
        quadrant = 3
    elif dx > 0 and dy > 0:
        quadrant = 4
    else:
        quadrant = 0

    if quadrant in [1, 4]:
        hemisphere = 'left'
    elif quadrant in [2, 3]:
        hemisphere = 'right'
    else:
        if dx > 0:
            hemisphere = 'left'
        elif dx < 0:
            hemisphere = 'right'
        elif dy > 0:
            hemisphere = 'left'
        elif dy < 0:
            hemisphere = 'right'
        else:
            hemisphere = 'unknown'

    return (px1, py1, px2, py2), hemisphere


def mirror_and_align_half_brain(image, px1, py1, px2, py2,
                                 threshold=3, align_threshold=15, margin=2, target_distance=5):
    """
    将半脑灰度图进行镜像并自动对齐左右边缘

    参数：
        image_path: 输入图像路径（灰度图）
        px1, py1, px2, py2: 对称轴两端点坐标
        threshold: 半脑区域灰度阈值
        align_threshold: 对齐时边缘检测灰度阈值
        margin: 中心左右搜索边缘时的间距
        target_distance: 镜像后两侧应靠近对称轴的目标距离
        visualize: 是否显示可视化结果

    返回：
        aligned: 镜像+对齐后的图像
        (left_shift, right_shift): 左右平移量
        (adj_px1, adj_py1, adj_px2, adj_py2): 平移后的对称轴坐标
        (cx_new, cy_new): 平移后的质心坐标
    """

    # --- Step 1: 读取图像并生成半脑掩码 ---
    h, w = image.shape

    mask = image > threshold
    if not np.any(mask):
        raise ValueError("半脑区域为空，检查 threshold 设置是否合理")

    # 获取半脑灰度像素位置与值
    ys, xs = np.where(mask)
    coords = np.stack([xs, ys], axis=1)
    intensities = image[ys, xs]

    # --- Step 2: 计算对称轴法向量 ---
    p1 = np.array([px1, py1], dtype=np.float32)
    p2 = np.array([px2, py2], dtype=np.float32)
    line_vec = p2 - p1
    line_vec /= np.linalg.norm(line_vec)
    normal = np.array([-line_vec[1], line_vec[0]])

    # --- Step 3: 镜像坐标计算 ---
    v = coords - p1
    d = np.dot(v, normal)
    mirrored_coords = coords - 2 * np.outer(d, normal)
    mirrored_coords = np.round(mirrored_coords).astype(int)

    # --- Step 4: 图像扩展以容纳镜像 ---
    min_x, min_y = mirrored_coords.min(axis=0)
    max_x, max_y = mirrored_coords.max(axis=0)
    offset_x = max(0, -min_x)
    offset_y = max(0, -min_y)
    new_w = max(w + offset_x, max_x + 1 + offset_x)
    new_h = max(h + offset_y, max_y + 1 + offset_y)

    # 创建扩展图并绘制原图和镜像区域
    new_image = np.zeros((new_h, new_w), dtype=np.uint8)
    new_image[offset_y:offset_y + h, offset_x:offset_x + w] = image
    for (x, y), val in zip(mirrored_coords, intensities):
        nx, ny = x + offset_x, y + offset_y
        if 0 <= nx < new_w and 0 <= ny < new_h:
            new_image[ny, nx] = val

    # --- Step 5: 质心计算（用 mask） ---
    M = cv2.moments(mask.astype(np.uint8))
    if M['m00'] == 0:
        cx, cy = w // 2, h // 2  # 默认中点
    else:
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
    cx_new = cx + offset_x
    cy_new = cy + offset_y

    # --- Step 6: 镜像区域左右边缘对齐 ---
    def find_edge_distance(img, center, side='left', threshold=15, margin=2):
        h, w = img.shape
        if side == 'left':
            for d in range(center):
                col = center - d - margin
                if col < 0: break
                if np.any(img[:, col] > threshold):
                    return d
        elif side == 'right':
            for d in range(w - center):
                col = center + d + margin
                if col >= w: break
                if np.any(img[:, col] > threshold):
                    return d
        return 0

    def shift_image(img, shift, direction='right'):
        h, w = img.shape
        shifted = np.zeros_like(img)
        if direction == 'right' and shift < w:
            shifted[:, shift:] = img[:, :w - shift]
        elif direction == 'left' and shift < w:
            shifted[:, :w - shift] = img[:, shift:]
        return shifted

    center_x = int((px1 + px2) / 2 + offset_x)
    left_edge_d = find_edge_distance(new_image, center_x, 'left', align_threshold, margin)
    right_edge_d = find_edge_distance(new_image, center_x, 'right', align_threshold, margin)
    left_shift = max(0, left_edge_d - target_distance)
    right_shift = max(0, right_edge_d - target_distance)

    # 图像左右分离平移
    left_half = new_image[:, :center_x]
    right_half = new_image[:, center_x:]
    left_half_shifted = shift_image(left_half, left_shift, 'right')
    right_half_shifted = shift_image(right_half, right_shift, 'left')

    aligned = np.zeros_like(new_image)
    aligned[:, :center_x] = left_half_shifted
    aligned[:, center_x:] = right_half_shifted

    # --- 返回结果 ---
    adj_px1, adj_py1 = px1 + offset_x, py1 + offset_y
    adj_px2, adj_py2 = px2 + offset_x, py2 + offset_y
    return aligned, (adj_px1, adj_py1, adj_px2, adj_py2), (cx_new, cy_new)



def crop_half_by_centroid_and_axis(image, px1, py1, px2, py2, cx, cy, fill_value=0):
    """
    根据质心和对称轴裁剪图像，只保留质心所在一侧，另一侧用指定颜色填充
    """
    h, w = image.shape[:2]
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    coords = np.stack([xx.ravel(), yy.ravel()], axis=1)

    p1 = np.array([px1, py1], dtype=np.float32)
    p2 = np.array([px2, py2], dtype=np.float32)
    line_vec = p2 - p1
    line_vec = line_vec / np.linalg.norm(line_vec)
    normal = np.array([-line_vec[1], line_vec[0]])

    centroid_vec = np.array([cx, cy]) - p1
    centroid_side = np.dot(centroid_vec, normal)

    rel_coords = coords - p1
    d = np.dot(rel_coords, normal)

    keep_mask = d >= 0 if centroid_side >= 0 else d <= 0
    final_mask = np.zeros(h * w, dtype=np.uint8)
    final_mask[keep_mask] = 1
    final_mask = final_mask.reshape((h, w))

    if image.ndim == 3 and image.shape[2] == 3:
        final_mask_3ch = final_mask[:, :, np.newaxis]
        fill_array = np.array(fill_value, dtype=image.dtype).reshape(1, 1, 3)
        masked_image = np.where(final_mask_3ch == 1, image, fill_array)
    else:
        masked_image = np.where(final_mask == 1, image, fill_value)

    return masked_image



def mask_half_image(image: np.ndarray, side: str = 'left', fill_color: tuple = (0, 0, 0)) -> np.ndarray:
    """
    将图像的左侧或右侧一半设置为指定颜色。
    
    参数:
        image: np.ndarray，形状为 (H, W, 3) 的彩色图像
        side: str，'left' 或 'right'，指定哪一侧需要被遮挡
        fill_color: tuple，形如 (R, G, B) 的颜色值，默认是黑色 (0, 0, 0)

    返回:
        np.ndarray，修改后的图像
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("输入必须是形状为 (H, W, 3) 的彩色图像")

    if not (isinstance(fill_color, tuple) and len(fill_color) == 3):
        raise ValueError("fill_color 必须是长度为 3 的元组 (R, G, B)")

    masked_image = image.copy()
    h, w, _ = image.shape
    mid = w // 2

    if side == 'right':
        masked_image[:, :mid, :] = fill_color
    elif side == 'left':
        masked_image[:, mid:, :] = fill_color
    else:
        raise ValueError("参数 'side' 必须为 'left' 或 'right'")

    return masked_image
