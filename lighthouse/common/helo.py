def calculate_anchor_centers_and_widths(segment_length, anchors):
    results = []
    for anchor in anchors:
        # 计算起始位置
        start = segment_length - anchor
        # 计算中心
        center = (start + (start + anchor)) / 2
        # 归一化中心到0-1范围
        normalized_center = center / segment_length
        # 归一化宽度到0-1范围
        normalized_width = anchor / segment_length
        
        results.append([normalized_center, normalized_width])
    
    return results

# 示例用法
segment_length = 10
anchors = [2, 3, 5]

output = calculate_anchor_centers_and_widths(segment_length, anchors)
print(output)
for i, (center, width) in enumerate(output):
    
    print(f"Anchor {i+1}: Center = {center:.2f}, Width = {width:.2f}")
