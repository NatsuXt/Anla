import os
import math

def split_file_for_ai(file_path, output_dir="ai_context_parts", max_chars=8000):
    """
    将大文本文件拆分为 AI 友好的小片段。
    
    Args:
        file_path (str): 原始文件的绝对路径或相对路径。
        output_dir (str): 输出分割文件的目录。
        max_chars (int): 每个片段的最大字符数（建议 8000-10000 以确保安全）。
    """
    
    if not os.path.exists(file_path):
        print(f"Error: 文件未找到: {file_path}")
        return

    # 读取源文件
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except UnicodeDecodeError:
        print("Error: 文件编码读取失败，请确保文件是 UTF-8 格式。")
        return

    # 预计算总字符数和预估分片数
    total_chars = sum(len(line) for line in lines)
    estimated_parts = math.ceil(total_chars / max_chars)
    
    # 准备输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"创建输出目录: {output_dir}")

    current_part = 1
    current_chars = 0
    current_lines = []
    
    parts_created = []

    print(f"开始拆分文件... (总字符数: {total_chars}, 限制: {max_chars}/part)")

    for i, line in enumerate(lines):
        line_len = len(line)
        
        # 如果当前行加入后会超过限制，并且当前块不为空，则先保存当前块
        if current_chars + line_len > max_chars and current_lines:
            save_part(output_dir, current_part, estimated_parts, current_lines, parts_created)
            current_part += 1
            current_lines = []
            current_chars = 0
        
        current_lines.append(line)
        current_chars += line_len

    # 保存最后一个块
    if current_lines:
        save_part(output_dir, current_part, estimated_parts, current_lines, parts_created)

    print(f"\n拆分完成！共生成 {len(parts_created)} 个文件，位于: {os.path.abspath(output_dir)}")
    print("请按照文件名顺序（part_01, part_02...）依次发送给 AI。")

def save_part(output_dir, part_num, total_estimated, lines, created_list):
    """保存单个片段，添加 AI 上下文标记"""
    filename = f"part_{part_num:03d}.txt"
    filepath = os.path.join(output_dir, filename)
    
    header = f"--- [PART {part_num} START] (这是第 {part_num} 部分，请阅读并等待后续部分) ---\n\n"
    footer = f"\n\n--- [PART {part_num} END] ---"
    
    content = "".join(lines)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(header + content + footer)
    
    created_list.append(filepath)
    print(f"已生成: {filename} ({len(content)} chars)")

if __name__ == "__main__":
    # ================= 配置区域 =================
    
    # 请在这里输入您要拆分的文档路径（支持绝对路径）
    # 例如: TARGET_FILE = r"D:\Program\Anla\Logs\20260202开发总结.md" 
    # 或者直接是对话记录文本
    
    TARGET_FILE = "conversation_history.txt"  # <--- 修改这里为您的大文件名
    
    # ===========================================
    
    # 检查是否修改了文件名，如果文件不存在提示输入
    if not os.path.exists(TARGET_FILE):
        user_input = input("请输入要拆分的文件路径: ").strip()
        # 去除可能存在的引号
        TARGET_FILE = user_input.replace('"', '').replace("'", "")

    split_file_for_ai(TARGET_FILE)
