"""
TTS 批量处理脚本
功能：读取 input.txt 文件，根据配置选择逐行生成语音文件或整文本生成一个语音文件，并生成处理报告
"""

import os
import re
import time
import wave
from datetime import datetime
from indextts.infer_v2 import IndexTTS2


class TTSBatchProcessor:
    """TTS 批量处理器类"""
    
    def __init__(self, config_path="checkpoints/config.yaml", 
                 model_dir="checkpoints", 
                 spk_audio_prompt='examples/charlie_munger_voice_01.MP3',
                 use_fp16=True, 
                 use_cuda_kernel=False, 
                 use_deepspeed=False,
                 start_line_num=1,
                 read_by_line=True):
        """
        初始化 TTS 批量处理器
        
        参数:
            config_path: TTS 模型配置文件路径
            model_dir: TTS 模型目录
            spk_audio_prompt: 说话人音频提示文件路径
            use_fp16: 是否使用 FP16 精度
            use_cuda_kernel: 是否使用 CUDA 内核
            use_deepspeed: 是否使用 DeepSpeed
            start_line_num: 输出文件的起始编号，默认为1
            read_by_line: 是否按行读取，True为逐行生成音频文件，False为整文本生成一个音频文件
        """
        # 初始化 TTS 模型
        self.tts = IndexTTS2(
            cfg_path=config_path, 
            model_dir=model_dir, 
            use_fp16=use_fp16, 
            use_cuda_kernel=use_cuda_kernel, 
            use_deepspeed=use_deepspeed
        )
        
        # 说话人音频提示文件路径
        self.spk_audio_prompt = spk_audio_prompt
        
        # 获取当前脚本所在目录
        self.current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # 设置输入输出路径
        self.input_file = os.path.join(self.current_dir, "input.txt")
        self.output_dir = os.path.join(self.current_dir, "outputs")
        
        # 确保输出目录存在
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 创建日志目录和日志文件路径
        self.log_dir = os.path.join(self.current_dir, "logs")
        os.makedirs(self.log_dir, exist_ok=True)
        self.log_file = os.path.join(
            self.log_dir, 
            f"tts_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        )
        
        # 初始化处理统计变量
        self.total_char_count = 0
        self.processing_records = []
        self.total_start_time = None
        self.start_line_num = start_line_num
        self.read_by_line = read_by_line
        
    def log_print(self, message, end='\n'):
        """
        日志输出函数：同时输出到控制台和日志文件
        
        参数:
            message: 要输出的消息
            end: 行结束符，默认为换行符
        """
        print(message, end=end)
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(message + end if end == '\n' else message)
    
    @staticmethod
    def sanitize_filename(text):
        """
        清理文件名中的非法字符
        
        参数:
            text: 原始文本
            
        返回:
            清理后的文本
        """
        # 移除或替换文件名中的非法字符
        text = re.sub(r'[<>:"/\\|?*]', '', text)
        # 移除首尾空格
        text = text.strip()
        return text
    
    def get_audio_duration(self, audio_path):
        """
        获取 WAV 音频文件的时长（秒）
        
        参数:
            audio_path: 音频文件路径
            
        返回:
            音频时长（秒），如果读取失败则返回 0.0
        """
        try:
            with wave.open(audio_path, 'r') as wav_file:
                frames = wav_file.getnframes()
                sample_rate = wav_file.getframerate()
                duration = frames / float(sample_rate)
                return duration
        except Exception as e:
            self.log_print(f"警告: 无法读取音频文件 {audio_path} 的时长: {e}")
            return 0.0
    
    def generate_output_filename(self, line_num, text, is_full_text=False):
        """
        生成输出文件名
        
        参数:
            line_num: 行号（整文本模式下不使用）
            text: 文本内容
            is_full_text: 是否为整文本模式
            
        返回:
            输出文件名和完整路径
        """
        # 截取前10个字符作为文件名的一部分
        text_prefix = text[:10]
        # 清理文件名
        safe_prefix = self.sanitize_filename(text_prefix)
        
        if is_full_text:
            # 整文本模式：使用时间戳 + 文本前缀
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_filename = f"full_text_{timestamp}_{safe_prefix}.wav"
        else:
            # 按行模式：行号 + 前10个字符
            output_filename = f"{line_num}_{safe_prefix}.wav"
        
        output_path = os.path.join(self.output_dir, output_filename)
        return output_filename, output_path
    
    def process_single_line(self, line_num, text):
        """
        处理单行文本，生成对应的音频文件
        
        参数:
            line_num: 行号
            text: 文本内容
            
        返回:
            处理记录字典，包含行号、字符数、处理时间、音频时长等信息
        """
        # 生成输出文件名
        output_filename, output_path = self.generate_output_filename(line_num, text, is_full_text=False)
        
        # 统计当前行的字符数量（包括标点符号）
        char_count = len(text)
        # 累计总字符数
        self.total_char_count += char_count
        
        # 输出处理信息
        self.log_print(f"正在处理第 {line_num} 行: {text[:50]}...字符数量: {char_count} 个")
        self.log_print(f"输出文件: {output_filename}")
        
        # 记录当前行开始处理的时间
        line_start_time = time.time()
        
        # 调用 TTS 生成音频
        self.tts.infer(
            spk_audio_prompt=self.spk_audio_prompt,
            text=text,
            output_path=output_path,
            verbose=True
        )
        
        # 计算当前行处理时间
        line_end_time = time.time()
        line_elapsed = line_end_time - line_start_time
        
        # 获取生成的音频文件时长
        audio_duration = self.get_audio_duration(output_path)
        
        # 构建处理记录
        record = {
            'line_num': line_num,
            'char_count': char_count,
            'elapsed_time': line_elapsed,
            'audio_duration': audio_duration
        }
        
        # 输出完成信息
        self.log_print(f"已完成: {output_filename} 处理时间: {line_elapsed:.2f} 秒\n")
        
        return record
    
    def process_full_text(self, text):
        """
        处理整文本，生成一个完整的音频文件
        
        参数:
            text: 完整文本内容
            
        返回:
            处理记录字典，包含字符数、处理时间、音频时长等信息
        """
        # 生成输出文件名
        output_filename, output_path = self.generate_output_filename(0, text, is_full_text=True)
        
        # 统计字符数量
        char_count = len(text)
        self.total_char_count = char_count
        
        # 输出处理信息
        self.log_print(f"正在处理完整文本: {text[:50]}...字符数量: {char_count} 个")
        self.log_print(f"输出文件: {output_filename}")
        
        # 记录开始处理的时间
        start_time = time.time()
        
        # 调用 TTS 生成音频
        self.tts.infer(
            spk_audio_prompt=self.spk_audio_prompt,
            text=text,
            output_path=output_path,
            verbose=True
        )
        
        # 计算处理时间
        end_time = time.time()
        elapsed = end_time - start_time
        
        # 获取生成的音频文件时长
        audio_duration = self.get_audio_duration(output_path)
        
        # 构建处理记录
        record = {
            'line_num': '完整文本',
            'char_count': char_count,
            'elapsed_time': elapsed,
            'audio_duration': audio_duration
        }
        
        # 输出完成信息
        self.log_print(f"已完成: {output_filename} 处理时间: {elapsed:.2f} 秒\n")
        
        return record
    
    def print_log_header(self):
        """输出日志文件头部信息"""
        self.log_print(f"=== TTS 批量处理开始 === 时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.log_print(f"输入文件: {self.input_file}")
        self.log_print(f"输出目录: {self.output_dir}")
        self.log_print(f"日志文件: {self.log_file}")
        self.log_print("")
    
    def print_summary(self, total_elapsed):
        """
        输出处理总结信息
        
        参数:
            total_elapsed: 总处理时间（秒）
        """
        total_minutes = int(total_elapsed // 60)
        total_seconds = int(total_elapsed % 60)
        
        self.log_print("")
        self.log_print(f"所有音频文件生成完成！总字符数: {self.total_char_count} 个，总处理时间: {total_minutes} 分 {total_seconds} 秒")
        self.log_print(f"=== TTS 批量处理结束 === 时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.log_print("")
    
    def generate_report(self, total_elapsed):
        """
        生成并输出处理报告
        
        参数:
            total_elapsed: 总处理时间（秒）
        """
        # 定义各列的固定宽度
        COL_WIDTH_LINE_NUM = 12      # 行号列宽度
        COL_WIDTH_CHAR_COUNT = 15    # 字符数列宽度
        COL_WIDTH_AUDIO_TIME = 20    # 音频时间列宽度
        COL_WIDTH_ELAPSED_TIME = 20  # 消耗时间列宽度
        COL_WIDTH_CHAR_RATIO = 16    # 字符比率列宽度
        COL_WIDTH_TIME_RATIO = 16     # 时间比率列宽度
        
        # 计算总宽度（各列宽度之和）
        total_width = (COL_WIDTH_LINE_NUM + COL_WIDTH_CHAR_COUNT + COL_WIDTH_AUDIO_TIME + 
                      COL_WIDTH_ELAPSED_TIME + COL_WIDTH_CHAR_RATIO + COL_WIDTH_TIME_RATIO)
        
        # 输出报告标题
        self.log_print("=" * total_width)
        self.log_print("处理报告".center(total_width))
        self.log_print("=" * total_width)
        
        # 输出表头（使用统一的列宽度格式化）
        header_line = (
            f"{'行号':<{COL_WIDTH_LINE_NUM}}"
            f"{'字符数':<{COL_WIDTH_CHAR_COUNT}}"
            f"{'音频时间(秒)':<{COL_WIDTH_AUDIO_TIME}}"
            f"{'消耗时间(秒)':<{COL_WIDTH_ELAPSED_TIME}}"
            f"{'字符比率':<{COL_WIDTH_CHAR_RATIO}}"
            f"{'时间比率':<{COL_WIDTH_TIME_RATIO}}"
        )
        self.log_print(header_line)
        self.log_print("-" * total_width)
        
        # 计算总音频时长并输出每行数据
        total_audio_duration = 0.0
        for record in self.processing_records:
            # 计算字符比率（消耗时间/字符数）
            ratio = record['elapsed_time'] / record['char_count'] if record['char_count'] > 0 else 0.0
            # 计算时间比率（消耗时间/音频时间）
            time_ratio = record['elapsed_time'] / record['audio_duration'] if record['audio_duration'] > 0 else 0.0
            # 累计总音频时长
            total_audio_duration += record['audio_duration']
            # 输出单行记录（使用统一的列宽度格式化）
            data_line = (
                f"{record['line_num']:<{COL_WIDTH_LINE_NUM}}"
                f"{record['char_count']:<{COL_WIDTH_CHAR_COUNT}}"
                f"{record['audio_duration']:<{COL_WIDTH_AUDIO_TIME}.2f}"
                f"{record['elapsed_time']:<{COL_WIDTH_ELAPSED_TIME}.2f}"
                f"{ratio:<{COL_WIDTH_CHAR_RATIO}.2f}"
                f"{time_ratio:<{COL_WIDTH_TIME_RATIO}.2f}"
            )
            self.log_print(data_line)
        
        # 输出总计行
        self.log_print("-" * total_width)
        total_ratio = total_elapsed / self.total_char_count if self.total_char_count > 0 else 0.0
        total_time_ratio = total_elapsed / total_audio_duration if total_audio_duration > 0 else 0.0
        total_line = (
            f"{'总计':<{COL_WIDTH_LINE_NUM}}"
            f"{self.total_char_count:<{COL_WIDTH_CHAR_COUNT}}"
            f"{total_audio_duration:<{COL_WIDTH_AUDIO_TIME}.2f}"
            f"{total_elapsed:<{COL_WIDTH_ELAPSED_TIME}.2f}"
            f"{total_ratio:<{COL_WIDTH_CHAR_RATIO}.2f}"
            f"{total_time_ratio:<{COL_WIDTH_TIME_RATIO}.2f}"
        )
        self.log_print(total_line)
        self.log_print("=" * total_width)
    
    def process_batch(self):
        """
        批量处理主函数：根据 read_by_line 参数决定处理方式
        """
        # 记录总开始时间
        self.total_start_time = time.time()
        
        # 输出日志头部信息
        self.print_log_header()
        
        if self.read_by_line:
            # 按行读取模式：逐行处理，每行生成一个音频文件
            self.log_print("处理模式: 按行读取（每行生成一个音频文件）")
            self.log_print("")
            
            with open(self.input_file, 'r', encoding='utf-8') as f:
                line_counter = 0
                for line in f:
                    text = line.strip()
                    if not text:  # 跳过空行
                        continue
                    
                    line_counter += 1
                    # 计算实际输出文件编号（从start_line_num开始）
                    output_line_num = self.start_line_num + (line_counter - 1)
                    
                    # 处理单行文本
                    record = self.process_single_line(output_line_num, text)
                    # 保存处理记录
                    self.processing_records.append(record)
        else:
            # 整文本读取模式：读取整个文件，忽略换行，生成一个音频文件
            self.log_print("处理模式: 整文本读取（生成一个完整音频文件）")
            self.log_print("")
            
            # 读取整个文件内容，忽略换行回车
            with open(self.input_file, 'r', encoding='utf-8') as f:
                lines = []
                for line in f:
                    stripped_line = line.strip()
                    if stripped_line:  # 只添加非空行
                        lines.append(stripped_line)
                
                # 将所有行合并成一个文本（用空格连接，也可以不用分隔符）
                full_text = ' '.join(lines)
            
            if full_text:
                # 处理整文本
                record = self.process_full_text(full_text)
                # 保存处理记录
                self.processing_records.append(record)
            else:
                self.log_print("警告: 输入文件为空，无法生成音频文件")
        
        # 计算总处理时间
        total_end_time = time.time()
        total_elapsed = total_end_time - self.total_start_time
        
        # 输出处理总结
        self.print_summary(total_elapsed)
        
        # 生成并输出处理报告
        self.generate_report(total_elapsed)
        
        return total_elapsed


def main():
    """
    主函数：程序入口点
    """
    # 定义是否按行读取
    # True: 每读取一行，生成一个对应的音频文件
    # False: 读取整个输入文本，忽略换行回车，生成一个完整的大音频文件
    read_by_line = False
    
    # 定义输出文件的起始编号，初始值为1
    # 如果改为10，则第一行文本输出的文件名前缀从10开始编号，依次类推
    # 注意：仅在 read_by_line=True 时有效
    start_line_num = 1
    
    # 创建 TTS 批量处理器实例
    processor = TTSBatchProcessor(
        config_path="checkpoints/config.yaml",
        model_dir="checkpoints",
        spk_audio_prompt='examples/charlie_munger_voice_01.MP3',
        use_fp16=False,
        use_cuda_kernel=False,
        use_deepspeed=False,
        start_line_num=start_line_num,
        read_by_line=read_by_line
    )
    
    # 执行批量处理
    total_elapsed = processor.process_batch()
    
    # 返回总处理时间（可选，用于进一步处理）
    return total_elapsed


if __name__ == "__main__":
    main()
