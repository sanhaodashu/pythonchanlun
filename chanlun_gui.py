# chanlun_gui.py - 缠论分析GUI界面（布局优化版）
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import pandas as pd
from datetime import datetime
import sys
import os

# 导入缠论处理模块
try:
    import chanlun_processor as processor
    import chanlun_huitu as huitu
    PROCESSOR_AVAILABLE = True
    HUITU_AVAILABLE = True
except ImportError as e:
    print(f"导入模块失败: {e}")
    PROCESSOR_AVAILABLE = False
    HUITU_AVAILABLE = False

class ChanlunGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("缠论分析系统")
        self.root.geometry("1200x900")  # 增大默认窗口尺寸
        self.root.minsize(1000, 700)   # 增大最小窗口尺寸
        
        # 日志缓冲区，减少刷屏
        self.log_buffer = []
        self.last_log_update = 0
        self.log_update_interval = 0.1  # 100ms更新一次
        
        # 显示选项变量
        self.display_options = {
            'fractals': tk.BooleanVar(value=True),    # 分型
            'bi_lines': tk.BooleanVar(value=True),     # 笔
            'line_segments': tk.BooleanVar(value=True), # 线段
            'centers': tk.BooleanVar(value=True),       # 中枢
            'buy_points': tk.BooleanVar(value=True),   # 买点
            'sell_points': tk.BooleanVar(value=True)   # 卖点
        }
        
        # 创建界面
        self.create_widgets()
        
        # 初始化状态
        self.is_processing = False
        
    def create_widgets(self):
        # 主框架
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 配置网格权重 - 让图表区域获得更多空间
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(8, weight=3)  # 图表区域权重更大
        main_frame.rowconfigure(7, weight=1)  # 日志区域权重
        
        # 标题
        title_label = ttk.Label(main_frame, text="缠论分析系统", font=("微软雅黑", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=4, pady=(0, 10), sticky=(tk.W, tk.E))
        
        # 股票代码输入（支持纯数字和带前缀）
        code_frame = ttk.Frame(main_frame)
        code_frame.grid(row=1, column=0, columnspan=4, sticky=(tk.W, tk.E), pady=(0, 10))
        code_frame.columnconfigure(1, weight=1)
        
        ttk.Label(code_frame, text="股票代码:").grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        self.stock_code_var = tk.StringVar()
        self.stock_code_entry = ttk.Entry(code_frame, textvariable=self.stock_code_var)
        self.stock_code_entry.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(0, 5))
        self.stock_code_entry.bind('<FocusOut>', self.normalize_stock_code)  # 失去焦点时标准化股票代码
        
        # 线程数设置
        thread_frame = ttk.Frame(main_frame)
        thread_frame.grid(row=2, column=0, columnspan=4, sticky=(tk.W, tk.E), pady=(0, 10))
        
        ttk.Label(thread_frame, text="线程数:").grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        self.thread_count_var = tk.StringVar(value="4")
        thread_spinbox = ttk.Spinbox(thread_frame, from_=1, to=16, width=5, textvariable=self.thread_count_var)
        thread_spinbox.grid(row=0, column=1, sticky=tk.W, padx=(0, 10))
        
        # 功能选择
        func_frame = ttk.LabelFrame(main_frame, text="分析功能", padding="10")
        func_frame.grid(row=3, column=0, columnspan=4, sticky=(tk.W, tk.E), pady=(0, 10))
        func_frame.columnconfigure(0, weight=1)
        
        self.func_var = tk.StringVar()
        self.func_combo = ttk.Combobox(func_frame, textvariable=self.func_var, state="readonly")
        self.func_combo['values'] = [
            "1. 包含关系处理(单个)",
            "1.1 包含关系处理(批量)", 
            "2. 分型识别(单个)", 
            "2.1 分型识别(批量)",
            "3. 笔划分(单个)",
            "4. 线段识别(单个)",
            "5. 中枢构建(单个)",
            "6. 买卖点识别(单个)"
        ]
        self.func_combo.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=(0, 5))
        self.func_combo.set("1. 包含关系处理(单个)")
        
        # 绑定选择事件
        self.func_var.trace('w', self.on_function_change)
        
        # 显示选项框架
        options_frame = ttk.LabelFrame(main_frame, text="显示选项", padding="5")
        options_frame.grid(row=4, column=0, columnspan=4, sticky=(tk.W, tk.E), pady=(0, 10))
        options_frame.columnconfigure((0,1,2,3,4,5), weight=1)
        
        # 分型选项
        ttk.Checkbutton(options_frame, text="分型", 
                       variable=self.display_options['fractals']).grid(row=0, column=0, padx=(0, 15), sticky=tk.W)
        
        # 笔选项
        ttk.Checkbutton(options_frame, text="笔", 
                       variable=self.display_options['bi_lines']).grid(row=0, column=1, padx=(0, 15), sticky=tk.W)
        
        # 线段选项
        ttk.Checkbutton(options_frame, text="线段", 
                       variable=self.display_options['line_segments']).grid(row=0, column=2, padx=(0, 15), sticky=tk.W)
        
        # 中枢选项
        ttk.Checkbutton(options_frame, text="中枢", 
                       variable=self.display_options['centers']).grid(row=0, column=3, padx=(0, 15), sticky=tk.W)
        
        # 买点选项
        ttk.Checkbutton(options_frame, text="买点", 
                       variable=self.display_options['buy_points']).grid(row=0, column=4, padx=(0, 15), sticky=tk.W)
        
        # 卖点选项
        ttk.Checkbutton(options_frame, text="卖点", 
                       variable=self.display_options['sell_points']).grid(row=0, column=5, padx=(0, 15), sticky=tk.W)
        
        # 控制按钮框架
        self.button_frame = ttk.Frame(main_frame)
        self.button_frame.grid(row=5, column=0, columnspan=4, pady=(0, 10), sticky=(tk.W, tk.E))
        
        self.process_btn = ttk.Button(self.button_frame, text="开始分析", command=self.start_analysis)
        self.process_btn.grid(row=0, column=0, padx=(0, 5))
        self.stop_btn = ttk.Button(self.button_frame, text="停止", command=self.stop_analysis, state="disabled")
        self.stop_btn.grid(row=0, column=1, padx=(0, 5))
        
        # 添加显示图表按钮
        self.chart_btn = ttk.Button(self.button_frame, text="显示K线图", command=self.show_chart)
        self.chart_btn.grid(row=0, column=2, padx=(10, 0))
        
        # 进度条
        self.progress_var = tk.IntVar()
        self.progress_bar = ttk.Progressbar(self.button_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.grid(row=0, column=3, padx=(10, 0), sticky=(tk.W, tk.E))
        self.button_frame.columnconfigure(3, weight=1)
        
        # 日志显示区域
        log_label = ttk.Label(main_frame, text="分析日志:")
        log_label.grid(row=6, column=0, columnspan=4, sticky=(tk.W), pady=(0, 5))
        
        log_frame = ttk.Frame(main_frame)
        log_frame.grid(row=7, column=0, columnspan=4, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        
        # 创建带滚动条的日志文本框
        self.log_text = tk.Text(log_frame, height=8, wrap=tk.WORD)  # 减少高度
        log_scrollbar_y = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=self.log_text.yview)
        log_scrollbar_x = ttk.Scrollbar(log_frame, orient=tk.HORIZONTAL, command=self.log_text.xview)
        self.log_text.configure(yscrollcommand=log_scrollbar_y.set, xscrollcommand=log_scrollbar_x.set)
        
        self.log_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        log_scrollbar_y.grid(row=0, column=1, sticky=(tk.N, tk.S))
        log_scrollbar_x.grid(row=1, column=0, sticky=(tk.W, tk.E))
        
        # 图表显示区域（优化版，更大的默认空间）
        self.chart_frame_container = ttk.LabelFrame(main_frame, text="K线图表", padding="5")
        self.chart_frame_container.grid(row=8, column=0, columnspan=4, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        self.chart_frame_container.grid_remove()  # 初始隐藏
        self.chart_frame_container.columnconfigure(0, weight=1)
        self.chart_frame_container.rowconfigure(0, weight=1)
        
        # 创建可滚动的图表容器
        self.chart_canvas = tk.Canvas(self.chart_frame_container)
        self.chart_scrollbar_y = ttk.Scrollbar(self.chart_frame_container, orient=tk.VERTICAL, command=self.chart_canvas.yview)
        self.chart_scrollbar_x = ttk.Scrollbar(self.chart_frame_container, orient=tk.HORIZONTAL, command=self.chart_canvas.xview)
        
        self.chart_inner_frame = ttk.Frame(self.chart_canvas)
        self.chart_inner_frame.bind("<Configure>", self.on_chart_frame_configure)
        
        self.chart_canvas.create_window((0, 0), window=self.chart_inner_frame, anchor="nw")
        self.chart_canvas.configure(yscrollcommand=self.chart_scrollbar_y.set, xscrollcommand=self.chart_scrollbar_x.set)
        
        self.chart_canvas.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.chart_scrollbar_y.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.chart_scrollbar_x.grid(row=1, column=0, sticky=(tk.W, tk.E))
        
        # 状态栏
        self.status_var = tk.StringVar()
        self.status_var.set("就绪")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN)
        status_bar.grid(row=9, column=0, columnspan=4, sticky=(tk.W, tk.E))
        
    def on_chart_frame_configure(self, event=None):
        """当图表框架大小改变时调整Canvas滚动区域"""
        self.chart_canvas.configure(scrollregion=self.chart_canvas.bbox("all"))
        
    def normalize_stock_code(self, event=None):
        """标准化股票代码（自动添加SH#或SZ#前缀）"""
        stock_code = self.stock_code_var.get().strip()
        if not stock_code:
            return
            
        # 如果已经包含前缀，不做处理
        if stock_code.startswith(('SH#', 'SZ#')):
            return
            
        # 如果是纯数字，自动添加前缀
        if stock_code.isdigit():
            # 根据股票代码判断是上海还是深圳
            if len(stock_code) == 6:
                if stock_code.startswith(('6', '9')):  # 6开头或9开头是上海
                    normalized_code = f"SH#{stock_code}"
                elif stock_code.startswith(('0', '3', '2')):  # 0、3、2开头是深圳
                    normalized_code = f"SZ#{stock_code}"
                else:
                    # 默认上海
                    normalized_code = f"SH#{stock_code}"
                self.stock_code_var.set(normalized_code)
            else:
                # 非6位数字，保持原样
                pass
        else:
            # 包含字母或其他字符，保持原样
            pass
        
    def on_function_change(self, *args):
        """功能选择变化时的回调"""
        selected = self.func_var.get()
        # 如果是批量处理，禁用股票代码输入
        if "批量" in selected:
            self.stock_code_entry.config(state="disabled")
        else:
            self.stock_code_entry.config(state="normal")
        
    def start_analysis(self):
        """开始分析"""
        if self.is_processing:
            messagebox.showwarning("警告", "正在处理中，请等待完成")
            return
            
        selected_func = self.func_var.get()
        
        # 设置线程数
        try:
            thread_count = int(self.thread_count_var.get())
            if PROCESSOR_AVAILABLE:
                processor.set_max_workers(thread_count)
                self.log_message(f"设置线程数为: {thread_count}")
        except ValueError:
            self.log_message("线程数设置无效，使用默认值")
        
        # 解析功能
        if "批量" in selected_func:
            # 批量处理，不需要股票代码
            stock_code = None
            func_type = "batch"
        else:
            # 单个处理，需要股票代码
            stock_code = self.stock_code_var.get().strip()
            # 标准化股票代码
            if stock_code and not stock_code.startswith(('SH#', 'SZ#')) and stock_code.isdigit():
                self.normalize_stock_code()
                stock_code = self.stock_code_var.get().strip()
                
            if not stock_code:
                messagebox.showerror("错误", "请输入股票代码")
                return
            func_type = "single"
        
        # 解析功能索引
        func_index = 1  # 默认包含关系
        if "分型" in selected_func:
            func_index = 2
        elif "笔" in selected_func:
            func_index = 3
        elif "线段" in selected_func:
            func_index = 4
        elif "中枢" in selected_func:
            func_index = 5
        elif "买卖点" in selected_func:
            func_index = 6
        
        # 确认开始处理
        confirm_msg = f"确定要进行{selected_func}分析吗？"
        if func_type == "single":
            confirm_msg = f"确定要对股票 {stock_code} 进行{selected_func}分析吗？"
        
        if not messagebox.askyesno("确认", confirm_msg):
            return
            
        # 开始处理
        self.is_processing = True
        self.process_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self.status_var.set("正在分析...")
        self.progress_var.set(0)
        
        # 清空日志
        self.log_text.delete(1.0, tk.END)
        self.log_buffer = []
        
        # 启动处理线程
        if func_type == "single":
            self.process_thread = threading.Thread(
                target=self.process_analysis, 
                args=(stock_code, func_index)
            )
        else:
            self.process_thread = threading.Thread(
                target=self.process_batch_analysis, 
                args=(func_index,)
            )
            
        self.process_thread.daemon = True
        self.process_thread.start()
        
    def stop_analysis(self):
        """停止分析"""
        if self.is_processing:
            # 设置取消标志
            if PROCESSOR_AVAILABLE:
                processor.cancel_processing()
            self.log_message("⚠️ 正在请求停止分析...")
            self.stop_btn.config(state="disabled")
            self.status_var.set("正在停止...")
            
    def process_analysis(self, stock_code, func_index):
        """处理分析的线程函数（单个股票）"""
        try:
            if not PROCESSOR_AVAILABLE:
                self.log_message("❌ 处理模块不可用，请检查安装")
                return
                
            def progress_callback(progress):
                # 确保在主线程中更新UI
                self.root.after(0, lambda: self.progress_var.set(progress))
                
            def log_callback(message):
                self.log_message(message)
                
            # 根据功能索引调用不同的处理函数
            if func_index == 1:
                success, message = processor.process_inclusion_relation(
                    stock_code, 
                    progress_callback=progress_callback,
                    log_callback=log_callback
                )
            elif func_index == 2:
                success, message = processor.process_fractals(
                    stock_code, 
                    progress_callback=progress_callback,
                    log_callback=log_callback
                )
            else:
                success, message = False, "该功能尚未实现"
            
            if success:
                self.root.after(0, lambda: self.log_message("✅ 分析完成"))
            else:
                self.root.after(0, lambda: self.log_message("⚠️ 分析失败或被中断"))
                
        except Exception as e:
            self.root.after(0, lambda: self.log_message(f"❌ 分析过程中发生错误: {str(e)}"))
        finally:
            self.is_processing = False
            self.root.after(0, lambda: self.process_btn.config(state="normal"))
            self.root.after(0, lambda: self.stop_btn.config(state="disabled"))
            self.root.after(0, lambda: self.status_var.set("分析完成"))
            # 刷新剩余日志
            self.flush_log_buffer()
            
    def process_batch_analysis(self, func_index):
        """批量处理分析"""
        try:
            if not PROCESSOR_AVAILABLE:
                self.log_message("❌ 处理模块不可用，请检查安装")
                return
                
            def progress_callback(progress):
                self.root.after(0, lambda: self.progress_var.set(progress))
                
            def log_callback(message):
                self.log_message(message)
                
            # 根据功能索引调用不同的批量处理函数
            if func_index == 1:  # 批量包含关系处理
                success, message = processor.batch_process_all_stocks(
                    "inclusion",
                    progress_callback=progress_callback,
                    log_callback=log_callback
                )
            elif func_index == 2:  # 批量分型识别
                success, message = processor.batch_process_all_stocks(
                    "fractal",
                    progress_callback=progress_callback,
                    log_callback=log_callback
                )
            else:
                success, message = False, "该批量功能尚未实现"
            
            if success:
                self.root.after(0, lambda: self.log_message("✅ 批量分析完成"))
            else:
                self.root.after(0, lambda: self.log_message("⚠️ 批量分析失败或被中断"))
                
        except Exception as e:
            self.root.after(0, lambda: self.log_message(f"❌ 批量分析过程中发生错误: {str(e)}"))
        finally:
            self.is_processing = False
            self.root.after(0, lambda: self.process_btn.config(state="normal"))
            self.root.after(0, lambda: self.stop_btn.config(state="disabled"))
            self.root.after(0, lambda: self.status_var.set("分析完成"))
            # 刷新剩余日志
            self.flush_log_buffer()
            
    def show_chart(self):
        """显示K线图（调用绘图模块）"""
        if not HUITU_AVAILABLE:
            messagebox.showerror("错误", "绘图模块不可用，请检查安装")
            return
            
        # 标准化股票代码
        stock_code = self.stock_code_var.get().strip()
        if stock_code and not stock_code.startswith(('SH#', 'SZ#')) and stock_code.isdigit():
            self.normalize_stock_code()
            stock_code = self.stock_code_var.get().strip()
            
        if not stock_code:
            messagebox.showerror("错误", "请输入股票代码")
            return
            
        try:
            # 准备显示选项
            display_options = {
                'fractals': self.display_options['fractals'].get()
            }
            
            # 调用绘图模块显示图表
            fig, message = huitu.show_stock_chart(stock_code, display_options, 100)
            
            if fig is None:
                messagebox.showerror("错误", message)
                return
            
            # 显示图表框架
            self.chart_frame_container.grid()
            
            # 清除之前的图表
            for widget in self.chart_inner_frame.winfo_children():
                widget.destroy()
            
            # 嵌入到tkinter中
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
            canvas = FigureCanvasTkAgg(fig, self.chart_inner_frame)
            canvas.draw()
            
            # 获取图表的实际尺寸
            chart_widget = canvas.get_tk_widget()
            chart_widget.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
            
            # 添加关闭按钮
            close_btn = ttk.Button(self.chart_inner_frame, text="关闭图表", command=self.hide_chart)
            close_btn.grid(row=1, column=0, pady=(10, 0))
            
            # 更新Canvas的滚动区域
            self.root.after(100, self.update_chart_scroll_region)
            
            # 自动调整窗口大小以适应图表
            self.root.after(200, self.adjust_window_for_chart)
            
        except Exception as e:
            import traceback
            error_msg = f"显示图表时出错: {str(e)}\n{traceback.format_exc()}"
            print(error_msg)
            messagebox.showerror("错误", f"显示图表时出错: {str(e)}")
    
    def adjust_window_for_chart(self):
        """调整窗口以更好地显示图表"""
        # 确保窗口大小适合显示图表
        current_width = self.root.winfo_width()
        current_height = self.root.winfo_height()
        
        # 如果窗口太小，适当增大
        if current_width < 1200:
            self.root.geometry(f"1200x{current_height}")
        if current_height < 800:
            self.root.geometry(f"{max(current_width, 1200)}x800")
    
    def update_chart_scroll_region(self):
        """更新图表Canvas的滚动区域"""
        self.chart_canvas.configure(scrollregion=self.chart_canvas.bbox("all"))
    
    def hide_chart(self):
        """隐藏图表"""
        self.chart_frame_container.grid_remove()
    
    def log_message(self, message):
        """添加日志消息（带缓冲，减少刷屏）"""
        import time
        current_time = time.time()
        
        # 将消息添加到缓冲区
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted_message = f"[{timestamp}] {message}\n"
        self.log_buffer.append(formatted_message)
        
        # 如果缓冲区太大或时间间隔到了，刷新显示
        if (len(self.log_buffer) >= 5 or 
            current_time - self.last_log_update > self.log_update_interval):
            self.flush_log_buffer()
            self.last_log_update = current_time
        else:
            # 延迟刷新
            self.root.after(int(self.log_update_interval * 1000), self.flush_log_buffer)
    
    def flush_log_buffer(self):
        """刷新日志缓冲区"""
        if self.log_buffer:
            # 批量更新UI
            messages = ''.join(self.log_buffer)
            self.log_text.insert(tk.END, messages)
            self.log_text.see(tk.END)
            self.log_buffer.clear()
            # 强制更新UI
            self.root.update_idletasks()

def main():
    root = tk.Tk()
    app = ChanlunGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()