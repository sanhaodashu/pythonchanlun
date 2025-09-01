# stock_data_gui.py - 股票数据处理GUI界面
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import os
from datetime import datetime
import sys
import importlib

# 尝试导入处理模块
try:
    import stock_data_processor_optimized as processor
    PROCESSOR_AVAILABLE = True
except ImportError as e:
    print(f"导入处理模块失败: {e}")
    PROCESSOR_AVAILABLE = False

class StockDataGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("股票数据处理系统")
        self.root.geometry("950x800")
        self.root.minsize(800, 600)
        
        # 创建界面
        self.create_widgets()
        
        # 初始化状态
        self.is_processing = False
        self.process_thread = None
        
        # 日志管理
        self.log_queue = []  # 日志队列
        self.log_update_timer = None  # 日志更新定时器
        self.max_log_lines = 1000  # 最大日志行数
        self.log_line_count = 0    # 当前日志行数
        
        # 检查数据库连接
        self.check_database_connection()
        
    def create_widgets(self):
        # 主框架
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 配置网格权重
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(6, weight=1)
        
        # 标题
        title_label = ttk.Label(main_frame, text="股票数据处理系统", font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=5, pady=(0, 10))
        
        # 数据库连接状态
        db_frame = ttk.LabelFrame(main_frame, text="数据库连接", padding="5")
        db_frame.grid(row=1, column=0, columnspan=5, sticky=(tk.W, tk.E), pady=(0, 10))
        db_frame.columnconfigure(1, weight=1)
        
        self.db_status_var = tk.StringVar()
        self.db_status_var.set("未检测")
        ttk.Label(db_frame, text="状态:").grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        ttk.Label(db_frame, textvariable=self.db_status_var).grid(row=0, column=1, sticky=tk.W, padx=(0, 10))
        self.db_test_btn = ttk.Button(db_frame, text="检测连接", command=self.check_database_connection)
        self.db_test_btn.grid(row=0, column=2, padx=(0, 5))
        self.db_create_btn = ttk.Button(db_frame, text="创建表", command=self.create_tables)
        self.db_create_btn.grid(row=0, column=3)
        
        # 文件夹选择
        folder_frame = ttk.Frame(main_frame)
        folder_frame.grid(row=2, column=0, columnspan=5, sticky=(tk.W, tk.E), pady=(0, 10))
        folder_frame.columnconfigure(1, weight=1)
        
        ttk.Label(folder_frame, text="数据文件夹:").grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        self.folder_var = tk.StringVar()
        self.folder_entry = ttk.Entry(folder_frame, textvariable=self.folder_var, state="readonly")
        self.folder_entry.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(0, 5))
        self.browse_btn = ttk.Button(folder_frame, text="浏览...", command=self.browse_folder)
        self.browse_btn.grid(row=0, column=2)
        
        # 功能按钮框架
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=3, column=0, columnspan=5, sticky=(tk.W, tk.E), pady=(0, 10))
        
        self.process_btn = ttk.Button(button_frame, text="开始导入", command=self.start_processing)
        self.process_btn.grid(row=0, column=0, padx=(0, 5))
        self.stop_btn = ttk.Button(button_frame, text="停止导入", command=self.stop_processing, state="disabled")
        self.stop_btn.grid(row=0, column=1, padx=(0, 5))
        self.clear_db_btn = ttk.Button(button_frame, text="清空数据库", command=self.clear_database)
        self.clear_db_btn.grid(row=0, column=2, padx=(0, 5))
        self.recovery_btn = ttk.Button(button_frame, text="紧急恢复", command=self.emergency_recovery)
        self.recovery_btn.grid(row=0, column=3, padx=(5, 0))
        
        # 表操作框架
        table_frame = ttk.LabelFrame(main_frame, text="表操作", padding="10")
        table_frame.grid(row=4, column=0, columnspan=5, sticky=(tk.W, tk.E), pady=(0, 10))
        table_frame.columnconfigure(1, weight=1)
        
        ttk.Label(table_frame, text="表名:").grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        
        # 表名下拉框
        self.table_var = tk.StringVar()
        self.table_combo = ttk.Combobox(table_frame, textvariable=self.table_var, width=30)
        self.table_combo.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(0, 5))
        self.refresh_tables_btn = ttk.Button(table_frame, text="刷新", command=self.refresh_tables)
        self.refresh_tables_btn.grid(row=0, column=2, padx=(0, 5))
        self.drop_table_btn = ttk.Button(table_frame, text="删除表", command=self.drop_table)
        self.drop_table_btn.grid(row=0, column=3)
        
        # 进度条
        self.progress_var = tk.IntVar()
        self.progress_bar = ttk.Progressbar(main_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.grid(row=5, column=0, columnspan=5, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # 日志显示区域
        log_frame = ttk.LabelFrame(main_frame, text="处理日志", padding="5")
        log_frame.grid(row=6, column=0, columnspan=5, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        
        # 创建带滚动条的文本框
        log_frame_inner = ttk.Frame(log_frame)
        log_frame_inner.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        log_frame_inner.columnconfigure(0, weight=1)
        log_frame_inner.rowconfigure(0, weight=1)
        
        self.log_text = tk.Text(log_frame_inner, height=15, wrap=tk.WORD)
        self.log_scrollbar = ttk.Scrollbar(log_frame_inner, orient=tk.VERTICAL, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=self.log_scrollbar.set)
        
        self.log_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.log_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        # 底部状态栏
        self.status_var = tk.StringVar()
        self.status_var.set("就绪")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN)
        status_bar.grid(row=7, column=0, columnspan=5, sticky=(tk.W, tk.E))
        
    def check_database_connection(self):
        """检查数据库连接"""
        try:
            if PROCESSOR_AVAILABLE:
                success, message = processor.test_database_connection()
                if success:
                    self.db_status_var.set("✅ 连接成功")
                    self.db_test_btn.config(text="重新检测")
                    self.log_message(message)
                    # 自动刷新表列表
                    self.refresh_tables()
                else:
                    self.db_status_var.set("❌ 连接失败")
                    self.log_message(message)
            else:
                self.db_status_var.set("❌ 模块不可用")
                self.log_message("处理模块不可用，请检查安装")
        except Exception as e:
            self.db_status_var.set("❌ 连接错误")
            self.log_message(f"数据库连接检测出错: {str(e)}")
            
    def create_tables(self):
        """创建数据表"""
        try:
            if PROCESSOR_AVAILABLE:
                success, message = processor.create_tables_if_not_exists()
                if success:
                    self.log_message("✅ " + message)
                    messagebox.showinfo("成功", "数据表创建/检查完成")
                    self.refresh_tables()
                else:
                    self.log_message("❌ " + message)
                    messagebox.showerror("错误", message)
            else:
                messagebox.showerror("错误", "处理模块不可用")
        except Exception as e:
            error_msg = f"创建数据表时出错: {str(e)}"
            self.log_message("❌ " + error_msg)
            messagebox.showerror("错误", error_msg)
            
    def browse_folder(self):
        """浏览文件夹"""
        folder = filedialog.askdirectory(title="选择包含TXT文件的文件夹")
        if folder:
            self.folder_var.set(folder)
            self.log_message(f"已选择文件夹: {folder}")
            
    def start_processing(self):
        """开始处理数据"""
        if self.is_processing:
            messagebox.showwarning("警告", "正在处理中，请等待完成")
            return
            
        folder_path = self.folder_var.get()
        if not folder_path:
            messagebox.showerror("错误", "请先选择数据文件夹")
            return
            
        if not os.path.exists(folder_path):
            messagebox.showerror("错误", "选择的文件夹不存在")
            return
            
        # 确认开始处理
        if not messagebox.askyesno("确认", f"确定要处理文件夹中的所有TXT文件吗？\n路径: {folder_path}"):
            return
            
        # 开始处理（在新线程中）
        self.is_processing = True
        self.process_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self.status_var.set("正在处理数据...")
        self.progress_var.set(0)
        
        # 清空日志队列
        self.log_queue.clear()
        self.log_line_count = 0
        
        # 启动处理线程
        self.process_thread = threading.Thread(target=self.process_data, args=(folder_path,))
        self.process_thread.daemon = True
        self.process_thread.start()
        
    def stop_processing(self):
        """停止处理数据"""
        if self.is_processing and self.process_thread and self.process_thread.is_alive():
            if messagebox.askyesno("确认", "确定要停止当前处理吗？"):
                # 设置取消标志
                if PROCESSOR_AVAILABLE:
                    processor.cancel_processing()
                self.log_message("⚠️ 正在请求停止处理...")
                self.stop_btn.config(state="disabled")
                self.status_var.set("正在停止处理...")
        else:
            self.log_message("⚠️ 没有正在进行的处理任务")
            
    def emergency_recovery(self):
        """紧急恢复 - 重置所有状态"""
        try:
            # 重置处理状态
            if PROCESSOR_AVAILABLE:
                processor.reset_cancel_flag()
            
            self.is_processing = False
            self.process_btn.config(state="normal")
            self.stop_btn.config(state="disabled")
            self.status_var.set("系统已恢复")
            self.log_message("✅ 系统状态已重置")
            
            # 清理日志队列
            self.log_queue.clear()
            if self.log_update_timer:
                self.root.after_cancel(self.log_update_timer)
                self.log_update_timer = None
                
        except Exception as e:
            self.log_message(f"❌ 恢复失败: {str(e)}")
        
    def process_data(self, folder_path):
        """处理数据的线程函数"""
        try:
            if not PROCESSOR_AVAILABLE:
                self.log_message("❌ 处理模块不可用，请检查安装")
                return
                
            def progress_callback(progress):
                # 确保在主线程中更新UI
                self.root.after(0, lambda: self.progress_var.set(progress))
                
            def log_callback(message):
                # 使用队列批量处理日志
                self.log_message(message)
                
            success, message = processor.batch_process_stocks(
                folder_path, 
                progress_callback=progress_callback,
                log_callback=log_callback
            )
            
            if success:
                self.root.after(0, lambda: self.log_message("✅ 数据处理完成"))
            else:
                self.root.after(0, lambda: self.log_message("⚠️ 数据处理被中断或失败"))
                
        except Exception as e:
            self.root.after(0, lambda: self.log_message(f"❌ 处理过程中发生错误: {str(e)}"))
        finally:
            self.is_processing = False
            self.root.after(0, lambda: self.process_btn.config(state="normal"))
            self.root.after(0, lambda: self.stop_btn.config(state="disabled"))
            self.root.after(0, lambda: self.status_var.set("处理完成"))
            # 刷新日志队列
            self.root.after(0, self.flush_log_queue)
            
    def clear_database(self):
        """清空数据库"""
        if self.is_processing:
            messagebox.showwarning("警告", "正在处理中，请等待完成")
            return
            
        # 先检查数据库连接
        if not self.is_database_connected():
            if not messagebox.askyesno("连接问题", "数据库连接可能有问题，是否继续尝试清空操作？"):
                return
            
        if not messagebox.askyesno("确认", "确定要清空数据库中的所有数据吗？此操作不可恢复！"):
            return
            
        try:
            if not PROCESSOR_AVAILABLE:
                messagebox.showerror("错误", "处理模块不可用")
                return
                
            self.log_message("正在清空数据库...")
            success, message = processor.clear_database()
            
            if success:
                self.log_message("✅ " + message)
                messagebox.showinfo("成功", message)
                # 刷新表列表
                self.refresh_tables()
            else:
                self.log_message("❌ " + message)
                messagebox.showerror("错误", message)
                
        except Exception as e:
            error_msg = f"清空数据库时发生错误: {str(e)}"
            self.log_message("❌ " + error_msg)
            messagebox.showerror("错误", error_msg)
            
    def is_database_connected(self):
        """检查数据库是否连接正常"""
        try:
            if PROCESSOR_AVAILABLE:
                success, _ = processor.test_database_connection()
                return success
            return False
        except:
            return False
            
    def refresh_tables(self):
        """刷新表列表"""
        try:
            if PROCESSOR_AVAILABLE:
                # 先检查数据库连接
                success, _ = processor.test_database_connection()
                if not success:
                    self.log_message("数据库连接异常，无法刷新表列表")
                    return
                    
                tables = processor.get_all_table_names()
                self.table_combo['values'] = tables
                if tables:
                    self.table_combo.set(tables[0])
                self.log_message(f"已刷新表列表，共 {len(tables)} 个表")
            else:
                self.table_combo['values'] = []
                self.log_message("处理模块不可用，无法刷新表列表")
        except Exception as e:
            self.log_message(f"刷新表列表时出错: {str(e)}")
            
    def drop_table(self):
        """删除指定表"""
        if self.is_processing:
            messagebox.showwarning("警告", "正在处理中，请等待完成")
            return
            
        table_name = self.table_var.get().strip()
        if not table_name:
            messagebox.showerror("错误", "请先选择或输入表名")
            return
            
        # 先检查数据库连接
        if not self.is_database_connected():
            if not messagebox.askyesno("连接问题", "数据库连接可能有问题，是否继续尝试删除操作？"):
                return
            
        if not messagebox.askyesno("确认", f"确定要删除表 '{table_name}' 吗？此操作不可恢复！"):
            return
            
        try:
            if not PROCESSOR_AVAILABLE:
                messagebox.showerror("错误", "处理模块不可用")
                return
                
            self.log_message(f"正在删除表: {table_name}")
            success, message = processor.drop_table(table_name)
            
            if success:
                self.log_message("✅ " + message)
                messagebox.showinfo("成功", message)
                self.refresh_tables()  # 刷新表列表
            else:
                self.log_message("❌ " + message)
                messagebox.showerror("错误", message)
                
        except Exception as e:
            error_msg = f"删除表时发生错误: {str(e)}"
            self.log_message("❌ " + error_msg)
            messagebox.showerror("错误", error_msg)
            
    def log_message(self, message):
        """添加日志消息 - 批量处理"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted_message = f"[{timestamp}] {message}\n"
        
        # 添加到队列而不是立即更新UI
        self.log_queue.append(formatted_message)
        self.log_line_count += 1
        
        # 如果队列太长，批量更新
        if len(self.log_queue) >= 5:  # 每5条消息更新一次
            self.flush_log_queue()
        elif self.log_update_timer is None:
            # 设置延迟更新
            self.log_update_timer = self.root.after(50, self.flush_log_queue)
        
    def flush_log_queue(self):
        """批量刷新日志队列"""
        if self.log_queue:
            # 批量更新UI
            messages = ''.join(self.log_queue)
            self.log_text.insert(tk.END, messages)
            self.log_text.see(tk.END)
            self.log_queue.clear()
            
            # 限制日志行数
            self.limit_log_lines()
            
        if self.log_update_timer:
            self.root.after_cancel(self.log_update_timer)
            self.log_update_timer = None
            
    def limit_log_lines(self):
        """限制日志行数"""
        try:
            line_count = int(self.log_text.index('end-1c').split('.')[0])
            if line_count > self.max_log_lines:
                # 删除前一半的行
                self.log_text.delete(1.0, f"{line_count - self.max_log_lines // 2}.0")
        except:
            pass  # 忽略错误
            
    def append_log(self, message):
        """向后兼容的追加日志方法"""
        self.log_message(message)

def main():
    root = tk.Tk()
    app = StockDataGUI(root)
    
    # 设置窗口关闭事件
    def on_closing():
        if app.is_processing:
            if not messagebox.askyesno("确认退出", "正在处理数据，确定要退出吗？"):
                return
            # 强制取消处理
            if PROCESSOR_AVAILABLE:
                processor.cancel_processing()
        root.destroy()
        
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()

if __name__ == "__main__":
    main()