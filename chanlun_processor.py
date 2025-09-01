# chanlun_processor.py - 缠论处理主模块（优化版）
import pandas as pd
import numpy as np
import psycopg2
from psycopg2 import pool
import logging
import threading
import os
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from collections import defaultdict

# 导入各个功能模块
try:
    import inclusion_processor  # 修正导入方式
    import fractal_processor    # 修正导入方式
    PROCESSORS_AVAILABLE = True
except ImportError as e:
    print(f"导入处理模块失败: {e}")
    PROCESSORS_AVAILABLE = False

# 数据库配置
DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "dbname": "stock_db",
    "user": "postgres",
    "password": "shingowolf123"
}

# 处理配置
PROCESSING_CONFIG = {
    'MAX_WORKERS': 32,  # 默认线程数
    'TIMEOUT_PER_STOCK': 120,  # 每只股票处理超时时间（秒）
}

# 创建日志目录
log_dir = "log"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(log_dir, "chanlun_processing.log"), encoding='utf-8')
    ]
)
logger = logging.getLogger("ChanlunProcessor")

# 数据库连接池
DB_POOL = None
processing_cancelled = False
processing_lock = threading.Lock()

def set_max_workers(num_workers):
    """设置最大工作线程数"""
    global PROCESSING_CONFIG
    PROCESSING_CONFIG['MAX_WORKERS'] = max(1, min(num_workers, 64))  # 增加到64线程

def initialize_db_pool():
    """初始化数据库连接池"""
    global DB_POOL
    try:
        DB_POOL = psycopg2.pool.ThreadedConnectionPool(
            minconn=20,   # 增加最小连接数
            maxconn=100,  # 增加最大连接数
            **DB_CONFIG
        )
        return True
    except Exception as e:
        logger.error(f"数据库连接池初始化失败: {str(e)}")
        return False

def get_db_connection():
    """获取数据库连接"""
    global DB_POOL
    if DB_POOL is None:
        if not initialize_db_pool():
            raise Exception("数据库连接池未初始化")
    return DB_POOL.getconn()

def put_db_connection(conn):
    """归还数据库连接"""
    global DB_POOL
    if DB_POOL and conn:
        try:
            DB_POOL.putconn(conn)
        except Exception as e:
            logger.warning(f"归还数据库连接时出错: {str(e)}")

def is_processing_cancelled():
    """检查是否被取消"""
    global processing_cancelled
    with processing_lock:
        return processing_cancelled

def cancel_processing():
    """取消处理过程"""
    global processing_cancelled
    with processing_lock:
        processing_cancelled = True
    logger.info("处理已被取消")

def reset_cancel_flag():
    """重置取消标志"""
    global processing_cancelled
    with processing_lock:
        processing_cancelled = False
    logger.info("取消标志已重置")

def get_stock_data(stock_code, limit_days=1000):
    """获取股票数据"""
    conn = None
    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            cur.execute("""
                SELECT date, open, high, low, close, volume 
                FROM daily_data 
                WHERE stock_code = %s 
                ORDER BY date DESC 
                LIMIT %s
            """, (stock_code, limit_days))
            
            rows = cur.fetchall()
            if not rows:
                return None
                
            # 转换为DataFrame并按日期升序排列
            df = pd.DataFrame(rows, columns=['date', 'open', 'high', 'low', 'close', 'volume'])
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date').reset_index(drop=True)
            return df
            
    except Exception as e:
        logger.error(f"获取股票 {stock_code} 数据时出错: {str(e)}")
        return None
    finally:
        if conn:
            put_db_connection(conn)

def get_all_stock_codes():
    """获取所有股票代码"""
    conn = None
    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            cur.execute("SELECT DISTINCT stock_code FROM daily_data ORDER BY stock_code")
            rows = cur.fetchall()
            return [row[0] for row in rows] if rows else []
    except Exception as e:
        logger.error(f"获取所有股票代码时出错: {str(e)}")
        return []
    finally:
        if conn:
            put_db_connection(conn)

def get_processed_klines(stock_code, kline_type='processed', limit=100):
    """获取处理后的K线数据"""
    conn = None
    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            cur.execute("""
                SELECT date, open, high, low, close, volume 
                FROM processed_klines 
                WHERE stock_code = %s AND kline_type = %s
                ORDER BY date DESC 
                LIMIT %s
            """, (stock_code, kline_type, limit))
            
            rows = cur.fetchall()
            if not rows:
                return None
                
            # 转换为DataFrame并按日期升序排列
            df = pd.DataFrame(rows, columns=['date', 'open', 'high', 'low', 'close', 'volume'])
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date').reset_index(drop=True)
            return df
            
    except Exception as e:
        logger.error(f"获取处理后的K线数据时出错: {str(e)}")
        return None
    finally:
        if conn:
            put_db_connection(conn)

def create_tables_if_not_exists():
    """自动创建表（如果不存在）"""
    conn = None
    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            # 创建日线数据表
            cur.execute("""
                CREATE TABLE IF NOT EXISTS daily_data (
                    stock_code VARCHAR(20),
                    date DATE,
                    open DECIMAL,
                    high DECIMAL,
                    low DECIMAL,
                    close DECIMAL,
                    volume BIGINT,
                    amount DECIMAL,
                    macd DECIMAL,
                    macd_signal DECIMAL,
                    macd_histogram DECIMAL,
                    PRIMARY KEY (stock_code, date)
                )
            """)
            
            # 创建周线数据表
            cur.execute("""
                CREATE TABLE IF NOT EXISTS weekly_data (
                    stock_code VARCHAR(20),
                    date DATE,
                    open DECIMAL,
                    high DECIMAL,
                    low DECIMAL,
                    close DECIMAL,
                    volume BIGINT,
                    amount DECIMAL,
                    macd DECIMAL,
                    macd_signal DECIMAL,
                    macd_histogram DECIMAL,
                    PRIMARY KEY (stock_code, date)
                )
            """)
            
            # 创建月线数据表
            cur.execute("""
                CREATE TABLE IF NOT EXISTS monthly_data (
                    stock_code VARCHAR(20),
                    date DATE,
                    open DECIMAL,
                    high DECIMAL,
                    low DECIMAL,
                    close DECIMAL,
                    volume BIGINT,
                    amount DECIMAL,
                    macd DECIMAL,
                    macd_signal DECIMAL,
                    macd_histogram DECIMAL,
                    PRIMARY KEY (stock_code, date)
                )
            """)
            
            # 创建处理后的K线数据表（包含关系处理后）
            cur.execute("""
                CREATE TABLE IF NOT EXISTS processed_klines (
                    stock_code VARCHAR(20),
                    date DATE,
                    open DECIMAL,
                    high DECIMAL,
                    low DECIMAL,
                    close DECIMAL,
                    volume BIGINT,
                    kline_type VARCHAR(20), -- 'original' 或 'processed'
                    PRIMARY KEY (stock_code, date, kline_type)
                )
            """)
            
            # 创建分型数据表
            cur.execute("""
                CREATE TABLE IF NOT EXISTS fractals (
                    id SERIAL PRIMARY KEY,
                    stock_code VARCHAR(20),
                    date DATE,
                    fractal_type VARCHAR(10), -- 'top' 或 'bottom'
                    high DECIMAL,
                    low DECIMAL,
                    confirmation_date DATE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # 创建笔数据表
            cur.execute("""
                CREATE TABLE IF NOT EXISTS bi_lines (
                    id SERIAL PRIMARY KEY,
                    stock_code VARCHAR(20),
                    start_date DATE,
                    end_date DATE,
                    start_price DECIMAL,
                    end_price DECIMAL,
                    direction VARCHAR(10), -- 'up' 或 'down'
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # 创建线段数据表
            cur.execute("""
                CREATE TABLE IF NOT EXISTS line_segments (
                    id SERIAL PRIMARY KEY,
                    stock_code VARCHAR(20),
                    start_date DATE,
                    end_date DATE,
                    start_price DECIMAL,
                    end_price DECIMAL,
                    direction VARCHAR(10), -- 'up' 或 'down'
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # 创建中枢数据表
            cur.execute("""
                CREATE TABLE IF NOT EXISTS centers (
                    id SERIAL PRIMARY KEY,
                    stock_code VARCHAR(20),
                    start_date DATE,
                    end_date DATE,
                    high DECIMAL,
                    low DECIMAL,
                    level INTEGER, -- 中枢级别
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.commit()
            return True, "数据表创建/检查完成"
    except Exception as e:
        error_msg = f"创建数据表时出错: {str(e)}"
        logger.error(error_msg)
        if conn:
            conn.rollback()
        return False, error_msg
    finally:
        if conn:
            put_db_connection(conn)

def process_inclusion_relation(stock_code, progress_callback=None, log_callback=None):
    """处理包含关系"""
    if is_processing_cancelled():
        return False, "处理被取消"
    
    if PROCESSORS_AVAILABLE:
        return inclusion_processor.process_inclusion_relation_single(stock_code, progress_callback, log_callback)
    else:
        if log_callback:
            log_callback("❌ 包含关系处理模块不可用")
        return False, "包含关系处理模块不可用"

def process_fractals(stock_code, progress_callback=None, log_callback=None):
    """处理分型识别"""
    if is_processing_cancelled():
        return False, "处理被取消"
    
    if PROCESSORS_AVAILABLE:
        return fractal_processor.process_fractals_single(stock_code, progress_callback, log_callback)
    else:
        if log_callback:
            log_callback("❌ 分型识别模块不可用")
        return False, "分型识别模块不可用"

def process_single_stock_wrapper(stock_code, process_type, log_callback=None):
    """单个股票处理包装函数（用于多线程）"""
    try:
        if process_type == "inclusion":
            success, message = process_inclusion_relation(stock_code, log_callback=log_callback)
        elif process_type == "fractal":
            success, message = process_fractals(stock_code, log_callback=log_callback)
        else:
            success = False
            message = f"未知的处理类型: {process_type}"
        
        # 确保返回正确的格式
        return stock_code, success, message
    except Exception as e:
        return stock_code, False, f"处理异常: {str(e)}"

def batch_process_all_stocks(process_type, progress_callback=None, log_callback=None):
    """批量处理所有股票（多线程版本）- 全量更新"""
    reset_cancel_flag()
    
    start_time = time.time()
    
    if log_callback:
        log_callback(f"开始批量处理所有股票的{process_type}（{PROCESSING_CONFIG['MAX_WORKERS']}线程）...")
    
    try:
        # 获取所有股票代码
        stock_codes = get_all_stock_codes()
        if not stock_codes:
            error_msg = "未找到任何股票数据"
            if log_callback:
                log_callback("❌ " + error_msg)
            return False, error_msg
            
        if log_callback:
            log_callback(f"共找到 {len(stock_codes)} 只股票")
            
        # 为了测试，先处理前100只股票
        # stock_codes = stock_codes[:min(100, len(stock_codes))]
            
        total_stocks = len(stock_codes)
        successful = 0
        failed_stocks = []
        completed = 0
        
        # 使用线程池处理 - 处理所有股票
        with ThreadPoolExecutor(max_workers=PROCESSING_CONFIG['MAX_WORKERS']) as executor:
            # 提交所有任务 - 处理所有股票
            future_to_stock = {
                executor.submit(process_single_stock_wrapper, stock_code, process_type): stock_code 
                for stock_code in stock_codes
            }
            
            # 处理完成的任务
            for future in as_completed(future_to_stock):
                if is_processing_cancelled():
                    if log_callback:
                        log_callback("⚠️ 处理被用户取消")
                    # 取消未完成的任务
                    for f in future_to_stock:
                        f.cancel()
                    break
                
                try:
                    stock_code, success, message = future.result(timeout=180)  # 3分钟超时
                    
                    if success:
                        successful += 1
                        # 每处理5个就显示一次进度（更频繁）
                        if log_callback and (successful % 5 == 0 or successful <= 20):
                            elapsed_time = time.time() - start_time
                            speed = successful / elapsed_time if elapsed_time > 0 else 0
                            progress_percent = (successful / total_stocks) * 100
                            log_callback(f"✅ 已完成 {successful}/{total_stocks} 只股票 ({progress_percent:.1f}%) - 速度: {speed:.1f}只/秒")
                    else:
                        failed_stocks.append((stock_code, message))
                        # 每失败3个显示一次
                        if log_callback and (len(failed_stocks) % 3 == 0 or len(failed_stocks) <= 10):
                            log_callback(f"❌ {stock_code}: {message}")
                    
                    completed += 1
                    
                    # 更新进度条
                    if progress_callback and total_stocks > 0:
                        progress = int((completed / total_stocks) * 100)
                        progress_callback(progress)
                        
                except Exception as e:
                    stock_code = future_to_stock[future]
                    error_msg = f"处理股票 {stock_code} 时出错: {str(e)}"
                    failed_stocks.append((stock_code, error_msg))
                    if log_callback:
                        log_callback("❌ " + error_msg)
        
        # 统计结果 - 显示完整统计
        elapsed_total = time.time() - start_time
        result_msg = f"批量处理完成! 总数: {total_stocks}, 成功: {successful}, 失败: {len(failed_stocks)}, 耗时: {elapsed_total:.1f}秒"
        if log_callback:
            log_callback("✅ " + result_msg)
            if failed_stocks:
                log_callback("❌ 失败股票:")
                for code, error in failed_stocks[:10]:  # 显示前10个错误
                    log_callback(f"  - {code}: {error}")
                if len(failed_stocks) > 10:
                    log_callback(f"  ... 还有 {len(failed_stocks) - 10} 个失败")
        
        return not is_processing_cancelled(), result_msg
        
    except Exception as e:
        error_msg = f"批量处理过程中发生错误: {str(e)}"
        logger.error(error_msg)
        if log_callback:
            log_callback("❌ " + error_msg)
        return False, error_msg

# 初始化数据库连接池
if __name__ != "__main__":
    initialize_db_pool()
    create_tables_if_not_exists()