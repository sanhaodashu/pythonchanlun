# fractal_processor.py - 分型识别处理模块（性能优化版）
import pandas as pd
import psycopg2
from psycopg2 import pool
import logging
import threading
from datetime import datetime
import time
import numpy as np

# 数据库配置
DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "dbname": "stock_db",
    "user": "postgres",
    "password": "shingowolf123"
}

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("log/fractal_processing.log", encoding='utf-8')
    ]
)
logger = logging.getLogger("FractalProcessor")

# 数据库连接池
DB_POOL = None

def initialize_db_pool():
    """初始化数据库连接池"""
    global DB_POOL
    if DB_POOL is None:
        try:
            DB_POOL = psycopg2.pool.ThreadedConnectionPool(
                minconn=20,   # 增加最小连接数
                maxconn=100,  # 增加最大连接数
                **DB_CONFIG
            )
            logger.info("数据库连接池初始化成功")
        except Exception as e:
            logger.error(f"数据库连接池初始化失败: {str(e)}")

def get_db_connection():
    """获取数据库连接"""
    global DB_POOL
    if DB_POOL is None:
        initialize_db_pool()
        if DB_POOL is None:
            raise Exception("数据库连接池初始化失败")
    return DB_POOL.getconn()

def put_db_connection(conn):
    """归还数据库连接"""
    global DB_POOL
    if DB_POOL and conn:
        try:
            DB_POOL.putconn(conn)
        except Exception as e:
            logger.warning(f"归还数据库连接时出错: {str(e)}")

def get_stock_data_batch(stock_code, limit_days=2000):
    """批量获取股票数据"""
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
        logger.error(f"批量获取股票 {stock_code} 数据时出错: {str(e)}")
        return None
    finally:
        if conn:
            put_db_connection(conn)

def get_processed_klines_batch(stock_code, kline_type='processed', limit=2000):
    """批量获取处理后的K线数据"""
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
        logger.error(f"批量获取处理后的K线数据时出错: {str(e)}")
        return None
    finally:
        if conn:
            put_db_connection(conn)

# 向量化分型检测函数
def is_top_fractal_vectorized(highs, lows):
    """向量化顶分型检测"""
    n = len(highs)
    if n < 3:
        return np.zeros(n, dtype=bool)
    
    # 创建结果数组
    is_top = np.zeros(n, dtype=bool)
    
    # 向量化比较（只检查中间的点）
    middle_indices = np.arange(1, n-1)
    left_highs = highs[middle_indices - 1]
    middle_highs = highs[middle_indices]
    right_highs = highs[middle_indices + 1]
    left_lows = lows[middle_indices - 1]
    middle_lows = lows[middle_indices]
    right_lows = lows[middle_indices + 1]
    
    # 顶分型条件
    contained = (middle_highs >= left_highs) & (middle_highs >= right_highs) & \
                (middle_lows >= left_lows) & (middle_lows >= right_lows)
    
    not_equal = (middle_highs > left_highs) | (middle_highs > right_highs) | \
                (middle_lows > left_lows) | (middle_lows > right_lows)
    
    is_top[middle_indices] = contained & not_equal
    return is_top

def is_bottom_fractal_vectorized(highs, lows):
    """向量化底分型检测"""
    n = len(highs)
    if n < 3:
        return np.zeros(n, dtype=bool)
    
    # 创建结果数组
    is_bottom = np.zeros(n, dtype=bool)
    
    # 向量化比较（只检查中间的点）
    middle_indices = np.arange(1, n-1)
    left_highs = highs[middle_indices - 1]
    middle_highs = highs[middle_indices]
    right_highs = highs[middle_indices + 1]
    left_lows = lows[middle_indices - 1]
    middle_lows = lows[middle_indices]
    right_lows = lows[middle_indices + 1]
    
    # 底分型条件
    contained = (middle_highs <= left_highs) & (middle_highs <= right_highs) & \
                (middle_lows <= left_lows) & (middle_lows <= right_lows)
    
    not_equal = (middle_highs < left_highs) | (middle_highs < right_highs) | \
                (middle_lows < left_lows) | (middle_lows < right_lows)
    
    is_bottom[middle_indices] = contained & not_equal
    return is_bottom

def detect_fractals_optimized(klines_df):
    """优化的分型检测（向量化+正确逻辑）"""
    if len(klines_df) < 3:
        return []
    
    # 向量化处理
    highs = klines_df['high'].values.astype(np.float64)
    lows = klines_df['low'].values.astype(np.float64)
    dates = klines_df['date'].values
    
    # 向量化检测所有可能的分型
    is_top = is_top_fractal_vectorized(highs, lows)
    is_bottom = is_bottom_fractal_vectorized(highs, lows)
    
    # 使用贪心算法确保分型不重叠且有间隔
    fractals = []
    i = 1
    last_fractal_end_index = -1
    
    while i < len(highs) - 1:
        if i <= last_fractal_end_index:
            i += 1
            continue
            
        if is_top[i]:
            fractal = {
                'date': dates[i],
                'type': 'top',
                'high': float(highs[i]),
                'low': float(lows[i]),
                'index': i
            }
            fractals.append(fractal)
            last_fractal_end_index = i + 1  # 分型占用到i+1位置
            i += 4  # 跳过被占用的K线和间隔K线
        elif is_bottom[i]:
            fractal = {
                'date': dates[i],
                'type': 'bottom',
                'high': float(highs[i]),
                'low': float(lows[i]),
                'index': i
            }
            fractals.append(fractal)
            last_fractal_end_index = i + 1  # 分型占用到i+1位置
            i += 4  # 跳过被占用的K线和间隔K线
        else:
            i += 1
    
    return fractals

def save_fractals_batch_optimized(stock_code, fractals):
    """优化的批量保存分型到数据库"""
    conn = None
    try:
        conn = get_db_connection()
        if not conn:
            logger.error(f"❌ 无法获取数据库连接，股票: {stock_code}")
            return False
            
        with conn.cursor() as cur:
            # 先清空该股票的旧分型数据
            cur.execute("""
                DELETE FROM fractals WHERE stock_code = %s
            """, (stock_code,))
            
            # 如果没有分型数据，直接提交
            if not fractals:
                conn.commit()
                return True
            
            # 准备批量插入数据（使用executemany的优化版本）
            data_to_insert = []
            for fractal in fractals:
                try:
                    stock_code_safe = str(stock_code)[:20]
                    # 直接使用字符串格式化日期，避免类型转换开销
                    date_str = fractal['date'].strftime('%Y-%m-%d') if hasattr(fractal['date'], 'strftime') else str(fractal['date'])[:10]
                    fractal_type_safe = str(fractal['type'])[:10]
                    high_safe = float(fractal['high'])
                    low_safe = float(fractal['low'])
                    confirmation_date_str = date_str
                    
                    data_to_insert.append((
                        stock_code_safe,
                        date_str,
                        fractal_type_safe,
                        high_safe,
                        low_safe,
                        confirmation_date_str
                    ))
                except Exception as convert_error:
                    logger.warning(f"数据转换警告，股票 {stock_code}: {str(convert_error)}")
                    continue
            
            if not data_to_insert:
                conn.commit()
                return True
            
            # 使用execute_values进行批量插入（更快）
            from psycopg2.extras import execute_values
            insert_sql = """
                INSERT INTO fractals (stock_code, date, fractal_type, high, low, confirmation_date)
                VALUES %s
            """
            
            execute_values(cur, insert_sql, data_to_insert, template=None, page_size=100)
            
            conn.commit()
            logger.info(f"✅ 成功保存 {len(data_to_insert)} 个分型到数据库，股票代码: {stock_code}")
            return True
            
    except Exception as e:
        error_msg = f"批量保存分型到数据库时出错: {str(e)}"
        logger.error(f"❌ 股票 {stock_code} 保存分型时出错: {error_msg}")
        if conn:
            try:
                conn.rollback()
            except:
                pass
        return False
    finally:
        if conn:
            put_db_connection(conn)

def process_fractals_single(stock_code, progress_callback=None, log_callback=None):
    """优化的单个股票分型识别"""
    start_time = time.time()
    
    try:
        # 批量获取所有数据
        df = get_processed_klines_batch(stock_code, 'processed', 2000)
        if df is None or len(df) == 0:
            # 如果没有处理后的数据，尝试获取原始数据
            df = get_processed_klines_batch(stock_code, 'original', 2000)
            if df is None or len(df) == 0:
                # 如果还没有数据，从daily_data获取
                df = get_stock_data_batch(stock_code, 2000)
                if df is None or len(df) == 0:
                    error_msg = f"未找到股票 {stock_code} 的数据"
                    if log_callback:
                        log_callback("❌ " + error_msg)
                    return False, error_msg
        
        if log_callback:
            log_callback(f"获取到 {len(df)} 条K线数据")
        
        # 使用优化的向量化分型检测
        if log_callback:
            log_callback(f"开始优化的分型检测...")
        fractals = detect_fractals_optimized(df)
        
        # 使用优化的批量保存
        if fractals:
            save_start = time.time()
            save_result = save_fractals_batch_optimized(stock_code, fractals)
            save_time = time.time() - save_start
            if log_callback:
                log_callback(f"数据库保存耗时: {save_time:.2f}秒")
            
            if save_result:
                if log_callback:
                    log_callback(f"✅ 成功保存 {len(fractals)} 个分型到数据库")
            else:
                if log_callback:
                    log_callback(f"❌ 批量保存分型到数据库失败")
                return False, "批量保存分型到数据库失败"
        else:
            if log_callback:
                log_callback(f"⚠️ 未检测到分型")
        
        # 统计信息
        elapsed_time = time.time() - start_time
        processing_speed = len(df) / elapsed_time if elapsed_time > 0 else 0
        
        if log_callback:
            log_callback(f"📊 分型检测完成:")
            log_callback(f"   总K线数: {len(df)}")
            log_callback(f"   检测到分型: {len(fractals)} 个")
            log_callback(f"   处理速度: {processing_speed:.1f} K线/秒")
            log_callback(f"   总耗时: {elapsed_time:.2f} 秒")
            log_callback("✅ 分型识别完成")
        
        result_msg = f"分型识别完成，共检测到 {len(fractals)} 个分型"
        return True, result_msg
        
    except Exception as e:
        error_msg = f"处理分型识别时出错: {str(e)}"
        logger.error(f"股票 {stock_code} 处理分型识别时出错: {error_msg}")
        if log_callback:
            log_callback("❌ " + error_msg)
        return False, error_msg

# 初始化数据库连接池
if __name__ != "__main__":
    initialize_db_pool()