# chanlun_processor.py - 缠论处理核心模块（修正版）
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
    'MAX_WORKERS': 4,  # 默认线程数
    'TIMEOUT_PER_STOCK': 60,  # 每只股票处理超时时间（秒）
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
    PROCESSING_CONFIG['MAX_WORKERS'] = max(1, min(num_workers, 16))  # 限制在1-16之间

def initialize_db_pool():
    """初始化数据库连接池"""
    global DB_POOL
    try:
        DB_POOL = psycopg2.pool.ThreadedConnectionPool(
            minconn=1,
            maxconn=PROCESSING_CONFIG['MAX_WORKERS'] + 5,
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
        DB_POOL.putconn(conn)

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

def preprocess_klines_complete(klines):
    """完整预处理K线数据，合并所有同日数据"""
    if len(klines) <= 1:
        return klines
    
    # 按日期分组
    date_groups = defaultdict(list)
    for kline in klines:
        date_groups[kline['date']].append(kline)
    
    # 合并每组同日数据
    processed = []
    for date, group in date_groups.items():
        if len(group) == 1:
            processed.append(group[0])
        else:
            # 合并同日数据
            merged = {
                'date': date,
                'open': group[0]['open'],  # 第一根的开盘价
                'high': max([float(k['high']) for k in group]),
                'low': min([float(k['low']) for k in group]),
                'close': group[-1]['close'],  # 最后一根的收盘价
                'volume': sum([int(k['volume']) for k in group])
            }
            processed.append(merged)
    
    # 按日期排序
    processed.sort(key=lambda x: x['date'])
    return processed

def check_inclusion(k1, k2):
    """
    检查两根K线是否包含关系
    k1, k2: dict, 包含high, low字段
    返回: (是否包含, 方向) 
    方向: 1=向上包含(k2包含k1), -1=向下包含(k1包含k2), 0=无包含
    """
    # 同日数据不认为是包含关系
    if k1['date'] == k2['date']:
        return False, 0
    
    k1_high = round(float(k1['high']), 4)
    k1_low = round(float(k1['low']), 4)
    k2_high = round(float(k2['high']), 4)
    k2_low = round(float(k2['low']), 4)
    
    if k1_high >= k2_high and k1_low <= k2_low and not (k1_high == k2_high and k1_low == k2_low):
        return True, -1  # k1包含k2，向下包含
    elif k2_high >= k1_high and k2_low <= k1_low and not (k2_high == k1_high and k2_low == k1_low):
        return True, 1   # k2包含k1，向上包含
    else:
        return False, 0  # 无包含关系

def get_kline_direction(kline):
    """判断K线方向：1=上涨, -1=下跌, 0=平盘"""
    if float(kline['close']) > float(kline['open']):
        return 1
    elif float(kline['close']) < float(kline['open']):
        return -1
    else:
        return 0

def merge_klines_correct(k1, k2, direction, prev_direction):
    """
    正确的K线合并方法（缠论标准）
    direction: 1=向上包含(k2包含k1), -1=向下包含(k1包含k2)
    prev_direction: 1=前一根K线上涨, -1=前一根K线下跌
    """
    merged = {
        'date': k2['date'],  # 保留较新的日期
        'open': k2['open'],
        'high': max(float(k1['high']), float(k2['high'])),
        'low': min(float(k1['low']), float(k2['low'])),
        'close': k2['close'],
        'volume': int(k1['volume']) + int(k2['volume'])
    }
    
    # 根据前一根K线的走势决定包含处理方式（缠论标准）
    if direction == 1:  # 向上包含(k2包含k1)
        if prev_direction == 1:  # 前一根K线上涨
            # 取高high和高low（顺势）
            merged['high'] = max(float(k1['high']), float(k2['high']))
            merged['low'] = max(float(k1['low']), float(k2['low']))
        else:  # 前一根K线下跌
            # 取低high和低low（逆势）
            merged['high'] = min(float(k1['high']), float(k2['high']))
            merged['low'] = min(float(k1['low']), float(k2['low']))
    elif direction == -1:  # 向下包含(k1包含k2)
        if prev_direction == 1:  # 前一根K线上涨
            # 取低high和低low（逆势）
            merged['high'] = min(float(k1['high']), float(k2['high']))
            merged['low'] = min(float(k1['low']), float(k2['low']))
        else:  # 前一根K线下跌
            # 取高high和高low（顺势）
            merged['high'] = max(float(k1['high']), float(k2['high']))
            merged['low'] = max(float(k1['low']), float(k2['low']))
        
    return merged

def save_processed_klines(stock_code, klines, kline_type='processed'):
    """保存处理后的K线数据"""
    conn = None
    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            # 先删除旧数据
            cur.execute("""
                DELETE FROM processed_klines 
                WHERE stock_code = %s AND kline_type = %s
            """, (stock_code, kline_type))
            
            # 批量插入新数据
            data_to_insert = []
            for kline in klines:
                data_to_insert.append((
                    stock_code,
                    kline['date'],
                    float(kline['open']),
                    float(kline['high']),
                    float(kline['low']),
                    float(kline['close']),
                    int(kline['volume']),
                    kline_type
                ))
            
            cur.executemany("""
                INSERT INTO processed_klines (
                    stock_code, date, open, high, low, close, volume, kline_type
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """, data_to_insert)
            
            conn.commit()
            return True
            
    except Exception as e:
        logger.error(f"保存处理后的K线数据时出错: {str(e)}")
        if conn:
            conn.rollback()
        return False
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

def process_inclusion_relation_single(stock_code, progress_callback=None, log_callback=None):
    """处理单个股票的包含关系（修正的正确逻辑）"""
    start_time = time.time()
    
    try:
        # 获取股票数据
        df = get_stock_data(stock_code)
        if df is None or len(df) == 0:
            error_msg = f"未找到股票 {stock_code} 的数据"
            if log_callback:
                log_callback("❌ " + error_msg)
            return False, error_msg
            
        if log_callback:
            log_callback(f"获取到 {len(df)} 条K线数据")
            
        # 转换为K线列表
        klines = []
        for _, row in df.iterrows():
            klines.append({
                'date': row['date'],
                'open': float(row['open']),
                'high': float(row['high']),
                'low': float(row['low']),
                'close': float(row['close']),
                'volume': int(row['volume'])
            })
        
        # 完整预处理同日数据
        original_count = len(klines)
        klines = preprocess_klines_complete(klines)
        if log_callback and len(klines) < original_count:
            log_callback(f"预处理同日数据: {original_count} → {len(klines)} 根K线")
        
        # 保存原始K线数据
        save_processed_klines(stock_code, klines, 'original')
        
        if len(klines) < 2:
            error_msg = "数据不足，至少需要2根K线"
            if log_callback:
                log_callback("❌ " + error_msg)
            return False, error_msg
            
        # 处理包含关系（正确的缠论逻辑）
        processed_klines = []
        i = 0
        total_klines = len(klines)
        merge_count = 0  # 总合并次数
        
        while i < total_klines and not is_processing_cancelled():
            current_time = time.time()
            if current_time - start_time > PROCESSING_CONFIG['TIMEOUT_PER_STOCK']:
                if log_callback:
                    log_callback(f"⚠️ 股票 {stock_code} 处理超时")
                break
            
            # 如果是第一根K线，直接添加
            if len(processed_klines) == 0:
                processed_klines.append(klines[i])
                i += 1
                continue
            
            # 检查是否与前一根K线有包含关系
            has_inclusion, direction = check_inclusion(processed_klines[-1], klines[i])
            
            if has_inclusion:
                # 有包含关系：A和B包含处理
                prev_direction = get_kline_direction(processed_klines[-2]) if len(processed_klines) >= 2 else 1
                merged_result = merge_klines_correct(processed_klines[-1], klines[i], direction, prev_direction)
                
                # 正确的丢弃逻辑：
                # 1. 丢弃A：processed_klines[-1]（通过替换）
                # 2. 丢弃B：klines[i]（通过增加i跳过）
                # 3. 保留结果C：merged_result
                processed_klines[-1] = merged_result  # 用C替换A（丢弃A）
                i += 1  # 跳过B（丢弃B）
                
                merge_count += 1
                
                # 检查合并后的K线是否还能与更前面的K线合并（连续包含）
                consecutive_merge_count = 0
                while len(processed_klines) >= 2 and consecutive_merge_count < 10:
                    has_inclusion_again, direction_again = check_inclusion(processed_klines[-2], processed_klines[-1])
                    
                    if has_inclusion_again:
                        prev_direction_again = get_kline_direction(processed_klines[-3]) if len(processed_klines) >= 3 else 1
                        merged_result_again = merge_klines_correct(processed_klines[-2], processed_klines[-1], direction_again, prev_direction_again)
                        
                        # 连续包含的丢弃逻辑：
                        # 1. 丢弃processed_klines[-2]和processed_klines[-1]
                        # 2. 保留合并结果merged_result_again
                        processed_klines.pop()  # 丢弃最后一根
                        processed_klines[-1] = merged_result_again  # 用合并结果替换倒数第一根
                        
                        merge_count += 1
                        consecutive_merge_count += 1
                    else:
                        break
            else:
                # 无包含关系，添加当前K线
                processed_klines.append(klines[i])
                i += 1
            
            # 更新进度（每处理10根K线更新一次）
            if progress_callback and total_klines > 0 and i % 10 == 0:
                progress = int((min(i, total_klines) / total_klines) * 100)
                progress_callback(progress)
            
            # 检查取消标志
            if is_processing_cancelled():
                if log_callback:
                    log_callback(f"⚠️ 股票 {stock_code} 处理被取消")
                return False, "处理被取消"
        
        if is_processing_cancelled():
            return False, "处理被取消"
        
        # 保存处理后的K线数据
        save_processed_klines(stock_code, processed_klines, 'processed')
        
        # 添加详细的统计信息
        elapsed_time = time.time() - start_time
        compression_rate = ((len(klines) - len(processed_klines)) / len(klines) * 100) if len(klines) > 0 else 0
        
        if log_callback:
            log_callback(f"📊 统计信息:")
            log_callback(f"   原始K线数: {len(klines)}")
            log_callback(f"   处理后K线数: {len(processed_klines)}")
            log_callback(f"   合并次数: {merge_count}")
            log_callback(f"   压缩率: {compression_rate:.1f}%")
            log_callback(f"   处理耗时: {elapsed_time:.2f} 秒")
            
            # 检查是否有同日数据
            dates = [k['date'] for k in processed_klines]
            duplicate_dates = []
            for date in dates:
                if dates.count(date) > 1 and date not in duplicate_dates:
                    duplicate_dates.append(date)
            
            if duplicate_dates:
                log_callback(f"   ⚠️  发现同日数据: {len(duplicate_dates)} 个日期")
            else:
                log_callback(f"   ✅ 无同日数据")
        
        result_msg = f"包含关系处理完成，原始 {len(klines)} 根K线，处理后 {len(processed_klines)} 根K线，合并 {merge_count} 次"
        if log_callback:
            log_callback("✅ " + result_msg)
            log_callback("✅ 处理后的数据已保存到数据库")
            
        return True, result_msg
        
    except Exception as e:
        error_msg = f"处理包含关系时出错: {str(e)}"
        logger.error(error_msg)
        if log_callback:
            log_callback("❌ " + error_msg)
        return False, error_msg

def process_inclusion_relation(stock_code, progress_callback=None, log_callback=None):
    """处理包含关系（包装函数，支持取消检查）"""
    if is_processing_cancelled():
        return False, "处理被取消"
    
    return process_inclusion_relation_single(stock_code, progress_callback, log_callback)

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
        
        return stock_code, success, message
    except Exception as e:
        return stock_code, False, f"处理异常: {str(e)}"

def batch_process_all_stocks(process_type, progress_callback=None, log_callback=None):
    """批量处理所有股票（多线程版本）- 全量更新"""
    reset_cancel_flag()
    
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
            
        total_stocks = len(stock_codes)
        successful = 0
        failed_stocks = []
        completed = 0
        
        # 使用线程池处理 - 处理所有股票
        with ThreadPoolExecutor(max_workers=PROCESSING_CONFIG['MAX_WORKERS']) as executor:
            # 提交所有任务 - 处理所有股票
            future_to_stock = {
                executor.submit(process_single_stock_wrapper, stock_code, process_type): stock_code 
                for stock_code in stock_codes  # 处理所有股票
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
                    stock_code, success, message = future.result(timeout=120)  # 2分钟超时
                    
                    if success:
                        successful += 1
                        if log_callback and successful % 10 == 0:  # 每10个成功记录一次
                            progress_percent = (successful / total_stocks) * 100
                            log_callback(f"✅ 已完成 {successful}/{total_stocks} 只股票 ({progress_percent:.1f}%)")
                    else:
                        failed_stocks.append((stock_code, message))
                    
                    completed += 1
                    
                    # 更新进度
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
        result_msg = f"批量处理完成! 总数: {total_stocks}, 成功: {successful}, 失败: {len(failed_stocks)}"
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

def process_fractals(stock_code, progress_callback=None, log_callback=None):
    """分型识别（待实现）"""
    if log_callback:
        log_callback("分型识别功能尚未实现")
    return False, "分型识别功能尚未实现"

# 初始化数据库连接池
if __name__ != "__main__":
    initialize_db_pool()
    create_tables_if_not_exists()