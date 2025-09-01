# inclusion_processor.py - 包含关系处理模块（性能优化版）
import pandas as pd
import numpy as np
import psycopg2
from psycopg2 import pool
from psycopg2.extras import execute_values
import logging
import threading
import os
from datetime import datetime
import time

# 数据库配置
DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "dbname": "stock_db",
    "user": "postgres",
    "password": "shingowolf123"
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
        logging.FileHandler(os.path.join(log_dir, "inclusion_processing.log"), encoding='utf-8')
    ]
)
logger = logging.getLogger("InclusionProcessor")

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

def preprocess_klines_vectorized_optimized(klines):
    """优化的向量化预处理K线数据，合并所有同日数据"""
    if len(klines) <= 1:
        return klines
    
    # 使用pandas进行向量化处理（优化版）
    df = pd.DataFrame(klines)
    df['date'] = pd.to_datetime(df['date'])
    
    # 按日期分组并聚合（优化聚合函数）
    grouped = df.groupby('date').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).reset_index()
    
    # 向量化转换回klines格式
    processed = grouped.to_dict('records')
    
    # 确保数据类型正确
    for record in processed:
        record['open'] = float(record['open'])
        record['high'] = float(record['high'])
        record['low'] = float(record['low'])
        record['close'] = float(record['close'])
        record['volume'] = int(record['volume'])
    
    # 按日期排序
    processed.sort(key=lambda x: x['date'])
    return processed

def check_inclusion_vectorized_optimized(highs, lows, dates):
    """优化的向量化包含关系检查"""
    n = len(highs)
    if n < 2:
        return np.array([]), np.array([]), np.array([])
    
    # 向量化计算
    prev_highs = highs[:-1]
    prev_lows = lows[:-1]
    curr_highs = highs[1:]
    curr_lows = lows[1:]
    prev_dates = dates[:-1]
    curr_dates = dates[1:]
    
    # 过滤掉同日数据
    same_date_mask = prev_dates == curr_dates
    valid_mask = ~same_date_mask
    
    # 只处理不同日的数据
    valid_prev_highs = prev_highs[valid_mask]
    valid_prev_lows = prev_lows[valid_mask]
    valid_curr_highs = curr_highs[valid_mask]
    valid_curr_lows = curr_lows[valid_mask]
    valid_indices = np.where(valid_mask)[0]
    
    # 向上包含：curr包含prev
    up_inclusion = (valid_curr_highs >= valid_prev_highs) & (valid_curr_lows <= valid_prev_lows) & \
                   ~((valid_curr_highs == valid_prev_highs) & (valid_curr_lows == valid_prev_lows))
    
    # 向下包含：prev包含curr
    down_inclusion = (valid_prev_highs >= valid_curr_highs) & (valid_prev_lows <= valid_curr_lows) & \
                     ~((valid_prev_highs == valid_curr_highs) & (valid_prev_lows == valid_curr_lows))
    
    # 获取包含关系的索引
    up_indices = valid_indices[up_inclusion]
    down_indices = valid_indices[down_inclusion]
    up_directions = np.ones(len(up_indices), dtype=int)
    down_directions = -np.ones(len(down_indices), dtype=int)
    
    return up_indices, down_indices, up_directions, down_directions

def merge_klines_vectorized_optimized(k1_high, k1_low, k1_open, k1_close, k1_volume,
                                    k2_high, k2_low, k2_open, k2_close, k2_volume,
                                    k2_date, direction, prev_direction):
    """
    优化的向量化K线合并方法（缠论标准）
    """
    merged_high = max(k1_high, k2_high)
    merged_low = min(k1_low, k2_low)
    merged_volume = k1_volume + k2_volume
    
    # 根据前一根K线的走势决定包含处理方式（缠论标准）
    if direction == 1:  # 向上包含(k2包含k1)
        if prev_direction == 1:  # 前一根K线上涨
            # 取高high和高low（顺势）
            merged_high = max(k1_high, k2_high)
            merged_low = max(k1_low, k2_low)
        else:  # 前一根K线下跌
            # 取低high和低low（逆势）
            merged_high = min(k1_high, k2_high)
            merged_low = min(k1_low, k2_low)
    elif direction == -1:  # 向下包含(k1包含k2)
        if prev_direction == 1:  # 前一根K线上涨
            # 取低high和低low（逆势）
            merged_high = min(k1_high, k2_high)
            merged_low = min(k1_low, k2_low)
        else:  # 前一根K线下跌
            # 取高high和高low（顺势）
            merged_high = max(k1_high, k2_high)
            merged_low = max(k1_low, k2_low)
    
    return {
        'date': k2_date,
        'open': k2_open,
        'high': merged_high,
        'low': merged_low,
        'close': k2_close,
        'volume': merged_volume
    }

def save_processed_klines_batch_optimized(stock_code, klines, kline_type='processed'):
    """优化的批量保存处理后的K线数据"""
    conn = None
    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            # 批量删除旧数据
            cur.execute("""
                DELETE FROM processed_klines 
                WHERE stock_code = %s AND kline_type = %s
            """, (stock_code, kline_type))
            
            # 准备批量插入数据
            if not klines:
                conn.commit()
                return True
            
            # 使用execute_values进行批量插入（更快）
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
            
            # 使用execute_values进行批量插入
            insert_sql = """
                INSERT INTO processed_klines (
                    stock_code, date, open, high, low, close, volume, kline_type
                ) VALUES %s
            """
            
            execute_values(cur, insert_sql, data_to_insert, template=None, page_size=1000)
            
            conn.commit()
            logger.info(f"✅ 成功保存 {len(data_to_insert)} 条K线数据到数据库，股票代码: {stock_code}")
            return True
            
    except Exception as e:
        error_msg = f"批量保存处理后的K线数据时出错: {str(e)}"
        logger.error(error_msg)
        if conn:
            try:
                conn.rollback()
            except:
                pass
        return False
    finally:
        if conn:
            put_db_connection(conn)

def get_stock_data_batch_optimized(stock_code, limit_days=2000):
    """优化的批量获取股票数据"""
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
                
            # 转换为DataFrame并按日期升序排列（优化版）
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

def process_inclusion_relation_single(stock_code, progress_callback=None, log_callback=None):
    """高度优化的单个股票包含关系处理"""
    start_time = time.time()
    
    try:
        # 批量获取股票数据
        df = get_stock_data_batch_optimized(stock_code, 2000)
        if df is None or len(df) == 0:
            error_msg = f"未找到股票 {stock_code} 的数据"
            if log_callback:
                log_callback("❌ " + error_msg)
            return False, error_msg
            
        if log_callback:
            log_callback(f"批量获取到 {len(df)} 条K线数据")
        
        # 向量化预处理同日数据
        original_count = len(df)
        klines_list = df.to_dict('records')
        klines_list = preprocess_klines_vectorized_optimized(klines_list)
        processed_count = len(klines_list)
        
        if log_callback and processed_count < original_count:
            log_callback(f"向量化预处理同日数据: {original_count} → {processed_count} 根K线")
        
        # 批量保存原始K线数据
        save_start = time.time()
        save_processed_klines_batch_optimized(stock_code, klines_list, 'original')
        save_time = time.time() - save_start
        if log_callback:
            log_callback(f"原始数据保存耗时: {save_time:.2f}秒")
        
        if len(klines_list) < 2:
            error_msg = "数据不足，至少需要2根K线"
            if log_callback:
                log_callback("❌ " + error_msg)
            return False, error_msg
        
        # 向量化包含关系处理
        processing_start = time.time()
        
        # 转换为numpy数组进行向量化处理
        highs = np.array([float(k['high']) for k in klines_list])
        lows = np.array([float(k['low']) for k in klines_list])
        opens = np.array([float(k['open']) for k in klines_list])
        closes = np.array([float(k['close']) for k in klines_list])
        volumes = np.array([int(k['volume']) for k in klines_list])
        dates = np.array([k['date'] for k in klines_list])
        
        # 预分配结果数组
        processed_klines = []
        processed_klines.append(klines_list[0])  # 第一根K线直接添加
        merge_count = 0
        i = 1
        total_klines = len(klines_list)
        
        # 向量化处理包含关系
        while i < total_klines:
            current_time = time.time()
            if current_time - start_time > 180:  # 3分钟超时
                if log_callback:
                    log_callback(f"⚠️ 股票 {stock_code} 处理超时")
                break
            
            current_kline = klines_list[i]
            prev_kline = processed_klines[-1]
            
            # 跳过同日数据
            if prev_kline['date'] == current_kline['date']:
                processed_klines.append(current_kline)
                i += 1
                continue
            
            # 检查包含关系
            prev_high = float(prev_kline['high'])
            prev_low = float(prev_kline['low'])
            curr_high = float(current_kline['high'])
            curr_low = float(current_kline['low'])
            
            has_inclusion = False
            direction = 0
            
            # 向上包含：current包含prev
            if curr_high >= prev_high and curr_low <= prev_low and not (curr_high == prev_high and curr_low == prev_low):
                has_inclusion = True
                direction = 1
            # 向下包含：prev包含current
            elif prev_high >= curr_high and prev_low <= curr_low and not (prev_high == curr_high and prev_low == curr_low):
                has_inclusion = True
                direction = -1
            
            if has_inclusion:
                # 确定前一根K线的方向
                prev_direction = 1
                if len(processed_klines) >= 2:
                    prev_prev_kline = processed_klines[-2]
                    if float(prev_prev_kline['close']) < float(prev_prev_kline['open']):
                        prev_direction = -1
                
                # 合并K线
                merged_kline = merge_klines_vectorized_optimized(
                    prev_high, prev_low, float(prev_kline['open']), float(prev_kline['close']), int(prev_kline['volume']),
                    curr_high, curr_low, float(current_kline['open']), float(current_kline['close']), int(current_kline['volume']),
                    current_kline['date'], direction, prev_direction
                )
                
                # 替换最后一根K线
                processed_klines[-1] = merged_kline
                merge_count += 1
                i += 1  # 跳过当前K线
                
                # 检查连续包含
                consecutive_merge_count = 0
                while len(processed_klines) >= 2 and consecutive_merge_count < 10:
                    prev_check = processed_klines[-2]
                    curr_check = processed_klines[-1]
                    
                    # 跳过同日数据
                    if prev_check['date'] == curr_check['date']:
                        break
                    
                    prev_check_high = float(prev_check['high'])
                    prev_check_low = float(prev_check['low'])
                    curr_check_high = float(curr_check['high'])
                    curr_check_low = float(curr_check['low'])
                    
                    has_inclusion_again = False
                    direction_again = 0
                    
                    # 向上包含
                    if curr_check_high >= prev_check_high and curr_check_low <= prev_check_low and not (curr_check_high == prev_check_high and curr_check_low == prev_check_low):
                        has_inclusion_again = True
                        direction_again = 1
                    # 向下包含
                    elif prev_check_high >= curr_check_high and prev_check_low <= curr_check_low and not (prev_check_high == curr_check_high and prev_check_low == curr_check_low):
                        has_inclusion_again = True
                        direction_again = -1
                    
                    if has_inclusion_again:
                        # 确定前一根K线的方向
                        prev_direction_again = 1
                        if len(processed_klines) >= 3:
                            prev_prev_check = processed_klines[-3]
                            if float(prev_prev_check['close']) < float(prev_prev_check['open']):
                                prev_direction_again = -1
                        
                        # 合并K线
                        merged_kline_again = merge_klines_vectorized_optimized(
                            prev_check_high, prev_check_low, float(prev_check['open']), float(prev_check['close']), int(prev_check['volume']),
                            curr_check_high, curr_check_low, float(curr_check['open']), float(curr_check['close']), int(curr_check['volume']),
                            curr_check['date'], direction_again, prev_direction_again
                        )
                        
                        # 替换倒数第二根K线，删除最后一根
                        processed_klines[-2] = merged_kline_again
                        processed_klines.pop()
                        merge_count += 1
                        consecutive_merge_count += 1
                    else:
                        break
            else:
                # 无包含关系，添加当前K线
                processed_klines.append(current_kline)
                i += 1
            
            # 更新进度（每处理50根K线更新一次）
            if progress_callback and total_klines > 0 and i % 50 == 0:
                progress = int((min(i, total_klines) / total_klines) * 100)
                progress_callback(progress)
        
        processing_time = time.time() - processing_start
        
        # 批量保存处理后的K线数据
        save_start = time.time()
        save_result = save_processed_klines_batch_optimized(stock_code, processed_klines, 'processed')
        save_time = time.time() - save_start
        
        if not save_result:
            error_msg = "保存处理后数据失败"
            if log_callback:
                log_callback("❌ " + error_msg)
            return False, error_msg
        
        # 统计信息
        elapsed_time = time.time() - start_time
        processing_speed = total_klines / processing_time if processing_time > 0 else 0
        compression_rate = ((total_klines - len(processed_klines)) / total_klines * 100) if total_klines > 0 else 0
        
        if log_callback:
            log_callback(f"📊 统计信息:")
            log_callback(f"   原始K线数: {total_klines}")
            log_callback(f"   处理后K线数: {len(processed_klines)}")
            log_callback(f"   合并次数: {merge_count}")
            log_callback(f"   压缩率: {compression_rate:.1f}%")
            log_callback(f"   处理速度: {processing_speed:.1f} K线/秒")
            log_callback(f"   处理耗时: {processing_time:.2f} 秒")
            log_callback(f"   保存耗时: {save_time:.2f} 秒")
            log_callback(f"   总耗时: {elapsed_time:.2f} 秒")
            
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
        
        result_msg = f"包含关系处理完成，原始 {total_klines} 根K线，处理后 {len(processed_klines)} 根K线，合并 {merge_count} 次"
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

# 初始化数据库连接池
if __name__ != "__main__":
    initialize_db_pool()