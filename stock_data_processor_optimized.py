# stock_data_processor_optimized.py - 优化的股票数据处理脚本
import os
import re
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import psycopg2
from psycopg2 import pool
import logging
import gc
import time
import threading

# 确保log目录存在
log_dir = "log"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# 尝试导入pandas_ta，如果没有则使用numpy实现
try:
    import pandas_ta as ta
    USE_PANDAS_TA = True
except ImportError:
    USE_PANDAS_TA = False
    print("未安装pandas_ta库，将使用numpy实现MACD计算")

# =============================================================================
# 🛠️ 配置区域
# =============================================================================

# 数据库配置
DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "dbname": "stock_db",
    "user": "postgres",
    "password": "shingowolf123"
}

# 处理参数配置
PROCESSING_CONFIG = {
    'MAX_WORKERS': 8,  # 减少线程数避免卡死
    'BATCH_INSERT_SIZE': 1000,
    'KEEP_DAYS': 3650,
    'MEMORY_CLEANUP_INTERVAL': 50
}

# MACD参数
MACD_PARAMS = {
    'fast': 12,
    'slow': 26, 
    'signal': 9
}

# 日志配置 - 只输出到文件，减少控制台输出
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(log_dir, "stock_processing.log"), encoding='utf-8')
    ]
)
logger = logging.getLogger("StockProcessor")

# =============================================================================
# 🗄️ 数据库连接池和全局变量
# =============================================================================

DB_POOL = None
processing_cancelled = False
processing_lock = threading.Lock()

def initialize_db_pool():
    """初始化数据库连接池"""
    global DB_POOL
    try:
        DB_POOL = psycopg2.pool.ThreadedConnectionPool(
            minconn=1,
            maxconn=PROCESSING_CONFIG['MAX_WORKERS'] + 2,
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

def test_database_connection():
    """测试数据库连接"""
    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            cur.execute("SELECT version()")
            version = cur.fetchone()
            put_db_connection(conn)
            return True, f"数据库连接成功: {version[0] if version else 'Unknown'}"
    except Exception as e:
        return False, f"数据库连接失败: {str(e)}"

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

# =============================================================================
# 📈 MACD计算函数
# =============================================================================

def calculate_macd(series, fast=12, slow=26, signal=9):
    """计算MACD指标"""
    if USE_PANDAS_TA and 'ta' in globals():
        # 使用pandas_ta库计算
        try:
            macd_result = ta.macd(series, fast=fast, slow=slow, signal=signal)
            if macd_result is not None and len(macd_result.columns) >= 3:
                return macd_result.iloc[:, 0], macd_result.iloc[:, 1], macd_result.iloc[:, 2]  # MACD, Signal, Histogram
        except:
            pass
    
    # 使用numpy实现MACD计算
    exp1 = series.ewm(span=fast, adjust=False).mean()
    exp2 = series.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    histogram = macd - signal_line
    
    return macd, signal_line, histogram

# =============================================================================
# 🔄 数据处理函数
# =============================================================================

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

def reset_cancel_flag():
    """重置取消标志"""
    global processing_cancelled
    with processing_lock:
        processing_cancelled = False

def safe_float_convert(value, default=0.0):
    """安全的浮点数转换"""
    try:
        if pd.isna(value):
            return default
        return float(value)
    except:
        return default

def safe_int_convert(value, default=0):
    """安全的整数转换"""
    try:
        if pd.isna(value):
            return default
        return int(float(value))
    except:
        return default

def process_txt_file(txt_path, last_db_date=None):
    """处理单个TXT文件，生成日线、周线、月线数据"""
    if is_processing_cancelled():
        return None, None, None
        
    try:
        # 1. 读取和解析TXT文件
        data = []
        cutoff_date = datetime.now() - timedelta(days=PROCESSING_CONFIG['KEEP_DAYS']) if PROCESSING_CONFIG['KEEP_DAYS'] > 0 else datetime(1900, 1, 1)
        
        with open(txt_path, 'r', encoding='gbk') as file:
            next(file)  # 跳过标题行
            line_count = 0
            for line in file:
                line_count += 1
                if line_count % 1000 == 0 and is_processing_cancelled():  # 每1000行检查一次取消
                    return None, None, None
                    
                match = re.match(
                    r'(\d{4}/\d{2}/\d{2})[\t\s]+([\d\.]+)[\t\s]+([\d\.]+)[\t\s]+([\d\.]+)[\t\s]+([\d\.]+)[\t\s]+([\d\.]+)[\t\s]+([\d\.]+)',
                    line.strip()
                )
                if match:
                    date_str = match.group(1)
                    file_date = datetime.strptime(date_str, '%Y/%m/%d')
                    # 如果是增量更新，只处理比数据库中更新的数据
                    if last_db_date is not None:
                        if file_date <= last_db_date:
                            continue
                    if file_date >= cutoff_date:
                        data.append({
                            'date': date_str,
                            'open': safe_float_convert(match.group(2)),
                            'high': safe_float_convert(match.group(3)),
                            'low': safe_float_convert(match.group(4)),
                            'close': safe_float_convert(match.group(5)),
                            'volume': safe_int_convert(match.group(6)),
                            'amount': safe_float_convert(match.group(7))
                        })
        
        if not data:  # 修复语法错误
            return None, None, None
            
        # 2. 转换为DataFrame并排序
        df = pd.DataFrame(data).sort_values('date')
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')
        
        # 3. 计算MACD（只在有新数据时计算）
        if len(df) > 0 and not is_processing_cancelled():
            macd, signal, histogram = calculate_macd(df['close'], 
                                                   MACD_PARAMS['fast'], 
                                                   MACD_PARAMS['slow'], 
                                                   MACD_PARAMS['signal'])
            df['macd'] = macd
            df['macd_signal'] = signal
            df['macd_histogram'] = histogram
        
        if is_processing_cancelled():
            return None, None, None
            
        # 4. 处理日线数据
        daily_df = df.copy()
        daily_df.reset_index(inplace=True)
        
        # 5. 处理周线数据
        weekly_df = resample_data(df, 'W-FRI')
        weekly_df.reset_index(inplace=True)
        
        # 6. 处理月线数据
        monthly_df = resample_data(df, 'ME')  # 修复警告：使用'ME'替代'M'
        monthly_df.reset_index(inplace=True)
        
        return daily_df, weekly_df, monthly_df
        
    except Exception as e:
        logger.error(f"处理文件 {txt_path} 时出错: {str(e)}")
        return None, None, None

def resample_data(df, freq):
    """周期转换（日线->周线/月线）"""
    if df.empty or is_processing_cancelled():
        return df
        
    try:
        if freq == 'W-FRI':  # 周线，周五为周末
            resampled = df.resample('W-FRI').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum',
                'amount': 'sum'
            })
        elif freq == 'ME':  # 月线（修复警告）
            resampled = df.resample('ME').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum',
                'amount': 'sum'
            })
        
        # 重新计算MACD（如果数据不为空）
        if not resampled.empty and 'close' in resampled.columns and not is_processing_cancelled():
            macd, signal, histogram = calculate_macd(resampled['close'], 
                                                   MACD_PARAMS['fast'], 
                                                   MACD_PARAMS['slow'], 
                                                   MACD_PARAMS['signal'])
            resampled['macd'] = macd
            resampled['macd_signal'] = signal
            resampled['macd_histogram'] = histogram
        
        return resampled
    except Exception as e:
        logger.error(f"重采样数据时出错: {str(e)}")
        return df

def get_last_data_date(stock_code, table_name):
    """获取数据库中该股票的最新数据日期"""
    if is_processing_cancelled():
        return None
        
    conn = None
    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            cur.execute(f"SELECT MAX(date) FROM {table_name} WHERE stock_code = %s", (stock_code,))
            result = cur.fetchone()
            return result[0] if result and result[0] else None
    except Exception as e:
        logger.error(f"获取股票 {stock_code} 最新数据日期时出错: {str(e)}")
        return None
    finally:
        if conn:
            put_db_connection(conn)

def save_to_database(stock_code, daily_df, weekly_df, monthly_df):
    """保存数据到数据库"""
    if is_processing_cancelled():
        return False
        
    conn = None
    try:
        conn = get_db_connection()
        
        with conn.cursor() as cur:
            # 批量插入日线数据
            if daily_df is not None and not daily_df.empty and not is_processing_cancelled():
                daily_data = []
                for _, row in daily_df.iterrows():
                    if is_processing_cancelled():
                        return False
                    daily_data.append((
                        stock_code, 
                        row['date'], 
                        safe_float_convert(row['open']), 
                        safe_float_convert(row['high']), 
                        safe_float_convert(row['low']), 
                        safe_float_convert(row['close']),
                        safe_int_convert(row['volume']), 
                        safe_float_convert(row.get('amount', 0)),
                        safe_float_convert(row.get('macd', 0)), 
                        safe_float_convert(row.get('macd_signal', 0)), 
                        safe_float_convert(row.get('macd_histogram', 0))
                    ))
                
                cur.executemany("""
                    INSERT INTO daily_data (
                        stock_code, date, open, high, low, close, volume, amount,
                        macd, macd_signal, macd_histogram
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (stock_code, date) DO UPDATE SET
                        open = EXCLUDED.open,
                        high = EXCLUDED.high,
                        low = EXCLUDED.low,
                        close = EXCLUDED.close,
                        volume = EXCLUDED.volume,
                        amount = EXCLUDED.amount,
                        macd = EXCLUDED.macd,
                        macd_signal = EXCLUDED.macd_signal,
                        macd_histogram = EXCLUDED.macd_histogram
                """, daily_data)
            
            # 批量插入周线数据
            if weekly_df is not None and not weekly_df.empty and not is_processing_cancelled():
                weekly_data = []
                for _, row in weekly_df.iterrows():
                    if is_processing_cancelled():
                        return False
                    weekly_data.append((
                        stock_code, 
                        row['date'], 
                        safe_float_convert(row['open']), 
                        safe_float_convert(row['high']), 
                        safe_float_convert(row['low']), 
                        safe_float_convert(row['close']),
                        safe_int_convert(row['volume']), 
                        safe_float_convert(row.get('amount', 0)),
                        safe_float_convert(row.get('macd', 0)), 
                        safe_float_convert(row.get('macd_signal', 0)), 
                        safe_float_convert(row.get('macd_histogram', 0))
                    ))
                
                cur.executemany("""
                    INSERT INTO weekly_data (
                        stock_code, date, open, high, low, close, volume, amount,
                        macd, macd_signal, macd_histogram
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (stock_code, date) DO UPDATE SET
                        open = EXCLUDED.open,
                        high = EXCLUDED.high,
                        low = EXCLUDED.low,
                        close = EXCLUDED.close,
                        volume = EXCLUDED.volume,
                        amount = EXCLUDED.amount,
                        macd = EXCLUDED.macd,
                        macd_signal = EXCLUDED.macd_signal,
                        macd_histogram = EXCLUDED.macd_histogram
                """, weekly_data)
            
            # 批量插入月线数据
            if monthly_df is not None and not monthly_df.empty and not is_processing_cancelled():
                monthly_data = []
                for _, row in monthly_df.iterrows():
                    if is_processing_cancelled():
                        return False
                    monthly_data.append((
                        stock_code, 
                        row['date'], 
                        safe_float_convert(row['open']), 
                        safe_float_convert(row['high']), 
                        safe_float_convert(row['low']), 
                        safe_float_convert(row['close']),
                        safe_int_convert(row['volume']), 
                        safe_float_convert(row.get('amount', 0)),
                        safe_float_convert(row.get('macd', 0)), 
                        safe_float_convert(row.get('macd_signal', 0)), 
                        safe_float_convert(row.get('macd_histogram', 0))
                    ))
                
                cur.executemany("""
                    INSERT INTO monthly_data (
                        stock_code, date, open, high, low, close, volume, amount,
                        macd, macd_signal, macd_histogram
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (stock_code, date) DO UPDATE SET
                        open = EXCLUDED.open,
                        high = EXCLUDED.high,
                        low = EXCLUDED.low,
                        close = EXCLUDED.close,
                        volume = EXCLUDED.volume,
                        amount = EXCLUDED.amount,
                        macd = EXCLUDED.macd,
                        macd_signal = EXCLUDED.macd_signal,
                        macd_histogram = EXCLUDED.macd_histogram
                """, monthly_data)
            
            conn.commit()
            return True
            
    except Exception as e:
        logger.error(f"保存股票 {stock_code} 数据到数据库时出错: {str(e)}")
        if conn:
            conn.rollback()
        return False
    finally:
        if conn:
            put_db_connection(conn)

def process_single_stock(txt_path, progress_callback=None):
    """处理单个股票文件的完整流程"""
    if is_processing_cancelled():
        return False, f"处理被取消: {os.path.basename(txt_path)}"
        
    try:
        stock_code = os.path.basename(txt_path).replace('.txt', '')
        
        # 添加详细的调试信息
        debug_info = []
        
        # 检查文件是否存在
        if not os.path.exists(txt_path):
            return False, f"文件不存在: {txt_path}"
            
        # 获取文件修改时间
        try:
            file_mod_time = datetime.fromtimestamp(os.path.getmtime(txt_path))
            debug_info.append(f"文件修改时间: {file_mod_time}")
        except Exception as e:
            return False, f"无法获取文件修改时间 {txt_path}: {str(e)}"
            
        # 获取数据库最新数据时间
        try:
            last_db_date = get_last_data_date(stock_code, 'daily_data')
            debug_info.append(f"数据库最新数据时间: {last_db_date}")
        except Exception as e:
            return False, f"无法获取数据库最新数据日期 {stock_code}: {str(e)}"
        
        # 记录比较结果
        debug_info.append(f"文件时间: {file_mod_time.date()}, 数据库时间: {last_db_date}")
        
        # 判断是否需要更新
        needs_update = False
        update_reason = ""
        
        if last_db_date is None:
            needs_update = True
            update_reason = "数据库中无该股票数据"
            debug_info.append("需要更新: 数据库中无该股票数据")
        elif file_mod_time.date() > last_db_date:
            needs_update = True
            update_reason = f"文件更新 ({file_mod_time.date()} > {last_db_date})"
            debug_info.append("需要更新: 文件比数据库新")
        else:
            update_reason = f"无需更新 (文件时间 {file_mod_time.date()} <= 数据库时间 {last_db_date})"
            debug_info.append("无需更新: 文件不比数据库新")
        
        debug_info.append(f"更新判断: {update_reason}")
        
        # 如果数据库中没有该股票数据，或者文件更新时间晚于数据库最新数据，则处理
        if needs_update:
            # 处理TXT文件生成三种周期数据
            try:
                daily_df, weekly_df, monthly_df = process_txt_file(txt_path, last_db_date)
                debug_info.append(f"数据处理结果: daily_df={daily_df is not None}, rows={len(daily_df) if daily_df is not None else 0}")
            except Exception as e:
                debug_info.append(f"数据处理异常: {str(e)}")
                return False, f"处理TXT文件失败 {stock_code}: {str(e)} 原始信息: {'; '.join(debug_info)}"
            
            if is_processing_cancelled():
                return False, f"处理被取消: {stock_code} 原始信息: {'; '.join(debug_info)}"
                
            if daily_df is None or len(daily_df) == 0:
                return False, f"股票 {stock_code} 数据处理失败: 未生成有效数据 原始信息: {'; '.join(debug_info)}"
            
            # 保存到数据库
            try:
                success = save_to_database(stock_code, daily_df, weekly_df, monthly_df)
                debug_info.append(f"数据库保存结果: {success}")
            except Exception as e:
                debug_info.append(f"数据库保存异常: {str(e)}")
                return False, f"保存到数据库失败 {stock_code}: {str(e)} 原始信息: {'; '.join(debug_info)}"
            
            if is_processing_cancelled():
                return False, f"处理被取消: {stock_code} 原始信息: {'; '.join(debug_info)}"
                
            if success:
                return True, f"股票 {stock_code} 处理成功 原始信息: {'; '.join(debug_info)}"
            else:
                return False, f"股票 {stock_code} 数据库保存失败 原始信息: {'; '.join(debug_info)}"
        else:
            return True, f"股票 {stock_code} {update_reason} 原始信息: {'; '.join(debug_info)}"
            
    except Exception as e:
        error_msg = f"处理股票 {stock_code} 时发生异常: {str(e)}"
        logger.error(error_msg)
        return False, error_msg

def get_all_table_names():
    """获取数据库中所有表名"""
    conn = None
    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            cur.execute("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public'
            """)
            return [row[0] for row in cur.fetchall()]
    except Exception as e:
        logger.error(f"获取表名列表时出错: {str(e)}")
        return []
    finally:
        if conn:
            put_db_connection(conn)

def clear_database():
    """清空数据库所有表"""
    conn = None
    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            # 获取所有表名
            tables = get_all_table_names()
            for table in tables:
                cur.execute(f"TRUNCATE TABLE {table} RESTART IDENTITY CASCADE")
            conn.commit()
            return True, f"成功清空 {len(tables)} 个表"
    except Exception as e:
        error_msg = f"清空数据库时出错: {str(e)}"
        logger.error(error_msg)
        if conn:
            conn.rollback()
        return False, error_msg
    finally:
        if conn:
            put_db_connection(conn)

def drop_table(table_name):
    """删除指定表"""
    conn = None
    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            cur.execute(f"DROP TABLE IF EXISTS {table_name} CASCADE")
            conn.commit()
            return True, f"成功删除表 {table_name}"
    except Exception as e:
        error_msg = f"删除表 {table_name} 时出错: {str(e)}"
        logger.error(error_msg)
        if conn:
            conn.rollback()
        return False, error_msg
    finally:
        if conn:
            put_db_connection(conn)

def batch_process_stocks(txt_dir, progress_callback=None, log_callback=None):
    """批量处理所有股票文件"""
    global processing_cancelled
    reset_cancel_flag()  # 重置取消标志
    
    logger.info("🚀 开始批量处理股票数据...")
    start_time = time.time()
    
    try:
        # 获取所有TXT文件
        if not os.path.exists(txt_dir):
            error_msg = f"文件夹不存在: {txt_dir}"
            if log_callback:
                log_callback("❌ " + error_msg)
            return False, error_msg
            
        txt_files = [
            os.path.join(txt_dir, f) 
            for f in os.listdir(txt_dir) 
            if f.endswith('.txt')
        ]
        
        logger.info(f"📋 发现 {len(txt_files)} 个TXT文件待处理")
        
        if not txt_files:
            if log_callback:
                log_callback("⚠️ 没有找到TXT文件")
            return False, "没有找到TXT文件"
        
        # 分批处理
        total_files = len(txt_files)
        successful = 0
        failed_files = []
        completed = 0
        last_progress_update = 0
        UPDATE_INTERVAL = 5  # 每5%更新一次进度
        
        # 使用多线程处理
        with ThreadPoolExecutor(max_workers=PROCESSING_CONFIG['MAX_WORKERS']) as executor:
            # 提交所有任务
            future_to_file = {
                executor.submit(process_single_stock, txt_path): txt_path 
                for txt_path in txt_files
            }
            
            # 处理完成的任务
            for future in as_completed(future_to_file):
                # 检查是否被取消
                if is_processing_cancelled():
                    if log_callback:
                        log_callback("❌ 处理已被用户取消")
                    break
                    
                txt_path = future_to_file[future]
                try:
                    # 添加超时控制
                    success, message = future.result(timeout=120)  # 2分钟超时
                    if success:
                        successful += 1
                    else:
                        failed_files.append((os.path.basename(txt_path), message))
                    
                    completed += 1
                    # 更新进度（减少更新频率）
                    if progress_callback and total_files > 0:
                        current_progress = int((completed / total_files) * 100)
                        if current_progress - last_progress_update >= UPDATE_INTERVAL or current_progress == 100:
                            progress_callback(current_progress)
                            last_progress_update = current_progress
                    
                    # 记录日志
                    if log_callback:
                        log_callback(message)
                        
                    # 定期清理内存
                    if successful % PROCESSING_CONFIG['MEMORY_CLEANUP_INTERVAL'] == 0:
                        gc.collect()
                        
                except Exception as e:
                    error_msg = f"处理文件 {txt_path} 时发生异常: {str(e)}"
                    failed_files.append((os.path.basename(txt_path), error_msg))
                    if log_callback:
                        log_callback("❌ " + error_msg)
                        
                # 再次检查取消状态
                if is_processing_cancelled():
                    if log_callback:
                        log_callback("❌ 处理已被用户取消")
                    break
    
    except Exception as e:
        error_msg = f"批量处理过程中发生错误: {str(e)}"
        if log_callback:
            log_callback("❌ " + error_msg)
        logger.error(error_msg)
        return False, error_msg
    
    # 统计结果
    total_time = time.time() - start_time
    if is_processing_cancelled():
        result_msg = f"⚠️ 处理被取消! 已处理: {completed}/{total_files}, 成功: {successful}, 耗时: {total_time:.2f}秒"
    else:
        result_msg = f"🎉 批量处理完成! 总文件数: {total_files}, 成功: {successful}, 失败: {len(failed_files)}, 耗时: {total_time:.2f}秒"
    
    if log_callback:
        log_callback(result_msg)
        if failed_files and not is_processing_cancelled():
            log_callback("❌ 失败文件列表:")
            for filename, error in failed_files[:10]:  # 只显示前10个错误
                log_callback(f"  - {filename}: {error}")
            if len(failed_files) > 10:
                log_callback(f"  ... 还有 {len(failed_files) - 10} 个错误文件")
    
    return not is_processing_cancelled(), result_msg

# 初始化数据库连接池和创建表
if __name__ != "__main__":
    initialize_db_pool()
    create_tables_if_not_exists()