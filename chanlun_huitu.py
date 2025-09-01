# chanlun_huitu.py - 缠论绘图模块（修复版）
import pandas as pd
import psycopg2
from psycopg2 import pool
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import logging
import os
from datetime import datetime, date
import numpy as np

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
        logging.FileHandler(os.path.join(log_dir, "chanlun_huitu.log"), encoding='utf-8')
    ]
)
logger = logging.getLogger("ChanlunHuitu")

# 数据库连接池
DB_POOL = None

def initialize_db_pool():
    """初始化数据库连接池"""
    global DB_POOL
    try:
        DB_POOL = psycopg2.pool.ThreadedConnectionPool(
            minconn=10,
            maxconn=50,
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

def get_fractals(stock_code, limit=50):
    """获取分型数据"""
    conn = None
    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            cur.execute("""
                SELECT date, fractal_type, high, low 
                FROM fractals 
                WHERE stock_code = %s 
                ORDER BY date DESC 
                LIMIT %s
            """, (stock_code, limit))
            
            rows = cur.fetchall()
            if not rows:
                return []
                
            # 转换为列表
            fractals = []
            for row in rows:
                fractal = {
                    'date': row[0],
                    'type': row[1],
                    'high': float(row[2]),
                    'low': float(row[3])
                }
                fractals.append(fractal)
            
            # 按日期升序排列
            fractals.sort(key=lambda x: x['date'])
            return fractals
            
    except Exception as e:
        logger.error(f"获取分型数据时出错: {str(e)}")
        return []
    finally:
        if conn:
            put_db_connection(conn)

def normalize_date(dt):
    """标准化日期格式"""
    if isinstance(dt, datetime):
        return dt.date()
    elif isinstance(dt, date):
        return dt
    else:
        return pd.to_datetime(dt).date()

def setup_chinese_font():
    """设置中文字体"""
    try:
        # 尝试多种中文字体
        chinese_fonts = [
            'SimHei',           # 黑体
            'Microsoft YaHei',  # 微软雅黑
            'STHeiti',          # 华文黑体
            'SimSun',           # 宋体
            'FangSong',         # 仿宋
            'KaiTi',            # 楷体
            'Arial Unicode MS', # Arial Unicode
            'DejaVu Sans'       # DejaVu Sans
        ]
        
        # 设置字体
        plt.rcParams['font.sans-serif'] = chinese_fonts
        plt.rcParams['axes.unicode_minus'] = False
        
        # 验证字体是否可用
        available_fonts = [f.name for f in fm.fontManager.ttflist]
        found_font = False
        for font in chinese_fonts:
            if font in available_fonts:
                plt.rcParams['font.sans-serif'] = [font]
                found_font = True
                break
        
        if not found_font:
            # 如果没有找到中文字体，使用系统默认字体并尝试解决负号问题
            plt.rcParams['axes.unicode_minus'] = False
            logger.warning("未找到合适的中文字体，可能显示为方块")
            
        return True
    except Exception as e:
        logger.error(f"设置中文字体时出错: {e}")
        try:
            plt.rcParams['axes.unicode_minus'] = False
            return True
        except:
            return False

def plot_candlestick_with_fractals(df, fractals, stock_code, display_options=None):
    """绘制带分型的K线图（修复版）"""
    if display_options is None:
        display_options = {'fractals': True}
    
    # 设置中文字体
    setup_chinese_font()
    
    # 创建图表（更大的尺寸以适应滚动）
    fig, ax = plt.subplots(figsize=(18, 12))  # 增大图表尺寸
    
    # 绘制K线主体
    width = 0.6
    
    for i in range(len(df)):
        kline = df.iloc[i]
        date_idx = i
        
        # 绘制影线
        ax.plot([date_idx, date_idx], [float(kline['low']), float(kline['high'])], color='black', linewidth=0.5)
        
        # 绘制实体
        open_price = float(kline['open'])
        close_price = float(kline['close'])
        
        if close_price >= open_price:  # 阳线（红色）
            ax.bar(date_idx, close_price - open_price, width, open_price, 
                  color='red', edgecolor='black', linewidth=0.5)
        else:  # 阴线（绿色）
            ax.bar(date_idx, open_price - close_price, width, close_price, 
                  color='green', edgecolor='black', linewidth=0.5)
    
    # 绘制分型标记（如果启用且有分型数据）
    if display_options.get('fractals', True) and fractals:
        # 创建日期到索引的映射
        date_to_index = {}
        for i in range(len(df)):
            date_key = normalize_date(df.iloc[i]['date'])
            date_to_index[date_key] = i
        
        top_fractals = []
        bottom_fractals = []
        
        matched_count = 0
        for fractal in fractals:
            date = fractal['date']
            fractal_type = fractal['type']
            high = float(fractal['high'])
            low = float(fractal['low'])
            
            # 使用标准化日期进行查找
            date_key = normalize_date(date)
            
            # 查找对应的x坐标
            if date_key in date_to_index:
                x_pos = date_to_index[date_key]
                y_pos = high if fractal_type == 'top' else low
                
                # 绘制分型标记（只使用图形，不使用文字避免乱码）
                if fractal_type == 'top':
                    ax.scatter(x_pos, y_pos, color='red', marker='v', s=150, zorder=5, alpha=0.8)
                    top_fractals.append((x_pos, y_pos))
                else:  # bottom
                    ax.scatter(x_pos, y_pos, color='blue', marker='^', s=150, zorder=5, alpha=0.8)
                    bottom_fractals.append((x_pos, y_pos))
                
                matched_count += 1
        
        # 连接分型点
        if len(top_fractals) > 1:
            top_x = [point[0] for point in top_fractals]
            top_y = [point[1] for point in top_fractals]
            ax.plot(top_x, top_y, color='red', linewidth=1.5, alpha=0.7, linestyle='--', label='顶分型连线')
        
        if len(bottom_fractals) > 1:
            bottom_x = [point[0] for point in bottom_fractals]
            bottom_y = [point[1] for point in bottom_fractals]
            ax.plot(bottom_x, bottom_y, color='blue', linewidth=1.5, alpha=0.7, linestyle='--', label='底分型连线')
        
        # 添加图例（只有当有线条时才添加）
        if len(top_fractals) > 1 or len(bottom_fractals) > 1:
            ax.legend()
    
    # 设置x轴
    ax.set_xlim(-2, len(df) + 1)  # 增加边距
    step = max(1, len(df)//15)  # 增加标签密度
    xticks = range(0, len(df), step)
    xtick_labels = [df.iloc[i]['date'].strftime('%Y-%m-%d') 
                   for i in range(0, len(df), step) if i < len(df)]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xtick_labels, rotation=45, fontsize=9)
    
    ax.set_title(f'{stock_code} K线图与分型标记', fontsize=16, pad=25)
    ax.grid(True, alpha=0.3)
    ax.set_ylabel('价格', fontsize=12)
    
    plt.tight_layout()
    
    return fig

def show_stock_chart(stock_code, display_options=None, kline_limit=100):
    """显示股票图表的主函数"""
    try:
        # 获取处理后的K线数据
        df = get_processed_klines(stock_code, 'processed', kline_limit)
        if df is None or len(df) == 0:
            # 如果没有处理后的数据，获取原始数据
            df = get_processed_klines(stock_code, 'original', kline_limit)
            if df is None or len(df) == 0:
                # 如果还没有数据，从daily_data获取
                df = get_stock_data(stock_code, kline_limit)
                if df is None or len(df) == 0:
                    return None, "未找到股票数据"
        
        # 获取分型数据
        fractals = get_fractals(stock_code, 50)
        
        # 创建图表
        fig = plot_candlestick_with_fractals(df, fractals, stock_code, display_options)
        return fig, "图表创建成功"
        
    except Exception as e:
        logger.error(f"显示股票图表时出错: {str(e)}")
        return None, f"显示图表时出错: {str(e)}"

# 初始化数据库连接池
if __name__ != "__main__":
    initialize_db_pool()