# test_fractal_fixed_logic.py - 修正分型逻辑测试 (修正版)
import pandas as pd
import psycopg2
from datetime import datetime, date
import matplotlib.pyplot as plt
import numpy as np

# 数据库配置
DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "dbname": "stock_db",
    "user": "postgres",
    "password": "shingowolf123"
}

def get_db_connection():
    """获取数据库连接"""
    return psycopg2.connect(**DB_CONFIG)

def get_processed_klines_for_stock(stock_code, limit=100):
    """获取处理后的K线数据"""
    conn = None
    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            cur.execute("""
                SELECT date, open, high, low, close, volume 
                FROM processed_klines 
                WHERE stock_code = %s AND kline_type = 'processed'
                ORDER BY date
                LIMIT %s
            """, (stock_code, limit))
            
            rows = cur.fetchall()
            if not rows:
                return None
                
            # 转换为DataFrame
            df = pd.DataFrame(rows, columns=['date', 'open', 'high', 'low', 'close', 'volume'])
            df['date'] = pd.to_datetime(df['date'])
            return df
            
    except Exception as e:
        print(f"获取K线数据时出错: {str(e)}")
        return None
    finally:
        if conn:
            conn.close()

def format_date_for_display(date_obj):
    """格式化日期用于显示"""
    if hasattr(date_obj, 'strftime'):
        return date_obj.strftime('%Y-%m-%d')
    else:
        # 处理numpy datetime64类型
        return str(date_obj)[:10]

def is_top_fractal(highs, lows, i):
    """判断是否为顶分型"""
    if i <= 0 or i >= len(highs) - 1:
        return False
    
    left_high, middle_high, right_high = highs[i-1], highs[i], highs[i+1]
    left_low, middle_low, right_low = lows[i-1], lows[i], lows[i+1]
    
    is_contained = (middle_high >= left_high and middle_high >= right_high and
                   middle_low >= left_low and middle_low >= right_low)
    
    not_equal = (middle_high > left_high or middle_high > right_high or
                middle_low > left_low or middle_low > right_low)
    
    return is_contained and not_equal

def is_bottom_fractal(highs, lows, i):
    """判断是否为底分型"""
    if i <= 0 or i >= len(lows) - 1:
        return False
    
    left_high, middle_high, right_high = highs[i-1], highs[i], highs[i+1]
    left_low, middle_low, right_low = lows[i-1], lows[i], lows[i+1]
    
    is_contained = (middle_high <= left_high and middle_high <= right_high and
                   middle_low <= left_low and middle_low <= right_low)
    
    not_equal = (middle_high < left_high or middle_high < right_high or
                middle_low < left_low or middle_low < right_low)
    
    return is_contained and not_equal

def detect_fractals_correct_logic(highs, lows, dates):
    """修正后的正确分型检测逻辑"""
    print("=== 修正后的分型检测逻辑 ===")
    fractals = []
    
    i = 1  # 从第二个K线开始检测
    next_valid_index = 1  # 下一个有效的检测位置
    
    print(f"开始检测，数据长度: {len(highs)}")
    
    while i < len(highs) - 1:
        date_str = format_date_for_display(dates[i])
        left_date_str = format_date_for_display(dates[i-1])
        right_date_str = format_date_for_display(dates[i+1])
        
        print(f"  检查位置 {i} ({date_str}): ", end="")
        
        # 检查是否可以检测
        if i < next_valid_index:
            print(f"被跳过 (等待位置 {next_valid_index})")
            i += 1
            continue
            
        print(f"检测中...")
        print(f"    左侧K线{i-1} ({left_date_str}): H={highs[i-1]:.2f}, L={lows[i-1]:.2f}")
        print(f"    中间K线{i} ({date_str}): H={highs[i]:.2f}, L={lows[i]:.2f}")
        print(f"    右侧K线{i+1} ({right_date_str}): H={highs[i+1]:.2f}, L={lows[i+1]:.2f}")
        
        is_top = is_top_fractal(highs, lows, i)
        is_bottom = is_bottom_fractal(highs, lows, i)
        
        if is_top:
            print(f"    ✅ 检测到顶分型")
            fractal = {
                'date': dates[i],
                'type': 'top',
                'high': float(highs[i]),
                'low': float(lows[i]),
                'index': i
            }
            fractals.append(fractal)
            # 正确的跳转逻辑：
            # 分型占用K线 i-1, i, i+1 (索引)
            # 需要间隔一根K线，所以下一个分型应该从 i+3 开始检测
            next_valid_index = i + 4  # 下一个有效检测位置
            print(f"    分型占用K线 {i-1}, {i}, {i+1}")
            print(f"    下一个检测位置: {next_valid_index}")
            i += 1
        elif is_bottom:
            print(f"    ✅ 检测到底分型")
            fractal = {
                'date': dates[i],
                'type': 'bottom',
                'high': float(highs[i]),
                'low': float(lows[i]),
                'index': i
            }
            fractals.append(fractal)
            # 同样的逻辑
            next_valid_index = i + 4  # 下一个有效检测位置
            print(f"    分型占用K线 {i-1}, {i}, {i+1}")
            print(f"    下一个检测位置: {next_valid_index}")
            i += 1
        else:
            print(f"    ❌ 不是分型")
            i += 1
    
    return fractals

def plot_candlestick_with_fractals(df, fractals, stock_code):
    """绘制K线图和分型标记"""
    # 设置中文字体
    try:
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
        plt.rcParams['axes.unicode_minus'] = False
    except:
        plt.rcParams['axes.unicode_minus'] = False
    
    # 创建图表
    fig, ax = plt.subplots(figsize=(15, 8))
    
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
    
    # 绘制分型标记
    if fractals:
        for fractal in fractals:
            i = fractal['index']
            fractal_type = fractal['type']
            price = fractal['high'] if fractal_type == 'top' else fractal['low']
            
            if fractal_type == 'top':
                ax.scatter(i, price, color='red', marker='v', s=100, zorder=5)
                ax.annotate('🔺', (i, price), 
                           xytext=(0, 10), textcoords='offset points',
                           ha='center', va='bottom', fontsize=12, color='red', fontweight='bold')
            else:  # bottom
                ax.scatter(i, price, color='blue', marker='^', s=100, zorder=5)
                ax.annotate('🔻', (i, price), 
                           xytext=(0, -20), textcoords='offset points',
                           ha='center', va='top', fontsize=12, color='blue', fontweight='bold')
    
    # 设置x轴
    ax.set_xlim(-1, len(df))
    step = max(1, len(df)//10)
    xticks = range(0, len(df), step)
    xtick_labels = [format_date_for_display(df.iloc[i]['date']) 
                   for i in range(0, len(df), step) if i < len(df)]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xtick_labels, rotation=45, fontsize=8)
    
    ax.set_title(f'{stock_code} K线图与分型标记 (修正逻辑测试)')
    ax.grid(True, alpha=0.3)
    ax.set_ylabel('价格')
    
    plt.tight_layout()
    
    return fig

def main():
    """主函数"""
    stock_code = "SH#600970"
    
    print(f"=== 修正分型逻辑测试 ===")
    print(f"股票代码: {stock_code}")
    
    # 获取K线数据
    df = get_processed_klines_for_stock(stock_code, 50)
    if df is None or len(df) == 0:
        print("❌ 未找到K线数据")
        return
    
    print(f"\n📊 获取到 {len(df)} 条K线数据:")
    for i, row in df.iterrows():
        date_str = format_date_for_display(row['date'])
        print(f"  {i:2d}. {date_str}: H={row['high']:.2f}, L={row['low']:.2f}")
    
    # 使用修正后的逻辑检测分型
    print(f"\n🔍 使用修正后的逻辑检测分型:")
    highs = df['high'].values
    lows = df['low'].values
    dates = df['date'].values
    
    detected_fractals = detect_fractals_correct_logic(highs, lows, dates)
    
    print(f"\n📈 检测到的分型:")
    for i, fractal in enumerate(detected_fractals):
        date_str = format_date_for_display(fractal['date'])
        print(f"  {i+1}. K线{fractal['index']} {date_str} "
              f"{fractal['type']} H:{fractal['high']:.2f} L:{fractal['low']:.2f}")
    
    # 检查分型间隔
    if len(detected_fractals) > 1:
        print(f"\n🔍 分型间隔检查:")
        for i in range(len(detected_fractals) - 1):
            current = detected_fractals[i]
            next_fractal = detected_fractals[i+1]
            index_diff = next_fractal['index'] - current['index']
            current_date = format_date_for_display(current['date'])
            next_date = format_date_for_display(next_fractal['date'])
            print(f"  K线{current['index']}({current_date}) -> K线{next_fractal['index']}({next_date}) = 间隔 {index_diff} 根K线")
            if index_diff < 3:
                print(f"    ❌ 警告：K线间隔过小！")
    
    # 绘制图表
    print(f"\n📊 绘制图表...")
    fig = plot_candlestick_with_fractals(df, detected_fractals, stock_code)
    plt.show()
    print("✅ 图表显示完成")

if __name__ == "__main__":
    main()