import psycopg2
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# 数据库配置
DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "dbname": "stock_db",
    "user": "postgres",
    "password": "shingowolf123"
}

def get_processed_klines(stock_code, kline_type='processed', limit=100):
    """获取处理后的K线数据"""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
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
        cur.close()
        conn.close()
        return df
        
    except Exception as e:
        print(f"获取处理后的K线数据时出错: {str(e)}")
        return None

def detect_fractals_absolute_correct(klines_df):
    """绝对正确的分型检测（严格按照缠论原则）"""
    if len(klines_df) < 3:
        print("❌ 数据不足，至少需要3根K线")
        return []
    
    fractals = []
    i = 1  # 从第2根K线开始检查（索引1）
    
    print(f"📊 开始检测分型，共 {len(klines_df)} 根K线")
    print(f"   索引范围: 1 到 {len(klines_df) - 2}")
    
    # 严格按照缠论标准检测分型
    while i < len(klines_df) - 1:
        print(f"   检查索引 {i}: {klines_df.iloc[i]['date'].strftime('%Y-%m-%d')}")
        
        prev_kline = klines_df.iloc[i-1]
        curr_kline = klines_df.iloc[i]
        next_kline = klines_df.iloc[i+1]
        
        # 顶分型：中间K线完全包含左右两根K线
        is_top_fractal = (
            curr_kline['high'] >= prev_kline['high'] and 
            curr_kline['high'] >= next_kline['high'] and
            curr_kline['low'] >= prev_kline['low'] and
            curr_kline['low'] >= next_kline['low'] and
            (curr_kline['high'] > prev_kline['high'] or curr_kline['high'] > next_kline['high'] or
             curr_kline['low'] > prev_kline['low'] or curr_kline['low'] > next_kline['low'])
        )
        
        # 底分型：中间K线完全包含左右两根K线
        is_bottom_fractal = (
            curr_kline['low'] <= prev_kline['low'] and 
            curr_kline['low'] <= next_kline['low'] and
            curr_kline['high'] <= prev_kline['high'] and
            curr_kline['high'] <= next_kline['high'] and
            (curr_kline['low'] < prev_kline['low'] or curr_kline['low'] < next_kline['low'] or
             curr_kline['high'] < prev_kline['high'] or curr_kline['high'] < next_kline['high'])
        )
        
        if is_top_fractal:
            fractal = {
                'date': curr_kline['date'],
                'type': 'top',
                'high': float(curr_kline['high']),
                'low': float(curr_kline['low']),
                'index': i
            }
            fractals.append(fractal)
            print(f"   🔺 顶分型: {curr_kline['date'].strftime('%Y-%m-%d')} (索引{i})")
            # 绝对正确的处理：检测到分型后，直接跳到 i+2（跳过下一根K线）
            i += 2  # 从1,2,3组成分型后，直接跳到索引4（即第5根K线）
            
        elif is_bottom_fractal:
            fractal = {
                'date': curr_kline['date'],
                'type': 'bottom',
                'high': float(curr_kline['high']),
                'low': float(curr_kline['low']),
                'index': i
            }
            fractals.append(fractal)
            print(f"   🔻 底分型: {curr_kline['date'].strftime('%Y-%m-%d')} (索引{i})")
            # 绝对正确的处理：检测到分型后，直接跳到 i+2（跳过下一根K线）
            i += 2  # 从1,2,3组成分型后，直接跳到索引4（即第5根K线）
        else:
            i += 1  # 正常移动到下一根K线
    
    return fractals

def setup_chinese_font():
    """设置中文字体"""
    try:
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
        plt.rcParams['axes.unicode_minus'] = False
        return True
    except:
        plt.rcParams['axes.unicode_minus'] = False
        return False

def plot_candlestick_with_fractals(df, fractals, stock_code):
    """绘制K线图并标记分型"""
    # 设置中文字体
    setup_chinese_font()
    
    # 创建图表
    fig, ax = plt.subplots(figsize=(15, 8))
    
    # 绘制K线
    for i in range(len(df)):
        kline = df.iloc[i]
        date_idx = i
        
        # 绘制影线
        ax.plot([date_idx, date_idx], [kline['low'], kline['high']], color='black', linewidth=0.5)
        
        # 绘制实体
        open_price = kline['open']
        close_price = kline['close']
        
        if close_price >= open_price:  # 阳线（红色）
            ax.bar(date_idx, close_price - open_price, 0.6, open_price, 
                  color='red', edgecolor='black', linewidth=0.5)
        else:  # 阴线（绿色）
            ax.bar(date_idx, open_price - close_price, 0.6, close_price, 
                  color='green', edgecolor='black', linewidth=0.5)
    
    # 标记分型
    top_fractals = [f for f in fractals if f['type'] == 'top']
    bottom_fractals = [f for f in fractals if f['type'] == 'bottom']
    
    # 绘制顶分型标记
    for fractal in top_fractals:
        ax.scatter(fractal['index'], fractal['high'], color='red', marker='v', s=100, zorder=5)
        ax.annotate('TOP', (fractal['index'], fractal['high']), 
                   xytext=(0, 10), textcoords='offset points',
                   ha='center', va='bottom', fontsize=10, color='red', fontweight='bold')
    
    # 绘制底分型标记
    for fractal in bottom_fractals:
        ax.scatter(fractal['index'], fractal['low'], color='blue', marker='^', s=100, zorder=5)
        ax.annotate('BOTTOM', (fractal['index'], fractal['low']), 
                   xytext=(0, -20), textcoords='offset points',
                   ha='center', va='top', fontsize=10, color='blue', fontweight='bold')
    
    # 连接分型点（如果有的话）
    if len(top_fractals) > 1:
        top_indices = [f['index'] for f in top_fractals]
        top_prices = [f['high'] for f in top_fractals]
        ax.plot(top_indices, top_prices, color='red', linewidth=1, alpha=0.7, linestyle='--', label='顶分型连线')
    
    if len(bottom_fractals) > 1:
        bottom_indices = [f['index'] for f in bottom_fractals]
        bottom_prices = [f['low'] for f in bottom_fractals]
        ax.plot(bottom_indices, bottom_prices, color='blue', linewidth=1, alpha=0.7, linestyle='--', label='底分型连线')
    
    # 添加图例
    ax.legend()
    
    # 设置x轴
    ax.set_xlim(-1, len(df))
    step = max(1, len(df)//10)
    xticks = range(0, len(df), step)
    xtick_labels = [df.iloc[i]['date'].strftime('%Y-%m-%d') for i in xticks]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xtick_labels, rotation=45)
    
    # 设置标题和标签
    ax.set_title(f'{stock_code} K线图与分型标记', fontsize=14, fontweight='bold')
    ax.set_ylabel('价格', fontsize=12)
    ax.set_xlabel('日期', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # 调整布局
    plt.tight_layout()
    
    # 显示图表
    plt.show()
    
    return fig

def test_fractal_detection():
    """测试分型识别"""
    stock_code = "SH#603712"
    print(f"=== 测试股票 {stock_code} 的分型识别 ===")
    
    # 获取处理后的K线数据（最近100天）
    df = get_processed_klines(stock_code, 'processed', 100)
    if df is None or len(df) == 0:
        print(f"❌ 未找到股票 {stock_code} 的处理后数据")
        return
    
    print(f"📊 获取到 {len(df)} 条处理后的K线数据")
    
    # 显示数据范围
    print(f"📅 数据时间范围: {df['date'].min().strftime('%Y-%m-%d')} 到 {df['date'].max().strftime('%Y-%m-%d')}")
    
    # 显示前10条数据
    print(f"\n📋 前10条K线数据:")
    for i in range(min(10, len(df))):
        kline = df.iloc[i]
        direction = "📈" if kline['close'] >= kline['open'] else "📉"
        print(f"   {i:2d}. {kline['date'].strftime('%Y-%m-%d')} H:{kline['high']:6.2f} L:{kline['low']:6.2f} {direction}")
    
    # 检测分型
    print(f"\n🔍 开始检测分型...")
    fractals = detect_fractals_absolute_correct(df)
    
    print(f"\n✅ 分型检测完成!")
    print(f"   发现分型数量: {len(fractals)}")
    if fractals:
        top_count = len([f for f in fractals if f['type'] == 'top'])
        bottom_count = len([f for f in fractals if f['type'] == 'bottom'])
        print(f"   顶分型: {top_count} 个")
        print(f"   底分型: {bottom_count} 个")
        
        # 检查分型间隔
        if len(fractals) > 1:
            print(f"\n📋 分型间隔检查:")
            for i in range(1, len(fractals)):
                prev_fractal = fractals[i-1]
                curr_fractal = fractals[i]
                interval = curr_fractal['index'] - prev_fractal['index']
                print(f"   索引{prev_fractal['index']} → 索引{curr_fractal['index']} 间隔: {interval-1} 根K线")
    
    # 显示所有分型
    if fractals:
        print(f"\n📋 所有分型列表:")
        for i, fractal in enumerate(fractals):
            fractal_type = "🔺" if fractal['type'] == 'top' else "🔻"
            print(f"   {i+1:2d}. {fractal_type} {fractal['date'].strftime('%Y-%m-%d')} H:{fractal['high']:6.2f} L:{fractal['low']:6.2f} (索引:{fractal['index']})")
    
    # 绘制图表
    print(f"\n📊 正在生成K线图...")
    try:
        fig = plot_candlestick_with_fractals(df, fractals, stock_code)
        print(f"✅ K线图已生成并显示")
    except Exception as e:
        print(f"❌ 生成K线图时出错: {str(e)}")
    
    print(f"\n🎉 测试完成!")

if __name__ == "__main__":
    test_fractal_detection()
