import streamlit as st
import pandas as pd
import numpy as np
import io
import time
import warnings
from datetime import datetime, timedelta
from collections import deque
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re
import json

warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------
# 1. 核心匹配引擎 (根据新需求更新)
# ---------------------------------------------------------

class HedgeMatchingEngine:
    """套保匹配引擎 - 更新版"""
    
    def __init__(self):
        self.df_paper = None
        self.df_physical = None
        self.df_paper_net = None
        self.df_relations = None
        self.df_physical_updated = None
        self.open_positions_summary = None  # 开仓汇总
        self.close_positions_summary = None  # 平仓汇总
        
    def clean_str(self, series):
        """清洗字符串 - 处理NaN和float值"""
        if series.empty:
            return pd.Series([], dtype=str)
            
        # 先转换为字符串，处理NaN
        series = series.copy()
        series = series.fillna('')
        
        # 安全转换为字符串
        def safe_str(x):
            if isinstance(x, (int, float)):
                return str(x)
            return str(x)
        
        series = series.apply(safe_str)
        return series.str.strip().str.upper().replace('NAN', '').replace('NONE', '')
    
    def standardize_month(self, series):
        """标准化月份格式 - 增强错误处理"""
        if series.empty:
            return pd.Series([], dtype=str)
            
        s = series.copy()
        s = s.fillna('')
        
        # 安全转换为字符串
        def safe_str(x):
            if isinstance(x, (int, float)):
                return str(x)
            return str(x)
        
        s = s.apply(safe_str)
        s = s.str.strip().str.upper()
        s = s.str.replace('-', ' ', regex=False).str.replace('/', ' ', regex=False)
        dates = pd.to_datetime(s, errors='coerce', format='mixed')
        result = dates.dt.strftime('%b %y').str.upper()
        mask_invalid = dates.isna()
        
        if mask_invalid.any():
            invalid = s[mask_invalid]
            def swap_if_match(val):
                if not isinstance(val, str):
                    return val
                m = re.match(r'^(\d{2})\s*([A-Z]{3})$', val)
                if m:
                    yr, mon = m.groups()
                    return f"{mon} {yr}"
                return val
            swapped = invalid.apply(swap_if_match)
            swapped_dates = pd.to_datetime(swapped, errors='coerce')
            swapped_formatted = swapped_dates.dt.strftime('%b %y').str.upper()
            result.loc[mask_invalid & swapped_dates.notna()] = swapped_formatted.loc[swapped_dates.notna()]
            result.loc[mask_invalid & swapped_dates.isna()] = swapped.loc[swapped_dates.isna()]
        
        # 填充空值
        result = result.fillna('')
        return result
    
    def safe_upper(self, value):
        """安全的字符串大写转换"""
        if pd.isna(value) or value is None:
            return ''
        if isinstance(value, (int, float)):
            return str(value).upper()
        return str(value).upper()
    
    def get_physical_priority(self, cargo_id):
        """获取实货匹配优先级 - 增强错误处理"""
        if pd.isna(cargo_id) or cargo_id is None:
            return 100
            
        # 安全转换为字符串
        cargo_id_str = self.safe_upper(str(cargo_id))
        
        # 按照你的要求：phy-2026-04 -> phy-2026-05 -> phy-2026-01 -> phy-2026-02 -> phy-2026-03
        priority_map = {
            'PHY-2026-04': 1,
            'PHY-2026-05': 2,
            'PHY-2026-01': 3,
            'PHY-2026-02': 4,
            'PHY-2026-03': 5
        }
        
        # 匹配 cargo_id 中的关键部分
        for key in priority_map:
            if key in cargo_id_str:
                return priority_map[key]
        
        # 如果没有匹配到，返回默认值
        return 100
    
    def get_commodity_priority(self, commodity):
        """获取商品优先级：BRENT优先，JCC次之"""
        if pd.isna(commodity) or commodity is None:
            return 3
            
        commodity_str = self.safe_upper(str(commodity))
        if 'BRENT' in commodity_str:
            return 1
        elif 'JCC' in commodity_str:
            return 2
        else:
            return 3
    
    def calculate_net_positions(self, df_paper, designation_date):
        """FIFO净仓计算 - 过滤指定日期前的交易"""
        st.info("🔄 执行纸货内部对冲 (FIFO Netting)...")
        progress_bar = st.progress(0)
        
        # 过滤指定日期之前的交易（不参与匹配）
        df_paper_filtered = df_paper.copy()
        
        # 确保Trade Date是datetime类型
        if 'Trade Date' in df_paper_filtered.columns:
            df_paper_filtered['Trade Date'] = pd.to_datetime(df_paper_filtered['Trade Date'], errors='coerce')
        
        if designation_date and 'Trade Date' in df_paper_filtered.columns:
            designation_dt = pd.to_datetime(designation_date)
            before_mask = df_paper_filtered['Trade Date'] < designation_dt
            if before_mask.any():
                st.warning(f"过滤掉 {before_mask.sum()} 条指定日期({designation_date})之前的纸货交易")
                df_paper_filtered = df_paper_filtered[~before_mask].copy()
        
        if df_paper_filtered.empty:
            st.error("指定日期之后没有可用的纸货交易数据")
            return pd.DataFrame()
        
        df_paper_filtered = df_paper_filtered.sort_values(by='Trade Date').reset_index(drop=True)
        df_paper_filtered['Group_Key'] = df_paper_filtered['Std_Commodity'] + "_" + df_paper_filtered['Month']
        
        # 创建副本，避免在迭代中修改原始记录的问题
        records = []
        for idx, row in df_paper_filtered.iterrows():
            record = row.to_dict()
            # 确保Close_Events是列表
            record['Close_Events'] = []
            records.append(record)
        
        groups = {}
        
        # 分组
        for i, row in enumerate(records):
            key = row['Group_Key']
            if key not in groups:
                groups[key] = []
            groups[key].append(i)
            if i % 100 == 0:
                progress_bar.progress(min(i / len(records) * 0.5, 0.5))
        
        # FIFO净额化
        group_count = 0
        total_groups = len(groups)
        for key, indices in groups.items():
            open_queue = deque()
            for idx in indices:
                row = records[idx]
                current_vol = row.get('Volume', 0)
                records[idx]['Net_Open_Vol'] = current_vol
                records[idx]['Closed_Vol'] = 0
                
                if abs(current_vol) < 0.0001:
                    continue
                
                current_sign = 1 if current_vol > 0 else -1
                
                # 尝试与队列中的交易抵消
                while open_queue:
                    q_idx, q_vol, q_sign = open_queue[0]
                    if q_sign != current_sign:  # 方向相反才能抵消
                        offset = min(abs(current_vol), abs(q_vol))
                        current_vol -= (current_sign * offset)
                        q_vol -= (q_sign * offset)
                        
                        # 记录平仓事件
                        close_event = {
                            'Ref': str(records[idx].get('Recap No', '')),
                            'Date': records[idx].get('Trade Date'),
                            'Vol': offset,
                            'Price': records[idx].get('Price', 0),
                            'Commodity': records[idx].get('Std_Commodity'),
                            'Month': records[idx].get('Month')
                        }
                        records[q_idx]['Close_Events'].append(close_event)
                        records[q_idx]['Closed_Vol'] += offset
                        records[q_idx]['Net_Open_Vol'] = q_vol
                        records[idx]['Closed_Vol'] += offset
                        records[idx]['Net_Open_Vol'] = current_vol
                        
                        if abs(q_vol) < 0.0001:
                            open_queue.popleft()
                        else:
                            open_queue[0] = (q_idx, q_vol, q_sign)
                        
                        if abs(current_vol) < 0.0001:
                            break
                    else:
                        break
                
                # 剩余部分入队
                if abs(current_vol) > 0.0001:
                    open_queue.append((idx, current_vol, current_sign))
            
            group_count += 1
            progress_bar.progress(0.5 + (group_count / total_groups) * 0.5)
        
        progress_bar.progress(1.0)
        st.success(f"✅ 纸货内部对冲完成！共处理 {len(groups)} 个商品-月份组合")
        return pd.DataFrame(records)
    
    def prepare_dataframe_for_operations(self, df):
        """准备数据框用于操作，确保没有不可哈希的类型"""
        df_copy = df.copy()
        
        # 将Close_Events列表转换为字符串表示，以便去重操作
        if 'Close_Events' in df_copy.columns:
            df_copy['Close_Events_Str'] = df_copy['Close_Events'].apply(
                lambda x: json.dumps(x) if isinstance(x, list) else str(x)
            )
            # 创建一个不带Close_Events列的数据框用于去重
            df_for_ops = df_copy.drop(columns=['Close_Events'])
        else:
            df_for_ops = df_copy
        
        return df_for_ops
    
    def match_hedges(self, df_physical, df_paper_net, designation_date):
        """实货匹配 - 根据新需求更新"""
        st.info("🔄 开始实货匹配...")
        progress_bar = st.progress(0)
        
        hedge_relations = []
        open_positions = []  # 记录开仓头寸
        close_positions = []  # 记录平仓头寸
        
        # 准备纸货数据用于操作
        active_paper = df_paper_net.copy()
        active_paper['Allocated_To_Phy'] = 0.0
        active_paper['_original_index'] = active_paper.index
        
        df_phy = df_physical.copy()
        df_phy['_orig_idx'] = df_phy.index
        
        # 数据清洗和验证
        st.info("🔄 数据清洗和验证中...")
        
        # 检查必要字段
        required_fields = ['Cargo_ID', 'Unhedged_Volume']
        for field in required_fields:
            if field not in df_phy.columns:
                st.error(f"实货数据缺少必要字段: {field}")
                return pd.DataFrame(), df_physical
        
        # 确保数据格式正确
        df_phy['Cargo_ID'] = df_phy['Cargo_ID'].apply(lambda x: '' if pd.isna(x) else str(x))
        df_phy['Unhedged_Volume'] = pd.to_numeric(df_phy['Unhedged_Volume'], errors='coerce').fillna(0)
        
        if 'Hedge_Proxy' in df_phy.columns:
            df_phy['Hedge_Proxy'] = df_phy['Hedge_Proxy'].apply(lambda x: '' if pd.isna(x) else str(x))
        
        # 按优先级排序实货
        # 1. 商品优先级：BRENT优先
        # 2. Cargo_ID优先级：phy-2026-04 -> 05 -> 01 -> 02 -> 03
        # 3. 按原索引作为最后排序依据
        
        if 'Hedge_Proxy' in df_phy.columns:
            df_phy['_commodity_priority'] = df_phy['Hedge_Proxy'].apply(self.get_commodity_priority)
        else:
            df_phy['_commodity_priority'] = 3
        
        if 'Cargo_ID' in df_phy.columns:
            df_phy['_cargo_priority'] = df_phy['Cargo_ID'].apply(self.get_physical_priority)
        else:
            df_phy['_cargo_priority'] = 100
        
        # 按优先级排序
        df_phy = df_phy.sort_values(
            by=['_commodity_priority', '_cargo_priority', '_orig_idx']
        ).reset_index(drop=True)
        
        # 移除临时列
        df_phy = df_phy.drop(columns=['_commodity_priority', '_cargo_priority'])
        
        total_cargos = len(df_phy)
        
        for idx, (_, cargo) in enumerate(df_phy.iterrows()):
            cargo_id = str(cargo.get('Cargo_ID', ''))
            phy_vol = float(cargo.get('Unhedged_Volume', 0))
            
            if abs(phy_vol) < 0.0001:
                continue
            
            proxy = str(cargo.get('Hedge_Proxy', ''))
            target_month = cargo.get('Target_Contract_Month', '')
            phy_dir = cargo.get('Direction', 'Buy')
            desig_date = cargo.get('Designation_Date', pd.NaT)
            
            # 筛选候选交易 - 优先匹配相同品种和月份
            candidates_df = active_paper.copy()
            
            # 安全处理字符串匹配
            if proxy:
                mask = candidates_df['Std_Commodity'].apply(
                    lambda x: proxy.upper() in self.safe_upper(x)
                )
                candidates_df = candidates_df[mask].copy()
            
            if not candidates_df.empty and target_month:
                month_mask = candidates_df['Month'].apply(
                    lambda x: str(target_month).upper() == self.safe_upper(str(x))
                )
                candidates_df = candidates_df[month_mask].copy()
            
            # 如果同月份不够，尝试匹配其他月份的相同品种
            if candidates_df.empty or candidates_df['Net_Open_Vol'].abs().sum() < abs(phy_vol):
                # 查找相同品种的所有交易
                if proxy:
                    all_same_commodity = active_paper[
                        active_paper['Std_Commodity'].apply(
                            lambda x: proxy.upper() in self.safe_upper(x)
                        )
                    ].copy()
                    
                    if len(all_same_commodity) > 0:
                        # 按时间排序（FIFO）
                        all_same_commodity = all_same_commodity.sort_values('Trade Date')
                        # 使用基于索引的合并来避免list类型问题
                        if not candidates_df.empty:
                            # 获取候选数据框的索引
                            candidates_indices = set(candidates_df.index)
                            all_indices = set(all_same_commodity.index)
                            # 合并索引
                            combined_indices = candidates_indices.union(all_indices)
                            # 获取合并后的数据
                            candidates_df = active_paper.loc[list(combined_indices)].copy()
                        else:
                            candidates_df = all_same_commodity
            
            if candidates_df.empty:
                continue
            
            # 时间排序：有指定日期按时间差，否则FIFO
            if pd.notna(desig_date) and not candidates_df['Trade Date'].isnull().all():
                candidates_df['Time_Lag_Days'] = (candidates_df['Trade Date'] - desig_date).dt.days
                candidates_df['Abs_Lag'] = candidates_df['Time_Lag_Days'].abs()
                candidates_df = candidates_df.sort_values(by=['Abs_Lag', 'Trade Date'])
            else:
                candidates_df['Time_Lag_Days'] = np.nan
                candidates_df = candidates_df.sort_values(by='Trade Date')
            
            # 分配匹配
            for _, ticket in candidates_df.iterrows():
                if abs(phy_vol) < 1:
                    break
                
                original_index = ticket['_original_index']
                curr_allocated = active_paper.at[original_index, 'Allocated_To_Phy']
                curr_total_vol = float(ticket.get('Volume', 0))
                curr_net_open = float(ticket.get('Net_Open_Vol', 0))
                avail = curr_net_open - curr_allocated
                
                if abs(avail) < 0.0001:
                    continue
                
                # 确定分配量（确保符号正确）
                alloc_amt_abs = abs(phy_vol) if abs(avail) >= abs(phy_vol) else abs(avail)
                # 分配量的符号与可用量的符号一致
                alloc_amt = np.sign(avail) * alloc_amt_abs
                phy_vol -= alloc_amt_abs if phy_vol > 0 else -alloc_amt_abs
                active_paper.at[original_index, 'Allocated_To_Phy'] += alloc_amt
                
                # 记录开仓和平仓
                ticket_commodity = str(ticket.get('Std_Commodity', ''))
                ticket_month = str(ticket.get('Month', ''))
                open_price = float(ticket.get('Price', 0))
                
                if alloc_amt > 0:  # 开仓
                    open_positions.append({
                        'Cargo_ID': cargo_id,
                        'Commodity': ticket_commodity,
                        'Month': ticket_month,
                        'Open_Date': ticket.get('Trade Date'),
                        'Volume': alloc_amt,
                        'Price': open_price,
                        'Ticket_ID': str(ticket.get('Recap No', ''))
                    })
                elif alloc_amt < 0:  # 平仓
                    close_positions.append({
                        'Cargo_ID': cargo_id,
                        'Commodity': ticket_commodity,
                        'Month': ticket_month,
                        'Close_Date': ticket.get('Trade Date'),
                        'Volume': alloc_amt,  # 负数
                        'Price': open_price,
                        'Ticket_ID': str(ticket.get('Recap No', ''))
                    })
                
                # 计算财务指标
                mtm_price = float(ticket.get('Mtm Price', open_price))
                total_pl_raw = float(ticket.get('Total P/L', 0))
                close_events = ticket.get('Close_Events', [])
                
                # 格式化平仓路径
                close_path_str = ""
                if close_events:
                    sorted_events = sorted(close_events, key=lambda x: x['Date'] if pd.notna(x['Date']) else pd.Timestamp.min)
                    details = []
                    for e in sorted_events:
                        d_str = e['Date'].strftime('%Y-%m-%d') if pd.notna(e['Date']) else 'N/A'
                        p_str = f"@{e['Price']}" if pd.notna(e['Price']) else ""
                        details.append(f"[{d_str} Tkt#{e['Ref']} Vol:{e['Vol']:.0f} {p_str}]")
                    close_path_str = " -> ".join(details)
                
                # 计算分配比例
                ratio = abs(alloc_amt) / abs(curr_total_vol) if abs(curr_total_vol) > 0 else 0
                unrealized_mtm = (mtm_price - open_price) * alloc_amt
                allocated_total_pl = total_pl_raw * ratio
                
                hedge_relations.append({
                    'Cargo_ID': cargo_id,
                    'Proxy': proxy,
                    'Designation_Date': desig_date,
                    'Open_Date': ticket.get('Trade Date'),
                    'Time_Lag': ticket.get('Time_Lag_Days'),
                    'Ticket_ID': str(ticket.get('Recap No', '')),
                    'Month': ticket_month,
                    'Commodity': ticket_commodity,
                    'Allocated_Vol': alloc_amt,  # 正数为开仓，负数为平仓
                    'Trade_Volume': curr_total_vol,
                    'Trade_Net_Open': curr_net_open,
                    'Trade_Closed_Vol': float(ticket.get('Closed_Vol', 0)),
                    'Open_Price': open_price,
                    'MTM_Price': mtm_price,
                    'Alloc_Unrealized_MTM': round(unrealized_mtm, 2),
                    'Alloc_Total_PL': round(allocated_total_pl, 2),
                    'Close_Path_Details': close_path_str,
                    'Position_Type': '开仓' if alloc_amt > 0 else '平仓'
                })
                
                # 更新实货未对冲量
                orig_idx = cargo.get('_orig_idx')
                if orig_idx in df_physical.index:
                    df_physical.at[orig_idx, 'Unhedged_Volume'] = phy_vol
            
            progress_bar.progress((idx + 1) / total_cargos)
        
        # 更新分配量
        cols_to_update = active_paper[['_original_index', 'Allocated_To_Phy']].set_index('_original_index')
        df_paper_net.update(cols_to_update)
        
        # 计算开仓和平仓汇总
        if open_positions:
            self.open_positions_summary = self.calculate_weighted_average(open_positions, '开仓')
        else:
            self.open_positions_summary = pd.DataFrame()
            
        if close_positions:
            self.close_positions_summary = self.calculate_weighted_average(close_positions, '平仓')
        else:
            self.close_positions_summary = pd.DataFrame()
        
        progress_bar.progress(1.0)
        df_relations = pd.DataFrame(hedge_relations) if hedge_relations else pd.DataFrame()
        st.success(f"✅ 实货匹配完成！共生成 {len(df_relations)} 条匹配记录")
        
        return df_relations, df_physical
    
    def calculate_weighted_average(self, positions, position_type):
        """计算加权平均价格"""
        if not positions:
            return pd.DataFrame()
        
        df = pd.DataFrame(positions)
        
        # 移除符号
        if position_type == '平仓':
            df['Volume_Abs'] = abs(df['Volume'])
        else:
            df['Volume_Abs'] = df['Volume']
        
        # 确保数值类型
        df['Volume_Abs'] = pd.to_numeric(df['Volume_Abs'], errors='coerce').fillna(0)
        df['Price'] = pd.to_numeric(df['Price'], errors='coerce').fillna(0)
        
        # 按商品和月份分组计算加权平均
        summary = df.groupby(['Commodity', 'Month']).apply(
            lambda x: pd.Series({
                '总数量': x['Volume_Abs'].sum(),
                '加权平均价格': np.average(x['Price'], weights=x['Volume_Abs']) if x['Volume_Abs'].sum() > 0 else 0,
                '交易次数': len(x),
                '最早交易日期': x.iloc[0]['Open_Date'] if position_type == '开仓' else x.iloc[0]['Close_Date'],
                '最晚交易日期': x.iloc[-1]['Open_Date'] if position_type == '开仓' else x.iloc[-1]['Close_Date']
            })
        ).reset_index()
        
        summary['头寸类型'] = position_type
        return summary
    
    def run_matching(self, df_paper_raw, df_physical_raw, designation_date="2025-11-12"):
        """执行完整匹配流程"""
        try:
            # 数据预处理
            st.info("🔄 数据预处理中...")
            
            # 纸货预处理
            df_paper = df_paper_raw.copy()
            
            # 确保必要的列存在
            required_cols_paper = ['Trade Date', 'Volume', 'Commodity']
            for col in required_cols_paper:
                if col not in df_paper.columns:
                    st.error(f"纸货数据缺少必要列: {col}")
                    return pd.DataFrame(), df_physical_raw, pd.DataFrame(), df_paper, pd.DataFrame(), pd.DataFrame()
            
            # 标准化处理
            df_paper['Trade Date'] = pd.to_datetime(df_paper['Trade Date'], errors='coerce')
            df_paper['Volume'] = pd.to_numeric(df_paper['Volume'], errors='coerce').fillna(0)
            df_paper['Std_Commodity'] = self.clean_str(df_paper['Commodity'])
            
            if 'Month' in df_paper.columns:
                df_paper['Month'] = self.standardize_month(df_paper['Month'])
            else:
                # 如果没有Month列，尝试从其他列推断或创建默认值
                df_paper['Month'] = df_paper['Trade Date'].dt.strftime('%b %y').str.upper()
            
            # 处理缺失字段
            if 'Recap No' not in df_paper.columns:
                df_paper['Recap No'] = [f"TKT-{i+1:04d}" for i in range(len(df_paper))]
            
            for col in ['Price', 'Mtm Price', 'Total P/L']:
                if col not in df_paper.columns:
                    df_paper[col] = 0.0
            
            # 实货预处理
            df_physical = df_physical_raw.copy()
            
            # 标准化列名
            col_mapping = {
                'Target_Pricing_Month': 'Target_Contract_Month',
                'Month': 'Target_Contract_Month',
                'Hedge_Proxy': 'Hedge_Proxy',
                'Direction': 'Direction'
            }
            
            for old_col, new_col in col_mapping.items():
                if old_col in df_physical.columns and new_col not in df_physical.columns:
                    df_physical[new_col] = df_physical[old_col]
            
            # 确保必要列
            if 'Volume' in df_physical.columns:
                df_physical['Volume'] = pd.to_numeric(df_physical['Volume'], errors='coerce').fillna(0)
                df_physical['Unhedged_Volume'] = df_physical['Volume']
            
            if 'Hedge_Proxy' in df_physical.columns:
                df_physical['Hedge_Proxy'] = self.clean_str(df_physical['Hedge_Proxy'])
            
            if 'Target_Contract_Month' in df_physical.columns:
                df_physical['Target_Contract_Month'] = self.standardize_month(df_physical['Target_Contract_Month'])
            
            # 指定日期
            date_cols = ['Designation_Date', 'Pricing_Start', 'Trade Date']
            for col in date_cols:
                if col in df_physical.columns:
                    df_physical['Designation_Date'] = pd.to_datetime(df_physical[col], errors='coerce')
                    break
            else:
                df_physical['Designation_Date'] = pd.NaT
            
            # 数据验证
            st.info("🔄 数据验证中...")
            
            # 检查实货Cargo_ID
            if 'Cargo_ID' in df_physical.columns:
                df_physical['Cargo_ID'] = df_physical['Cargo_ID'].apply(lambda x: '' if pd.isna(x) else str(x))
                unique_cargos = len(df_physical['Cargo_ID'].unique())
                st.info(f"实货Cargo_ID数量: {unique_cargos}")
            
            # 检查纸货数据
            paper_count = len(df_paper)
            st.info(f"纸货交易总数: {paper_count}")
            st.info(f"指定匹配开始日期: {designation_date}")
            
            # 检查指定日期之后的交易
            if 'Trade Date' in df_paper.columns:
                designation_dt = pd.to_datetime(designation_date)
                valid_paper = df_paper[df_paper['Trade Date'] >= designation_dt]
                st.info(f"指定日期之后的纸货交易数: {len(valid_paper)}")
            
            # 执行匹配
            self.df_paper_net = self.calculate_net_positions(df_paper, designation_date)
            
            if self.df_paper_net.empty:
                st.warning("纸货净仓数据为空，请检查数据或调整指定日期")
                return pd.DataFrame(), df_physical, pd.DataFrame(), df_paper, pd.DataFrame(), pd.DataFrame()
            
            self.df_relations, self.df_physical_updated = self.match_hedges(
                df_physical, self.df_paper_net, designation_date
            )
            
            return (self.df_relations, self.df_physical_updated, 
                    self.df_paper_net, df_paper, 
                    self.open_positions_summary, self.close_positions_summary)
            
        except Exception as e:
            st.error(f"匹配过程出现异常: {str(e)}")
            st.exception(e)
            return pd.DataFrame(), df_physical_raw, pd.DataFrame(), df_paper_raw, pd.DataFrame(), pd.DataFrame()

# ---------------------------------------------------------
# 2. 分析模块 (基于真实匹配结果)
# ---------------------------------------------------------

class HedgeAnalysis:
    """套保分析模块"""
    
    def __init__(self, df_relations, df_physical, df_paper_net, 
                 open_summary=None, close_summary=None):
        self.df_relations = df_relations if df_relations is not None else pd.DataFrame()
        self.df_physical = df_physical if df_physical is not None else pd.DataFrame()
        self.df_paper_net = df_paper_net if df_paper_net is not None else pd.DataFrame()
        self.open_summary = open_summary if open_summary is not None else pd.DataFrame()
        self.close_summary = close_summary if close_summary is not None else pd.DataFrame()
        self.summary_stats = {}
        self.calculate_summary()
    
    def calculate_summary(self):
        """计算汇总统计"""
        if self.df_relations.empty:
            self.summary_stats = {
                'total_matched': 0,
                'total_physical': 0,
                'match_rate': 0,
                'open_volume': 0,
                'close_volume': 0,
                'total_pl': 0,
                'total_unrealized': 0,
                'matched_cargos': 0,
                'total_cargos': 0,
                'total_tickets': 0,
                'open_count': 0,
                'close_count': 0,
                'avg_time_lag': 0,
                'std_time_lag': 0
            }
            return
        
        try:
            # 匹配统计
            if 'Allocated_Vol' in self.df_relations.columns:
                total_matched = abs(self.df_relations['Allocated_Vol']).sum()
            else:
                total_matched = 0
            
            if 'Volume' in self.df_physical.columns:
                total_physical = abs(self.df_physical['Volume']).sum()
            else:
                total_physical = 0
                
            match_rate = (total_matched / total_physical * 100) if total_physical > 0 else 0
            
            # 开仓平仓统计
            if 'Allocated_Vol' in self.df_relations.columns:
                open_positions = self.df_relations[self.df_relations['Allocated_Vol'] > 0]
                close_positions = self.df_relations[self.df_relations['Allocated_Vol'] < 0]
                
                open_volume = open_positions['Allocated_Vol'].sum() if not open_positions.empty else 0
                close_volume = abs(close_positions['Allocated_Vol'].sum()) if not close_positions.empty else 0
            else:
                open_volume = close_volume = 0
            
            # 财务统计
            total_pl = self.df_relations['Alloc_Total_PL'].sum() if 'Alloc_Total_PL' in self.df_relations.columns else 0
            total_unrealized = self.df_relations['Alloc_Unrealized_MTM'].sum() if 'Alloc_Unrealized_MTM' in self.df_relations.columns else 0
            
            # 数量统计
            matched_cargos = self.df_relations['Cargo_ID'].nunique() if 'Cargo_ID' in self.df_relations.columns else 0
            total_cargos = self.df_physical['Cargo_ID'].nunique() if 'Cargo_ID' in self.df_physical.columns else 0
            total_tickets = len(self.df_relations)
            
            # 时间统计
            if 'Time_Lag' in self.df_relations.columns:
                avg_time_lag = self.df_relations['Time_Lag'].abs().mean()
                std_time_lag = self.df_relations['Time_Lag'].abs().std()
            else:
                avg_time_lag = std_time_lag = 0
            
            self.summary_stats = {
                'total_matched': total_matched,
                'total_physical': total_physical,
                'match_rate': match_rate,
                'open_volume': open_volume,
                'close_volume': close_volume,
                'total_pl': total_pl,
                'total_unrealized': total_unrealized,
                'matched_cargos': matched_cargos,
                'total_cargos': total_cargos,
                'total_tickets': total_tickets,
                'avg_time_lag': avg_time_lag,
                'std_time_lag': std_time_lag
            }
            
        except Exception as e:
            st.warning(f"计算汇总统计时出现错误: {str(e)}")
            self.summary_stats = {
                'total_matched': 0,
                'total_physical': 0,
                'match_rate': 0,
                'open_volume': 0,
                'close_volume': 0,
                'total_pl': 0,
                'total_unrealized': 0,
                'matched_cargos': 0,
                'total_cargos': 0,
                'total_tickets': 0,
                'avg_time_lag': 0,
                'std_time_lag': 0
            }
    
    def create_summary_metrics(self):
        """创建概览指标卡片"""
        stats = self.summary_stats
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📊 匹配率", f"{stats['match_rate']:.1f}%", 
                     delta=f"{stats['total_matched']:,.0f}/{stats['total_physical']:,.0f}")
        
        with col2:
            coverage = (stats['matched_cargos'] / stats['total_cargos'] * 100) if stats['total_cargos'] > 0 else 0
            st.metric("📦 匹配覆盖率", f"{coverage:.1f}%",
                     delta=f"{stats['matched_cargos']}/{stats['total_cargos']}")
        
        with col3:
            st.metric("💰 总P/L", f"${stats['total_pl']:,.2f}",
                     delta=f"未实现: ${stats['total_unrealized']:,.2f}")
        
        with col4:
            st.metric("⚖️ 开仓/平仓", f"{stats['open_volume']:,.0f}/{stats['close_volume']:,.0f}",
                     delta=f"{stats['matched_cargos']}个实货")

# ---------------------------------------------------------
# 3. Streamlit 主应用
# ---------------------------------------------------------

def main():
    st.set_page_config(
        page_title="实纸货套保匹配分析系统 v2.0",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # 自定义CSS
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #374151;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
        border-bottom: 2px solid #E5E7EB;
        padding-bottom: 0.5rem;
    }
    .success-box {
        background-color: #D1FAE5;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #10B981;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #DBEAFE;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #3B82F6;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #FEF3C7;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #F59E0B;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #FEE2E2;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #EF4444;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # 标题
    st.markdown('<h1 class="main-header">📈 实纸货套保匹配分析系统 v2.0</h1>', unsafe_allow_html=True)
    st.markdown("### 专业套保匹配与有效性测试工具 | 支持加权均价计算")
    
    # 初始化session state
    if 'engine' not in st.session_state:
        st.session_state.engine = HedgeMatchingEngine()
    if 'analysis' not in st.session_state:
        st.session_state.analysis = None
    if 'matching_complete' not in st.session_state:
        st.session_state.matching_complete = False
    
    # 侧边栏
    with st.sidebar:
        st.markdown("### 📁 数据上传")
        
        paper_file = st.file_uploader(
            "纸货数据文件",
            type=["csv", "xlsx", "xls"],
            key="paper_uploader",
            help="支持CSV/Excel格式，需包含Trade Date, Volume, Commodity等字段"
        )
        
        physical_file = st.file_uploader(
            "实货数据文件",
            type=["csv", "xlsx", "xls"],
            key="physical_uploader",
            help="支持CSV/Excel格式，需包含Cargo_ID, Volume, Hedge_Proxy等字段"
        )
        
        st.markdown("---")
        st.markdown("### ⚙️ 匹配设置")
        
        # 指定日期设置 - 修正为2025年
        designation_date = st.date_input(
            "指定匹配开始日期",
            value=datetime(2025, 11, 12),  # 修正为2025年
            help="从该日期开始的纸货交易才会参与匹配"
        )
        
        st.markdown("---")
        st.markdown("### 📊 分析设置")
        
        show_charts = st.checkbox("显示分析图表", value=True)
        show_positions = st.checkbox("显示头寸汇总", value=True)
        show_risk = st.checkbox("显示风险指标", value=False)
        max_rows = st.slider("表格显示行数", 10, 200, 50)
        
        st.markdown("---")
        
        if st.button("🔄 重置所有数据", type="secondary"):
            st.session_state.engine = HedgeMatchingEngine()
            st.session_state.analysis = None
            st.session_state.matching_complete = False
            st.rerun()
    
    # 主内容区
    if paper_file is not None and physical_file is not None:
        # 读取数据
        try:
            # 读取纸货数据
            if paper_file.name.endswith(('.xlsx', '.xls')):
                df_paper_raw = pd.read_excel(paper_file)
            else:
                df_paper_raw = pd.read_csv(paper_file)
            
            # 读取实货数据
            if physical_file.name.endswith(('.xlsx', '.xls')):
                df_physical_raw = pd.read_excel(physical_file)
            else:
                df_physical_raw = pd.read_csv(physical_file)
            
            # 显示数据预览
            with st.expander("📋 原始数据预览", expanded=False):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"**纸货数据** ({len(df_paper_raw)}行, {len(df_paper_raw.columns)}列)")
                    st.dataframe(df_paper_raw.head(10), use_container_width=True)
                    st.caption(f"关键字段: {', '.join(df_paper_raw.columns.tolist()[:5])}...")
                
                with col2:
                    st.markdown(f"**实货数据** ({len(df_physical_raw)}行, {len(df_physical_raw.columns)}列)")
                    st.dataframe(df_physical_raw.head(10), use_container_width=True)
                    st.caption(f"关键字段: {', '.join(df_physical_raw.columns.tolist()[:5])}...")
            
            # 显示匹配规则说明
            st.markdown(f'''
            <div class="info-box">
            <h4>🎯 匹配规则说明</h4>
            <ul>
            <li><b>优先级1:</b> 优先匹配BRENT计价品种，JCC次之</li>
            <li><b>优先级2:</b> 按phy-2026-04 → 05 → 01 → 02 → 03顺序匹配</li>
            <li><b>时间限制:</b> 仅匹配指定日期（{designation_date}）之后的纸货交易</li>
            <li><b>数量:</b> 正数为开仓，负数为平仓</li>
            <li><b>加权均价:</b> 自动计算开仓/平仓加权平均价格</li>
            </ul>
            </div>
            ''', unsafe_allow_html=True)
            
            # 执行匹配按钮
            if st.button("🚀 执行套保匹配", type="primary", use_container_width=True):
                with st.spinner("正在执行套保匹配，请稍候..."):
                    try:
                        # 执行匹配
                        (df_relations, df_physical_updated, 
                         df_paper_net, df_paper_processed,
                         open_summary, close_summary) = st.session_state.engine.run_matching(
                            df_paper_raw, df_physical_raw, str(designation_date)
                        )
                        
                        if df_relations is not None and not df_relations.empty:
                            # 创建分析模块
                            st.session_state.analysis = HedgeAnalysis(
                                df_relations, df_physical_updated, df_paper_net,
                                open_summary, close_summary
                            )
                            st.session_state.matching_complete = True
                            
                            # 显示匹配成功信息
                            st.markdown(f'''
                            <div class="success-box">
                            <h4>✅ 套保匹配成功完成！</h4>
                            <p>匹配日期范围: {designation_date} 之后</p>
                            <p>匹配优先级: BRENT优先，实货按指定顺序匹配</p>
                            <p>匹配记录数: {len(df_relations)}条</p>
                            </div>
                            ''', unsafe_allow_html=True)
                            
                        else:
                            st.markdown('<div class="warning-box">⚠️ 匹配过程完成，但未生成匹配记录</div>', unsafe_allow_html=True)
                            st.info("可能的原因：")
                            st.write("1. 指定日期之后没有可用的纸货交易")
                            st.write("2. 实货数据中没有需要匹配的头寸")
                            st.write("3. 商品或月份不匹配")
                            
                            # 创建空的分析模块
                            st.session_state.analysis = HedgeAnalysis(
                                pd.DataFrame(), df_physical_raw, pd.DataFrame(),
                                pd.DataFrame(), pd.DataFrame()
                            )
                            st.session_state.matching_complete = True
                            
                    except Exception as e:
                        st.markdown(f'<div class="error-box">❌ 匹配过程中出现错误: {str(e)}</div>', unsafe_allow_html=True)
                        
                        # 显示详细错误信息
                        with st.expander("查看错误详情"):
                            st.exception(e)
        
        except Exception as e:
            st.error(f"数据读取错误: {str(e)}")
            st.info("请确保上传的文件格式正确，并包含必要的字段。")
    
    # 显示分析结果
    if st.session_state.matching_complete and st.session_state.analysis is not None:
        analysis = st.session_state.analysis
        
        if not analysis.df_relations.empty:
            st.markdown("---")
            st.markdown('<h2 class="sub-header">📊 匹配分析结果</h2>', unsafe_allow_html=True)
            
            # 1. 概览指标
            analysis.create_summary_metrics()
            
            # 2. 匹配明细表
            st.markdown('<h3 class="sub-header">📋 匹配明细表</h3>', unsafe_allow_html=True)
            
            # 添加筛选器
            col1, col2 = st.columns(2)
            with col1:
                position_filter = st.selectbox(
                    "头寸类型筛选",
                    ["全部", "开仓", "平仓"],
                    index=0
                )
            
            with col2:
                if 'Commodity' in analysis.df_relations.columns:
                    commodity_options = analysis.df_relations['Commodity'].unique()
                    commodity_filter = st.multiselect(
                        "商品筛选",
                        options=commodity_options,
                        default=commodity_options
                    )
                else:
                    commodity_filter = []
            
            # 应用筛选
            filtered_df = analysis.df_relations.copy()
            if position_filter != "全部":
                filtered_df = filtered_df[filtered_df['Position_Type'] == position_filter]
            
            if commodity_filter:
                filtered_df = filtered_df[filtered_df['Commodity'].isin(commodity_filter)]
            
            # 显示筛选后的数据
            st.dataframe(filtered_df.head(max_rows), use_container_width=True)
            st.caption(f"显示 {len(filtered_df.head(max_rows))} 条记录，共 {len(filtered_df)} 条匹配记录 (筛选后)")
            
            # 3. 头寸汇总表（开仓/平仓加权均价）
            if show_positions:
                st.markdown('<h3 class="sub-header">⚖️ 头寸汇总与加权平均价格</h3>', unsafe_allow_html=True)
                
                if not analysis.open_summary.empty or not analysis.close_summary.empty:
                    tabs = st.tabs(["开仓汇总", "平仓汇总"])
                    
                    with tabs[0]:
                        if not analysis.open_summary.empty:
                            st.dataframe(analysis.open_summary, use_container_width=True)
                            st.caption(f"开仓头寸汇总 ({len(analysis.open_summary)}个商品-月份组合)")
                            
                            # 显示开仓加权平均价格
                            st.subheader("开仓加权平均价格汇总")
                            for _, row in analysis.open_summary.iterrows():
                                st.write(f"**{row['Commodity']} - {row['Month']}**: "
                                        f"{row['总数量']:,.0f}桶 @ ${row['加权平均价格']:.2f}")
                        else:
                            st.info("无开仓头寸数据")
                    
                    with tabs[1]:
                        if not analysis.close_summary.empty:
                            st.dataframe(analysis.close_summary, use_container_width=True)
                            st.caption(f"平仓头寸汇总 ({len(analysis.close_summary)}个商品-月份组合)")
                            
                            # 显示平仓加权平均价格
                            st.subheader("平仓加权平均价格汇总")
                            for _, row in analysis.close_summary.iterrows():
                                st.write(f"**{row['Commodity']} - {row['Month']}**: "
                                        f"{row['总数量']:,.0f}桶 @ ${row['加权平均价格']:.2f}")
                        else:
                            st.info("无平仓头寸数据")
                else:
                    st.info("暂无头寸汇总数据")
            
            # 4. 数据导出
            st.markdown("---")
            st.markdown('<h3 class="sub-header">💾 数据导出</h3>', unsafe_allow_html=True)
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                # 导出匹配结果
                csv_data = analysis.df_relations.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 匹配结果",
                    data=csv_data,
                    file_name=f"hedge_matching_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            with col2:
                # 导出开仓汇总
                if not analysis.open_summary.empty:
                    open_csv = analysis.open_summary.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="⚖️ 开仓汇总",
                        data=open_csv,
                        file_name=f"open_positions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
            
            with col3:
                # 导出平仓汇总
                if not analysis.close_summary.empty:
                    close_csv = analysis.close_summary.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="⚖️ 平仓汇总",
                        data=close_csv,
                        file_name=f"close_positions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
            
            with col4:
                # 导出所有数据
                @st.cache_data
                def convert_to_excel(df_dict):
                    output = io.BytesIO()
                    with pd.ExcelWriter(output, engine='openpyxl') as writer:
                        for sheet_name, df in df_dict.items():
                            if df is not None and not df.empty:
                                df.to_excel(writer, sheet_name=sheet_name, index=False)
                    return output.getvalue()
                
                excel_data = convert_to_excel({
                    "匹配结果": analysis.df_relations,
                    "开仓汇总": analysis.open_summary,
                    "平仓汇总": analysis.close_summary,
                    "实货数据": analysis.df_physical,
                    "纸货净仓": analysis.df_paper_net
                })
                
                st.download_button(
                    label="📊 完整数据",
                    data=excel_data,
                    file_name=f"hedge_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )
        else:
            st.markdown('<div class="info-box">📊 匹配完成，但未找到匹配记录</div>', unsafe_allow_html=True)
            st.info("建议检查：")
            st.write("1. 确认指定日期是否正确")
            st.write("2. 检查纸货和实货的商品名称是否一致")
            st.write("3. 确认实货数据中有需要匹配的头寸")
    
    else:
        # 欢迎页面
        if not (paper_file and physical_file):
            st.markdown("---")
            st.markdown('<div class="info-box">👈 请在左侧上传纸货和实货数据文件开始分析</div>', unsafe_allow_html=True)
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown("""
                ### 🎯 系统工作流程 (v2.0)
                
                1. **数据上传**
                   - 纸货交易数据 (包含交易日期、交易量、商品、价格等)
                   - 实货持仓数据 (包含Cargo_ID、交易量、套保代理、目标月份等)
                
                2. **智能匹配 (按新规则)**
                   - **时间过滤**: 仅匹配指定日期(2025年11月12日)之后的纸货交易
                   - **优先级1**: BRENT计价品种优先，JCC次之
                   - **优先级2**: 实货按 phy-2026-04 → 05 → 01 → 02 → 03 顺序匹配
                   - **头寸区分**: 正数为开仓，负数为平仓
                
                3. **加权均价计算**
                   - **开仓均价**: 按商品和月份计算加权平均开仓价格
                   - **平仓均价**: 按商品和月份计算加权平均平仓价格
                   - **有效性测试**: 为套保有效性测试提供基础数据
                
                4. **数据导出**
                   - 匹配结果CSV
                   - 开仓/平仓汇总CSV
                   - 完整数据Excel
                """)
            
            with col2:
                st.markdown("""
                ### 📋 数据要求
                
                **纸货数据必需字段:**
                - `Trade Date`: 交易日期
                - `Volume`: 交易量 (正买负卖)
                - `Commodity`: 商品品种
                - `Month`: 合约月份 (可选)
                - `Price`: 交易价格 (推荐)
                
                **实货数据必需字段:**
                - `Cargo_ID`: 实货编号 (建议包含年份月份)
                - `Volume`: 交易量
                - `Hedge_Proxy`: 套保代理 (如BRENT, JCC)
                - `Target_Contract_Month`: 目标月份
                
                **匹配规则:**
                - 仅匹配指定日期之后的交易
                - BRENT优先于JCC
                - 特定Cargo_ID优先顺序
                - 自动计算加权均价
                """)

if __name__ == "__main__":
    main()
