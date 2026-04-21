# 导入包
import pandas as pd
import numpy as np
import statsmodels.api as sm
from functools import reduce
from scipy import stats
import os

# 定义 生成估计窗口和事件窗口虚拟变量 的函数
def mark_event_window(group, est_window, event_window):
    # 重置索引以确保索引从 0 开始且连续
    group = group.reset_index(drop=True)
    
    # Find the index where 'date1' = 0 (事件日当天)
    # 关键：如果存在 date1 = 0 的行，使用该行的索引；否则使用第一个 date1 > 0 的索引
    date1_eq_0 = group[group['date1'] == pd.Timedelta(days=0)]
    
    if len(date1_eq_0) > 0:
        # 使用 date1 = 0 的第一个索引作为 transition_index
        transition_index = date1_eq_0.index.min()
    else:
        # 如果没有 date1 = 0，使用 date1 > 0 的第一个索引
        date1_gt_0 = group[group['date1'] > pd.Timedelta(days=0)]
        if len(date1_gt_0) > 0:
            transition_index = date1_gt_0.index.min()
        else:
            # 如果没有 date1 >= 0 的数据，使用最后一个负索引
            date1_lt_0 = group[group['date1'] < pd.Timedelta(days=0)]
            if len(date1_lt_0) > 0:
                transition_index = date1_lt_0.index.max()
            else:
                transition_index = 0
    
    # Mark 5 days before and 10 days after the transition as the event window
    event_window_indices = range(max(transition_index + event_window[0], group.index.min()), 
                                 min(transition_index + event_window[1] + 1, group.index.max() + 1))
    est_window_indices = range(max(transition_index + est_window[0], group.index.min()), 
                               min(transition_index + est_window[1], group.index.max() + 1))
    group['event_window'] = group.index.isin(event_window_indices).astype(int)
    group['est_window'] = group.index.isin(est_window_indices).astype(int)
    return group

def event_study(df_eventstudy = None, event_window_list = None, est_window = (-210,-10), predict_model = 'market', save_path = None, suspension_file = None, min_est_days = 200, print_event_details = False, event_detail_days = 10, check_stockid = None):
    # 检查参数有效性
    # 检查 df_eventstudy
    if df_eventstudy is None:
        raise ValueError("df_eventstudy cannot be None")

    # 去除缺失值
    df_eventstudy = df_eventstudy.dropna()

    # 尝试将 'date' 和 'eventdate' 列转换为日期格式
    try:
        df_eventstudy['date'] = pd.to_datetime(df_eventstudy['date'])
        df_eventstudy['eventdate'] = pd.to_datetime(df_eventstudy['eventdate'])
    except ValueError as e:
        raise ValueError("Conversion to datetime failed. Please ensure 'date' and 'eventdate' columns are in a proper date format.") from e

    # 基本处理 - stockid格式化需要放在停牌检查之前
    df_eventstudy['stockid'] = df_eventstudy['stockid'].astype(str).str.zfill(6)

    # ============================================================
    # 收集所有异常情况
    # ============================================================
    exclusion_records = []  # 用于记录所有被排除的情况

    # ============================================================
    # 加载停牌日数据并过滤停牌公司
    # ============================================================
    if suspension_file is not None:
        try:
            df_suspension = pd.read_csv(suspension_file, encoding='utf-8-sig', low_memory=False)
            # 只对关键列进行缺失值处理，保留其他列
            df_suspension = df_suspension.dropna(subset=['Stkcd', 'Suspdate_start', 'Suspdate_end'])
            df_suspension['Stkcd'] = df_suspension['Stkcd'].astype(str).str.zfill(6)
            df_suspension['Suspdate_start'] = pd.to_datetime(df_suspension['Suspdate_start'], errors='coerce')
            df_suspension['Suspdate_end'] = pd.to_datetime(df_suspension['Suspdate_end'], errors='coerce')
            # 再次删除转换后变为NaT的行
            df_suspension = df_suspension.dropna(subset=['Suspdate_start', 'Suspdate_end'])
        except Exception as e:
            raise ValueError(f"Failed to load suspension file: {e}")

        # 检查哪些公司的事件日处于停牌期间 - 使用向量化方法提高效率
        original_count = df_eventstudy.shape[0]

        # 合并事件数据和停牌数据
        merged = df_eventstudy.merge(
            df_suspension[['Stkcd', 'Suspdate_start', 'Suspdate_end']],
            left_on='stockid',
            right_on='Stkcd',
            how='left'
        )

        # 检查事件日是否在停牌期间
        merged['in_suspension'] = (
            (merged['Suspdate_start'] <= merged['eventdate']) &
            (merged['Suspdate_end'] >= merged['eventdate'])
        )

        # 筛选出在停牌期的记录
        suspension_records_df = merged[merged['in_suspension']]

        if len(suspension_records_df) > 0:
            # 去重，获取唯一的(stockid, eventdate)组合
            suspension_records_df = suspension_records_df.drop_duplicates(subset=['stockid', 'eventdate'])
            suspension_records = suspension_records_df[['stockid', 'eventdate', 'Suspdate_start', 'Suspdate_end']].copy()
            suspension_records.columns = ['stockid', 'eventdate', 'susp_start', 'susp_end']
            suspension_records['reason'] = '事件日停牌'
            suspension_records['detail'] = suspension_records.apply(
                lambda x: f"停牌期: {x['susp_start'].strftime('%Y-%m-%d')} 至 {x['susp_end'].strftime('%Y-%m-%d')}", axis=1
            )

            # 添加到异常记录
            exclusion_records.extend(suspension_records[['stockid', 'eventdate', 'reason', 'detail']].to_dict('records'))

            # 从原始数据中移除停牌公司
            susp_stockid_dates = set(zip(suspension_records['stockid'], suspension_records['eventdate']))
            df_eventstudy = df_eventstudy[~df_eventstudy.apply(lambda x: (x['stockid'], x['eventdate']) in susp_stockid_dates, axis=1)]
            df_eventstudy = df_eventstudy.reset_index(drop=True)
            print(f"停牌过滤: 移除了 {len(suspension_records)} 条事件记录 (原 {original_count} 条 -> 现 {len(df_eventstudy)} 条)")
        else:
            print("停牌过滤: 没有需要移除的记录")
    else:
        print("停牌过滤: 未提供停牌数据文件，跳过停牌检查")

    if predict_model == 'fama3':
        required_columns = ['smb', 'hml', 'rf']
        if not all(column in df_eventstudy.columns for column in required_columns):
            raise ValueError("For 'fama3' model, df_eventstudy must contain columns: " + ', '.join(required_columns))

    # 检查 event_window_list
    if not isinstance(event_window_list, (list, tuple)) or not all(isinstance(window, (list, tuple)) for window in event_window_list):
        raise ValueError("event_window_list must be a list of lists or tuples")

    # 检查 est_window
    if not isinstance(est_window, (list, tuple)) or len(est_window) != 2:
        raise ValueError("est_window must be a tuple or list of two elements")
    
    # 检查 predict_model
    if predict_model not in ['market', 'market_adj', 'fama3']:
        raise ValueError("Invalid predict_model. Please choose 'market', 'market_adj', or 'fama3'.")

    # 检查 min_est_days
    if not isinstance(min_est_days, int) or min_est_days <= 0:
        raise ValueError("min_est_days must be a positive integer")
    if min_est_days > 365:
        raise ValueError(f"min_est_days ({min_est_days}) cannot exceed 365 (事件日前一年最多365个交易日)")
    
    # Basic processing
    df_eventstudy['date1'] = df_eventstudy['date'] - df_eventstudy['eventdate']

    # Sort the data by 'stockid' and 'date'
    df_eventstudy1 = df_eventstudy.sort_values(by=['stockid', 'date'], ascending=True).reset_index().drop(['index'], axis=1)

    # Filter the dataframe to keep only the rows where date1 is between -365 and 0 (inclusive)
    filtered_df = df_eventstudy1[(df_eventstudy1['date1'] >= pd.Timedelta(days=-365)) & (df_eventstudy1['date1'] <= pd.Timedelta(days=0))]

    # Group by (stockid, eventdate) and count the number of rows for each event
    event_counts = filtered_df.groupby(['stockid', 'eventdate']).size()

    # All events from original data
    all_events = df_eventstudy1[['stockid', 'eventdate']].drop_duplicates()

    # Identify events with less than min_est_days trading days in the year before event date
    events_to_remove = set()
    insufficient_event_records = []

    for _, row in all_events.iterrows():
        stockid = row['stockid']
        eventdate = row['eventdate']
        count = event_counts.get((stockid, eventdate), 0)
        if count < min_est_days:
            events_to_remove.add((stockid, eventdate))
            reason = '事件日前一年交易日不足'
            detail = f"事件日发生前一年中交易日总数仅有 {count} 个 (< {min_est_days})"
            if count == 0:
                reason = '事件日前一年完全无交易数据'
                # 检查事件日是否超出数据范围
                min_date = df_eventstudy1['date'].min()
                max_date = df_eventstudy1['date'].max()
                if eventdate > max_date:
                    detail = (
                        f"事件日 {eventdate.date()} 超出数据集日期范围 "
                        f"(数据最新日期: {max_date.date()})。"
                        f"请检查：1) 事件日是否写错；2) 指数/价格数据文件是否已更新到最新日期"
                    )
                elif eventdate < min_date:
                    detail = f"事件日 {eventdate.date()} 早于数据集最早日期 {min_date.date()}，事件日可能发生在上市日之前"
                else:
                    detail = f"事件日 {eventdate.date()} 前一年内无任何交易记录 (< {min_est_days})，可能存在数据缺失"
            insufficient_event_records.append({
                'stockid': stockid,
                'eventdate': eventdate,
                'reason': reason,
                'detail': detail
            })
    # Add to exclusion records
    exclusion_records.extend(insufficient_event_records)

    if len(events_to_remove) > 0:
        print(f"事件日前一年交易日不足检查: {len(events_to_remove)} 个事件因数据不足被移除")

    # Remove these events from the original dataframe
    df_eventstudy2 = df_eventstudy1[~df_eventstudy1.apply(lambda x: (x['stockid'], x['eventdate']) in events_to_remove, axis=1)]
    df_eventstudy2 = df_eventstudy2.reset_index(drop=True)

    df_eventstudy3 = df_eventstudy2[(df_eventstudy2['date1'] >= pd.Timedelta(days=-365)) & (df_eventstudy2['date1'] <= pd.Timedelta(days=365))]

    # Displaying a summary of the operation
    remove_summary = {
        "Total Events in Original Data": df_eventstudy1[['stockid', 'eventdate']].drop_duplicates().shape[0],
        "Events with Insufficient Data": len(events_to_remove),
        "Total Events in Cleaned Data": df_eventstudy2[['stockid', 'eventdate']].drop_duplicates().shape[0]
    }

    print(remove_summary)

    # Sort the data by 'stockid' and 'date'
    df_eventstudy4 = df_eventstudy3.sort_values(by=['stockid', 'date'], ascending=True).reset_index().drop(['index'],axis=1)

    # 用于存储所有事件窗口的AR数据
    all_ar_list = []

    for event_window in event_window_list:
        # Apply the function to each group of (stockid, eventdate)
        # 注意：必须按 (stockid, eventdate) 分组，而非仅按 stockid 分组
        # 否则同一只股票多个事件日时，mark_event_window 只找第一个 date1=0 的索引
        df_eventstudy5 = df_eventstudy4.groupby(['stockid', 'eventdate']).apply(lambda group: mark_event_window(group, est_window, event_window))
        df_eventstudy6 = df_eventstudy5.reset_index(drop=True)
        # Display the first few rows of the updated dataframe to verify the changes
        # df_eventstudy6

        # 取消对事件窗口数据不足的过滤，保留所有事件
        df_eventstudy7 = df_eventstudy6

        df_eventstudy8 = df_eventstudy7.sort_values(by=['stockid', 'date'], ascending=True).reset_index().drop(['index'],axis=1)

        # 更新股票ID列表（按事件过滤后的）
        remaining_events = list(df_eventstudy8[['stockid', 'eventdate']].drop_duplicates().itertuples(index=False, name=None))
        stock_list = list(set([e[0] for e in remaining_events]))

        if predict_model == 'market':
            for id in stock_list:
                # 利用估计窗口数据拟合模型
                condition3 = (
                (df_eventstudy8['est_window'] == 1) &
                (df_eventstudy8['stockid'] == id)
                )

                X = sm.add_constant(df_eventstudy8.loc[condition3, 'mreturn'], has_constant='add')  # 假设mreturn是自变量
                y = df_eventstudy8.loc[condition3, 'sreturn']  # 假设sreturn是因变量
                model = sm.OLS(y, X).fit()

                # 计算残差方差和自由度
                residuals_est = y - model.predict(X)
                df = len(y) - 2  # 自由度 = 观测数 - 参数个数(2: alpha + beta)
                var_res = residuals_est.var(ddof=1)  # 样本方差

                condition4 = (
                (df_eventstudy8['event_window'] == 1) &
                (df_eventstudy8['stockid'] == id)
                )

                # 在事件窗口生成预测值
                X_predict = sm.add_constant(df_eventstudy8.loc[condition4, 'mreturn'], has_constant='add')
                df_eventstudy8.loc[condition4, 'predicted'] = model.predict(X_predict)
                # 保存方差和自由度（用于后续计算统计量）
                df_eventstudy8.loc[condition4, 'var_AR'] = var_res
                df_eventstudy8.loc[condition4, 'df'] = df

        if predict_model == 'fama3':
            for id in stock_list:
                # 利用估计窗口数据拟合模型
                condition_est = (
                    (df_eventstudy8['est_window'] == 1) &
                    (df_eventstudy8['stockid'] == id)
                )

                # 计算超额收益（仅针对当前股票）
                df_eventstudy8.loc[condition_est, 'excess_return'] = (
                    df_eventstudy8.loc[condition_est, 'sreturn'] - df_eventstudy8.loc[condition_est, 'rf']
                )
                df_eventstudy8.loc[condition_est, 'market_excess'] = (
                    df_eventstudy8.loc[condition_est, 'mreturn'] - df_eventstudy8.loc[condition_est, 'rf']
                )

                X = sm.add_constant(df_eventstudy8.loc[condition_est, ['market_excess', 'smb', 'hml']], has_constant='add')
                y = df_eventstudy8.loc[condition_est, 'excess_return']
                model = sm.OLS(y, X).fit()

                # 计算残差方差和自由度
                residuals_est = y - model.predict(X)
                df = len(y) - 4  # 自由度 = 观测数 - 参数个数(4: alpha + beta1 + beta2 + beta3)
                var_res = residuals_est.var(ddof=1)  # 样本方差

                # 在事件窗口生成预测值
                condition_event = (
                    (df_eventstudy8['event_window'] == 1) &
                    (df_eventstudy8['stockid'] == id)
                )

                X_predict = sm.add_constant(df_eventstudy8.loc[condition_event, ['market_excess', 'smb', 'hml']], has_constant='add')  # 添加常数项
                df_eventstudy8.loc[condition_event, 'predicted'] = model.predict(X_predict)
                # 保存方差和自由度（用于后续计算统计量）
                df_eventstudy8.loc[condition_event, 'var_AR'] = var_res
                df_eventstudy8.loc[condition_event, 'df'] = df

        if predict_model == 'market_adj':
            for id in stock_list:
                condition_event = (
                    (df_eventstudy8['event_window'] == 1) &
                    (df_eventstudy8['stockid'] == id)
                )
                df_eventstudy8.loc[condition_event, 'predicted'] = df_eventstudy8.loc[condition_event, 'mreturn']

                condition_est = (
                    (df_eventstudy8['est_window'] == 1) &
                    (df_eventstudy8['stockid'] == id)
                )
                y = df_eventstudy8.loc[condition_est, 'sreturn']
                mean_est = y.mean()
                residuals_est = y - mean_est
                df_deg = len(y) - 1  # 自由度 = 观测数 - 1
                var_res = residuals_est.var(ddof=1)

                df_eventstudy8.loc[condition_event, 'var_AR'] = var_res
                df_eventstudy8.loc[condition_event, 'df'] = df_deg

        # 统一计算 AR = 实际收益率 - 预测收益率
        # 注意：对于 market_adj，predicted 已设为 mreturn
        df_eventstudy8['AR'] = df_eventstudy8['sreturn'] - df_eventstudy8['predicted']

        # 收集当前事件窗口的AR数据
        ar_df = df_eventstudy8[df_eventstudy8['event_window'] == 1].copy()
        
        # 确保 var_AR 和 df 列存在（对缺失值进行填充）
        if 'var_AR' not in ar_df.columns:
            ar_df['var_AR'] = np.nan
        if 'df' not in ar_df.columns:
            ar_df['df'] = np.nan
        '''
        # 验证缺失情况
        var_na = ar_df['var_AR'].isna().sum()
        df_na = ar_df['df'].isna().sum()
        print(f"[验证] var_AR 缺失: {var_na}/{len(ar_df)}, df 缺失: {df_na}/{len(ar_df)}")

        # 对于仍然缺失 var_AR 的行，使用非空均值填充；如果均值也是 NaN，使用 0
        if ar_df['var_AR'].isna().any():
            global_var_mean = ar_df['var_AR'].mean()
            if pd.isna(global_var_mean):
                global_var_mean = 0.0  # 备用默认值
            ar_df['var_AR'] = ar_df['var_AR'].fillna(global_var_mean)
            if ar_df['var_AR'].isna().sum() > 0:  # 重新检查
                ar_df['var_AR'] = ar_df['var_AR'].fillna(0.0)
            print(f"警告: var_AR 缺失已用 {global_var_mean:.6f} 填充")
        
        # 对于仍然缺失 df 的行，使用非空均值填充；如果均值也是 NaN，使用默认值 238
        if ar_df['df'].isna().any():
            global_df_mean = ar_df['df'].mean()
            if pd.isna(global_df_mean):
                global_df_mean = 238.0  # 备用默认值
            ar_df['df'] = ar_df['df'].fillna(global_df_mean)
            if ar_df['df'].isna().sum() > 0:  # 重新检查
                ar_df['df'] = ar_df['df'].fillna(238.0)
            print(f"警告: df 缺失已用 {global_df_mean:.2f} 填充")
        '''
        ar_df['relative_day_raw'] = ar_df['date1'].dt.days
        # 生成事件窗口的连续序号：-1,0,1 或 -3,-2,-1,0,1,2,3
        ar_df = ar_df.sort_values(['stockid', 'date'])
        ar_df['relative_day_adj'] = ar_df.groupby(['stockid', 'eventdate']).cumcount() + event_window[0]
        ar_df['event_window_str'] = f"[{event_window[0]},{event_window[1]}]"

        # 添加估计窗口和事件窗口的详细信息
        ar_df['est_window_start'] = (ar_df['eventdate'] + pd.Timedelta(days=est_window[0])).dt.date
        ar_df['est_window_end'] = (ar_df['eventdate'] + pd.Timedelta(days=est_window[1])).dt.date
        ar_df['est_window_length'] = est_window[1] - est_window[0]
        ar_df['event_window_start'] = (ar_df['eventdate'] + pd.Timedelta(days=event_window[0])).dt.date
        ar_df['event_window_end'] = (ar_df['eventdate'] + pd.Timedelta(days=event_window[1])).dt.date
        ar_df['event_window_length'] = event_window[1] - event_window[0] + 1

        # 计算事件窗口内的累积超额收益 (CAR)
        ar_df['CAR'] = ar_df.groupby(['stockid', 'eventdate'])['AR'].cumsum()
        
        # 计算统计量（参考 eventstudy-master 的实现）
        # Std.E.AR = sqrt(var_AR)，其中 var_AR 是估计窗口残差的方差
        ar_df['Std. E. AR'] = np.sqrt(ar_df['var_AR'])
        
        # 计算 Std.E.CAR：var_CAR 是 var_AR 的累加和
        ar_df['var_CAR'] = ar_df.groupby(['stockid', 'eventdate'])['var_AR'].cumsum()
        ar_df['Std. E. CAR'] = np.sqrt(ar_df['var_CAR'])
        
        # 计算 T-stat = CAR / Std.E.CAR
        ar_df['CAR_T-stat'] = ar_df['CAR'] / ar_df['Std. E. CAR']
        
        # 计算 P-value：双尾检验，基于 t 分布
        ar_df['CAR_P-value'] = 2 * (1 - stats.t.cdf(np.abs(ar_df['CAR_T-stat']), ar_df['df']))

        all_ar_list.append(ar_df[['stockid', 'date', 'eventdate',
                                  'est_window_start', 'est_window_end', 'est_window_length',
                                  'event_window_start', 'event_window_end', 'event_window_length', 'event_window_str',
                                  'relative_day_raw', 'relative_day_adj',
                                  'AR', 'var_AR', 'df', 'CAR', 'Std. E. AR', 'Std. E. CAR', 'CAR_T-stat', 'CAR_P-value']].copy())

    # ============================================================
    # 输出每个公司在事件窗口期内每一天的AR，以及所有公司的平均AR
    # ============================================================
    df_ar_all = pd.concat(all_ar_list, ignore_index=True)
    df_ar_all = df_ar_all.sort_values(['stockid', 'event_window_str','date','eventdate','relative_day_raw']).reset_index(drop=True)

    # 对每个事件窗口进行T检验：基于AR累计和计算Mean CAR
    t_test_results = {}
    for event_window_str in df_ar_all['event_window_str'].unique():
        df_window = df_ar_all[df_ar_all['event_window_str'] == event_window_str]
        # 计算每个公司AR在事件窗口内的累计和（即CAR）
        car_by_company = df_window.groupby(['stockid', 'eventdate'])['AR'].sum().reset_index()
        car_by_company.columns = ['stockid', 'eventdate', 'CAR']
        # Mean CAR = 所有公司CAR的均值 = AR累计和 / 事件总数
        Mean_CAR = car_by_company['CAR'].mean()
        t_stat, p_value = stats.ttest_1samp(car_by_company['CAR'].dropna(), 0)
        t_test_results[event_window_str] = {'Mean CAR': Mean_CAR, 't_stat': t_stat, 'p_value': p_value}

    # 将T检验结果转换为DataFrame
    # 字段	计算方式
    # Mean CAR：每个公司AR在事件窗口内的累计和 / 事件总数
    # t_stat：scipy.stats.ttest_1samp() 单样本 t 检验，检验 CAR 是否显著不为 0
    # p_value：t 检验的 p 值（双尾检验）

    t_test_df = pd.DataFrame(t_test_results).T
    t_test_df.reset_index(inplace=True)
    t_test_df.rename(columns={'index': 'Time Window'}, inplace=True)

    # 1. 每个公司在事件窗口期内每一天的AR（不含 var_AR, df）
    ar_cols = ['stockid', 'event_window_str', 'date', 'eventdate',
                'relative_day_raw', 'relative_day_adj',
               'est_window_start', 'est_window_end', 'est_window_length',
               'event_window_start', 'event_window_end', 'event_window_length',
               'AR', 'var_AR', 'df', 'Std. E. AR', 'CAR', 'Std. E. CAR', 'CAR_T-stat', 'CAR_P-value']
    df_ar_by_company = df_ar_all[ar_cols].copy()
    # 确保 stockid 保存为字符串（带前导零），避免读取时被 pandas 解析为整数
    df_ar_by_company['stockid'] = df_ar_by_company['stockid'].astype(str).str.zfill(6)
    ar_company_file = os.path.join(save_path, f'{predict_model}_model_AR_by_company.csv')
    # counter = 0
    # while os.path.isfile(ar_company_file):
    #     counter += 1
    #     ar_company_file = os.path.join(save_path, f'{predict_model}_model_AR_by_company_{counter}.csv')
    df_ar_by_company.to_csv(ar_company_file, index=False, encoding='utf-8-sig')
    print(f"每个公司在事件窗口期内的AR已保存: {ar_company_file}")

    # 2. 所有公司在事件窗口期内每一天的平均AR (AAR)、平均CAR (ACAR) 和平均累积AR (CAAR)
    # 首先计算每个公司的 var_AR 和 df 的汇总（按天）
    df_aar = df_ar_all.groupby(['event_window_str', 'relative_day_adj']).agg(
        AAR=('AR', 'mean'),
        Median_AR=('AR', 'median'),
        count=('AR', 'count'),
        sum_var_AR=('var_AR', 'sum'),
        sum_df=('df', 'sum')
    ).reset_index()
    df_aar = df_aar.sort_values(['event_window_str', 'relative_day_adj']).reset_index(drop=True)

    # 计算 Positive AR 和 Negative AR 的比例（转为百分比）
    def calc_positive_ratio(x):
        return (x > 0).sum() / len(x) * 100

    def calc_negative_ratio(x):
        return (x < 0).sum() / len(x) * 100

    positive_negative = df_ar_all.groupby(['event_window_str', 'relative_day_adj']).agg(
        Positive_AR=('AR', calc_positive_ratio),
        Negative_AR=('AR', calc_negative_ratio)
    ).reset_index()

    df_aar = df_aar.merge(positive_negative, on=['event_window_str', 'relative_day_adj'], how='left')

    # 计算 ACAR（平均CAR，是 CAR 的均值而非累加）
    df_aar['ACAR'] = df_ar_all.groupby(['event_window_str', 'relative_day_adj'])['CAR'].mean().values

    # 计算 Median_CAR（中位数 CAR）
    df_aar['Median_CAR'] = df_ar_all.groupby(['event_window_str', 'relative_day_adj'])['CAR'].median().values

    # 计算 CAR 的符号检验统计量（用于 Positive_CAR 和 Negative_CAR）
    car_positive = []
    car_negative = []
    for _, row in df_aar.iterrows():
        subset = df_ar_all[(df_ar_all['event_window_str'] == row['event_window_str']) &
                          (df_ar_all['relative_day_adj'] == row['relative_day_adj'])]['CAR']
        car_positive.append((subset > 0).sum() / len(subset) * 100)
        car_negative.append((subset < 0).sum() / len(subset) * 100)
    df_aar['Positive_CAR(%)'] = car_positive
    df_aar['Negative_CAR(%)'] = car_negative

    # 计算 Std.E.AAR
    N = df_aar['count']  # 每天的公司数量
    df_aar['Std. E. AAR'] = np.sqrt(df_aar['sum_var_AR'] / (N ** 2))

    # ACAR 和 CAAR 在数学上是相等的（ACAR = mean(CAR[i,t]), CAAR = sum(AAR[0:t])）
    # 但它们的方差含义不同：
    # - ACAR 的方差：横截面上的方差（CAR 值的离散程度）
    # - CAAR 的方差：时间序列上的方差（AAR 累加的不确定性）
    # 因此 Std.E.ACAR 与 Std.E.CAAR 应该分开计算

    # 计算 ACAR 的标准误（使用与 CAAR 相同的累加方法）
    # var_ACAR[t] = sum(var_AAR[0:t])，即 AAR 方差的累加和
    df_aar['var_ACAR'] = df_aar.groupby('event_window_str')['sum_var_AR'].cumsum() / (N ** 2)
    df_aar['Std. E. ACAR'] = np.sqrt(df_aar['var_ACAR'])
    df_aar['ACAR_T-stat'] = df_aar['ACAR'] / df_aar['Std. E. ACAR']
    df_aar['ACAR_P-value'] = 2 * (1 - stats.t.cdf(np.abs(df_aar['ACAR_T-stat']), df_aar['sum_df']))

    # CAAR = AAR 的累积和（按事件窗口分组）
    df_aar['CAAR'] = df_aar.groupby('event_window_str')['AAR'].cumsum()

    # Std.E.CAAR = sqrt(累计 var_AAR)
    # CAAR 与 ACAR 的方差累加方式相同（因为 CAAR = sum(AAR[0:t]) = ACAR）
    df_aar['var_CAAR'] = df_aar.groupby('event_window_str')['sum_var_AR'].cumsum() / (N ** 2)
    df_aar['Std. E. CAAR'] = np.sqrt(df_aar['var_CAAR'])

    # CAAR_T-stat = CAAR / Std.E.CAAR
    df_aar['CAAR_T-stat'] = df_aar['CAAR'] / df_aar['Std. E. CAAR']

    # CAAR_P-value：双尾检验，基于 t 分布
    # 自由度使用累计 df（所有公司的 df 之和）
    df_aar['df_cum'] = df_aar.groupby('event_window_str')['sum_df'].cumsum()
    df_aar['CAAR_P-value'] = 2 * (1 - stats.t.cdf(np.abs(df_aar['CAAR_T-stat']), df_aar['df_cum']))

    # 计算 AAR 的 t-statistic
    # AAR_T-stat = AAR / Std.E.AAR
    df_aar['AAR_T-stat'] = df_aar['AAR'] / df_aar['Std. E. AAR']
    # AAR_T-stat 的 P-value：双尾 t 检验
    df_aar['AAR_P-value'] = 2 * (1 - stats.t.cdf(np.abs(df_aar['AAR_T-stat']), df_aar['sum_df']))

    # 计算 Wilcoxon signed-rank test Z-statistic（用于中位数 AR 检验）
    # 需要对每个事件窗口-相对日组合单独计算
    from scipy.stats import wilcoxon, norm
    wilcoxon_z = []
    for _, row in df_aar.iterrows():
        subset = df_ar_all[(df_ar_all['event_window_str'] == row['event_window_str']) &
                          (df_ar_all['relative_day_adj'] == row['relative_day_adj'])]['AR']
        if len(subset) < 2 or subset.std() == 0:
            wilcoxon_z.append(np.nan)
        else:
            try:
                stat, p = wilcoxon(subset)
                # 转换为 Z-statistic：使用正态近似
                n = len(subset)
                # 在 n 较大时，Wilcoxon 统计量近似正态分布
                # E[W] = n(n+1)/4, Var[W] = n(n+1)(2n+1)/24
                z = (stat - n*(n+1)/4) / np.sqrt(n*(n+1)*(2*n+1)/24)
                wilcoxon_z.append(z)
            except:
                wilcoxon_z.append(np.nan)
    df_aar['Wilcoxon_Z_Median_AR'] = wilcoxon_z
    # Wilcoxon_Z 的 P-value：双尾正态检验
    df_aar['Wilcoxon_Z_Median_AR_P-value'] = 2 * (1 - norm.cdf(np.abs(df_aar['Wilcoxon_Z_Median_AR'])))

    # 计算 Median_CAR 的 Wilcoxon 检验（与 Median_CAR 对应的检验）
    wilcoxon_car_z = []
    for _, row in df_aar.iterrows():
        subset = df_ar_all[(df_ar_all['event_window_str'] == row['event_window_str']) &
                          (df_ar_all['relative_day_adj'] == row['relative_day_adj'])]['CAR']
        if len(subset) < 2 or subset.std() == 0:
            wilcoxon_car_z.append(np.nan)
        else:
            try:
                stat, p = wilcoxon(subset)
                n = len(subset)
                z = (stat - n*(n+1)/4) / np.sqrt(n*(n+1)*(2*n+1)/24)
                wilcoxon_car_z.append(z)
            except:
                wilcoxon_car_z.append(np.nan)
    df_aar['Wilcoxon_Z_Median_CAR'] = wilcoxon_car_z
    df_aar['Wilcoxon_Z_Median_CAR_P-value'] = 2 * (1 - norm.cdf(np.abs(df_aar['Wilcoxon_Z_Median_CAR'])))

    # 计算 Binomial sign test Z-statistic（用于正收益比例的检验）
    # 零假设：正收益比例 = 0.5（二项检验）
    # Z = (p - 0.5) / sqrt(0.25 / n)
    def binomial_z(positive_pct, n):
        if n < 2:
            return np.nan
        p = positive_pct / 100  # 转为比例
        se = np.sqrt(0.25 / n)
        if se == 0:
            return np.nan
        return (p - 0.5) / se

    # 先重命名列
    df_aar.rename(columns={'count': 'N_Events', 'Positive_AR': 'Positive_AR(%)', 'Negative_AR': 'Negative_AR(%)'}, inplace=True)

    # 计算 AR 的符号检验
    df_aar['Sign_Z_Positive_AR'] = df_aar.apply(lambda row: binomial_z(row['Positive_AR(%)'], row['N_Events']), axis=1)
    df_aar['Sign_Z_Positive_AR_P-value'] = 2 * (1 - norm.cdf(np.abs(df_aar['Sign_Z_Positive_AR'])))

    # 计算 CAR 的符号检验（与 Positive_CAR(%) 对应的检验）
    df_aar['Sign_Z_Positive_CAR'] = df_aar.apply(lambda row: binomial_z(row['Positive_CAR(%)'], row['N_Events']), axis=1)
    df_aar['Sign_Z_Positive_CAR_P-value'] = 2 * (1 - norm.cdf(np.abs(df_aar['Sign_Z_Positive_CAR'])))

    # 清理中间列
    df_aar = df_aar.drop(columns=['sum_var_AR', 'sum_df', 'var_ACAR', 'var_CAAR', 'df_cum'])

    # 按指定顺序重新排列列
    column_order = [
        'event_window_str', 'relative_day_adj',
        'AAR', 'Std. E. AAR', 'AAR_T-stat', 'AAR_P-value',
        'Median_AR', 'Wilcoxon_Z_Median_AR', 'Wilcoxon_Z_Median_AR_P-value',
        'Positive_AR(%)', 'Negative_AR(%)', 'Sign_Z_Positive_AR', 'Sign_Z_Positive_AR_P-value',
        'ACAR', 'Std. E. ACAR', 'ACAR_T-stat', 'ACAR_P-value',
        'Median_CAR','Wilcoxon_Z_Median_CAR', 'Wilcoxon_Z_Median_CAR_P-value',
        'Positive_CAR(%)', 'Negative_CAR(%)',
        'Sign_Z_Positive_CAR', 'Sign_Z_Positive_CAR_P-value',
        'CAAR', 'Std. E. CAAR', 'CAAR_T-stat', 'CAAR_P-value',
        'N_Events'
    ]
    df_aar = df_aar[column_order]

    aar_file = os.path.join(save_path, f'{predict_model}_model_AAR_daily.csv')
    df_aar.to_csv(aar_file, index=False, encoding='utf-8-sig')
    print(f"所有公司每天的平均AR (AAR) 已保存: {aar_file}")

    # 如果用户没有提供save_path，使用当前工作目录作为默认保存路径
    if save_path is None:
        save_path = os.getcwd()

    # 初始文件名
    file_name = f'{predict_model}_model_ttest.xlsx'
    file_path = os.path.join(save_path, file_name)

    # 如果文件已经存在，找到一个新的文件名
    # counter = 0
    # while os.path.isfile(file_path):
    #     counter += 1
    #     # 添加一个数字标识（1, 2, 3, ...）到文件名中
    #     file_name = f'{predict_model}_model_ttest_{counter}.xlsx'
    #     file_path = os.path.join(save_path, file_name)

    # 将DataFrame保存到指定的Excel文件
    t_test_df.to_excel(file_path, index=False)

    # 创建消息以确认文件已经保存
    message = f"The file has been saved to: {file_path}"
    print(message)

    # ============================================================
    # 保存异常记录到CSV文件
    # ============================================================
    df_exclusion = pd.DataFrame(exclusion_records)
    if len(df_exclusion) > 0:
        exclusion_file_path = os.path.join(save_path, f'{predict_model}_model_exclusion_records.csv')
        df_exclusion = df_exclusion.sort_values(['reason', 'stockid', 'eventdate']).reset_index(drop=True)
        df_exclusion.to_csv(exclusion_file_path, index=False, encoding='utf-8-sig')
        print(f"异常记录已保存: {exclusion_file_path}")
        print(f"异常记录汇总: 共 {len(df_exclusion)} 条")
        if not df_exclusion.empty:
            print(df_exclusion.groupby('reason').size())
    else:
        print("异常记录: 没有需要排除的记录")

    # ============================================================
    # 打印每个公司每个事件日的数据（如果启用）
    # ============================================================
    if print_event_details:
        print("\n" + "="*80)
        print(f"打印每个公司每个事件日前后 {event_detail_days} 天的数据")
        print("="*80)
        
        # 获取所有(stockid, eventdate)组合
        event_pairs = df_ar_by_company[['stockid', 'eventdate']].drop_duplicates()
        
        # 准备原始数据（用于显示 sreturn, mreturn, date1）
        df_display = df_eventstudy8[['stockid', 'date', 'eventdate', 'sreturn', 'mreturn', 'date1']].copy()
        
        # 获取被排除的记录
        excluded_pairs = set()
        for record in exclusion_records:
            excluded_pairs.add((record['stockid'], record['eventdate']))
        
        # 获取被过滤后保留下来的所有原始事件
        kept_pairs = set(df_eventstudy8[['stockid', 'eventdate']].drop_duplicates().apply(
            lambda x: (x['stockid'], x['eventdate']), axis=1
        ))
        
        # 如果指定了 check_stockid，先检查该股票的所有原始事件
        if check_stockid is not None:
            print(f"\n>>> 检查股票 {check_stockid} 的原始事件数据 <<<")
            # 从 df_eventstudy1 获取原始数据
            stock_original = df_eventstudy1[df_eventstudy1['stockid'] == check_stockid]
            stock_events = stock_original[['stockid', 'eventdate']].drop_duplicates()
            print(f"原始事件数: {len(stock_events)}")
            for _, event_row in stock_events.iterrows():
                sid = event_row['stockid']
                edate = event_row['eventdate']
                edate_str = pd.to_datetime(edate).strftime('%Y-%m-%d')
                if (sid, edate) in excluded_pairs:
                    exclusion_info = [r for r in exclusion_records if r['stockid'] == sid and r['eventdate'] == edate]
                    print(f"  - {sid} | {edate_str} [已被排除]")
                    for info in exclusion_info:
                        print(f"      原因: {info['reason']}")
                        print(f"      详情: {info['detail']}")
                elif (sid, edate) in kept_pairs:
                    print(f"  - {sid} | {edate_str} [已保留]")
                else:
                    print(f"  - {sid} | {edate_str} [未在最终数据中]")
            print()
        
        for _, row in event_pairs.iterrows():
            stockid = row['stockid']
            eventdate = row['eventdate']
            
            # 如果指定了 check_stockid，只打印该股票
            if check_stockid is not None and stockid != check_stockid:
                continue
            
            # 检查这个事件是否在排除列表中
            if (stockid, eventdate) in excluded_pairs:
                exclusion_info = [r for r in exclusion_records if r['stockid'] == stockid and r['eventdate'] == eventdate]
                print(f"\n--- 公司 {stockid} | 事件日 {pd.to_datetime(eventdate).strftime('%Y-%m-%d')} [已被排除] ---")
                for info in exclusion_info:
                    print(f"  排除原因: {info['reason']}")
                    print(f"  详情: {info['detail']}")
                continue
            
            # 从原始数据中筛选（确保日期格式一致）
            eventdate_dt = pd.to_datetime(eventdate)
            check_df = df_display[
                (df_display['stockid'] == stockid) & 
                (df_display['eventdate'] == eventdate_dt)
            ].copy()
            
            # 筛选日期范围内（事件日前后 event_detail_days 天）
            check_df = check_df[
                (check_df['date1'] >= pd.Timedelta(days=-event_detail_days)) & 
                (check_df['date1'] <= pd.Timedelta(days=event_detail_days))
            ].sort_values('date1')
            
            if len(check_df) > 0:
                print(f"\n--- 公司 {stockid} | 事件日 {pd.to_datetime(eventdate).strftime('%Y-%m-%d')} ---")
                # 格式化输出
                check_df = check_df.reset_index(drop=True)
                check_df['date'] = pd.to_datetime(check_df['date']).dt.strftime('%Y-%m-%d')
                check_df['eventdate'] = pd.to_datetime(check_df['eventdate']).dt.strftime('%Y-%m-%d')
                check_df['date1'] = check_df['date1'].astype(str)
                print(check_df.to_string(index=False))
            else:
                print(f"\n--- 公司 {stockid} | 事件日 {pd.to_datetime(eventdate).strftime('%Y-%m-%d')} ---")
                print("该事件日在指定范围内无数据")
        
        print("\n" + "="*80)

    return t_test_df, df_ar_by_company, df_aar, df_exclusion