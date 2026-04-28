"""
事件研究结果可视化脚本
"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator, PercentFormatter
import numpy as np
import pandas as pd
from scipy.stats import t, norm
import seaborn as sns
from IPython.display import display
from typing import Optional, List, Tuple, Dict, Union

# 设置中文字体支持
# plt.rcParams['font.sans-serif'] = ['SimSun', '宋体', 'DejaVu Sans']
# plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['STSong','Microsoft YaHei', 'SimHei', 'DejaVu Sans'],
    'axes.unicode_minus': False
})

# 可自行修改的配色
line_color = '#1f77b4'   # 曲线颜色
ci_color = '#1f77b4'     # 置信区间颜色（通常与曲线同色）
bar_color = 'grey'       # AAR 柱状图颜色
line_width = 1.8
bar_width=0.3
bar_alpha=0.4
ci_alpha = 0.15
class EventStudyPlotter:
    """
    专门用于事件研究结果可视化的类
    
    支持两种数据源：
    1. 单个事件研究结果（Single 类风格）
    2. 多个事件聚合结果（Multiple 类风格）
    3. 直接从 df_aar DataFrame 绘制（兼容 event_analysis.py）
    """
    
    def __init__(self, figsize: Tuple[float, float] = (12, 6)):
        """
        初始化绘图器
        
        Parameters
        ----------
        figsize : tuple, optional
            图形大小，默认 (12, 6)
        """
        self.figsize = figsize
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
    
    def plot_single_event(
        self,
        event_name: str,
        time: np.ndarray,
        CAR: np.ndarray,
        AR: Optional[np.ndarray] = None,
        var_CAR: Optional[np.ndarray] = None,
        df: Optional[int] = None,
        confidence: float = 0.90,
        ax: Optional[plt.Axes] = None,
        car_color: str = '#1f77b4',
        ar_color: str = '#2ca02c',
        ci_color: str = 'gray',
        show_ar: bool = False,
        show_ci: bool = True,
        event_line_style: dict = None,
    ) -> plt.Figure:
        """
        绘制单个事件的 CAR（可选 AR）图
        
        Parameters
        ----------
        event_name : str
            事件名称（用于标题）
        time : np.ndarray
            时间轴（相对天数）
        CAR : np.ndarray
            累积异常收益率
        AR : np.ndarray, optional
            每日异常收益率（柱状图）
        var_CAR : np.ndarray, optional
            CAR 的方差（用于置信区间）
        df : int, optional
            自由度（用于置信区间）
        confidence : float
            置信水平，默认 0.90
        ax : plt.Axes, optional
            指定的坐标轴（用于子图）
        car_color : str
            CAR 线条颜色
        ar_color : str
            AR 柱状图颜色
        ci_color : str
            置信区间填充颜色
        show_ar : bool
            是否显示 AR 柱状图
        show_ci : bool
            是否显示置信区间
        event_line_style : dict
            事件日垂直线样式
            
        Returns
        -------
        fig : matplotlib.figure.Figure
            图形对象
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize)
        else:
            fig = ax.figure
        
        ax.plot(time, CAR, 
                color=car_color, 
                linewidth=2.5, 
                marker='o', 
                markersize=5,
                label='CAR',
                zorder=3)
        
        if show_ci and var_CAR is not None and df is not None:
            delta = np.sqrt(var_CAR) * t.ppf(confidence, df)
            upper = CAR + delta
            lower = CAR - delta
            ax.fill_between(time, lower, upper, 
                           color=ci_color, 
                           alpha=0.2, 
                           label=f'{int(confidence*100)}% CI',
                           zorder=1)
        
        if show_ar and AR is not None:
            pos_mask = AR >= 0
            neg_mask = AR < 0
            if pos_mask.any():
                ax.vlines(time[pos_mask], 0, AR[pos_mask], 
                         color='green', 
                         linewidth=2,
                         alpha=0.6)
            if neg_mask.any():
                ax.vlines(time[neg_mask], 0, AR[neg_mask], 
                         color='red', 
                         linewidth=2,
                         alpha=0.6)
        
        if event_line_style is None:
            event_line_style = {'color': 'black', 'linewidth': 1.5, 'linestyle': '--'}
        ax.axvline(x=0, **event_line_style, label='Event Day')
        ax.axhline(y=0, color='black', linewidth=0.8, linestyle=':', alpha=0.5)
        
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_xlabel('Relative Day', fontsize=11)
        ax.set_ylabel('Return', fontsize=11)
        ax.set_title(f'Event Study: {event_name}', fontsize=13, fontweight='bold')
        ax.legend(loc='best', frameon=True, fancybox=True, shadow=True)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_multiple_events(
        self,
        sample_name: str,
        time: np.ndarray,
        CAAR: np.ndarray,
        AAR: Optional[np.ndarray] = None,
        var_CAAR: Optional[np.ndarray] = None,
        df: Optional[Union[int, np.ndarray]] = None,
        confidence: float = 0.90,
        ax: Optional[plt.Axes] = None,
        caar_color: str = '#d62728',
        aar_color: str = '#ff7f0e',
        ci_color: str = 'gray',
        show_aar: bool = False,
        show_ci: bool = True,
    ) -> plt.Figure:
        """
        绘制多个事件聚合的 CAAR（可选 AAR）图
        
        Parameters
        ----------
        sample_name : str
            样本名称（用于标题）
        time : np.ndarray
            时间轴（相对天数）
        CAAR : np.ndarray
            累积平均异常收益率
        AAR : np.ndarray, optional
            平均异常收益率
        var_CAAR : np.ndarray, optional
            CAAR 的方差
        df : int or np.ndarray, optional
            自由度
        confidence : float
            置信水平
        ax : plt.Axes, optional
            指定的坐标轴
        caar_color : str
            CAAR 线条颜色
        aar_color : str
            AAR 柱状图颜色
        ci_color : str
            置信区间颜色
        show_aar : bool
            是否显示 AAR 柱状图
        show_ci : bool
            是否显示置信区间
            
        Returns
        -------
        fig : matplotlib.figure.Figure
            图形对象
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize)
        else:
            fig = ax.figure
        
        ax.plot(time, CAAR, 
                color=caar_color, 
                linewidth=2.5, 
                marker='s', 
                markersize=5,
                label='CAAR',
                zorder=3)
        
        if show_ci and var_CAAR is not None and df is not None:
            delta = np.sqrt(var_CAAR) * t.ppf(confidence, df)
            upper = CAAR + delta
            lower = CAAR - delta
            ax.fill_between(time, lower, upper, 
                           color=ci_color, 
                           alpha=0.2, 
                           label=f'{int(confidence*100)}% CI',
                           zorder=1)
        
        if show_aar and AAR is not None:
            ax.bar(time, AAR, 
                  color=aar_color, 
                  alpha=0.6, 
                  width=0.6,
                  label='AAR',
                  zorder=2)
        
        ax.axvline(x=0, color='black', linewidth=1.5, linestyle='--', label='Event Day')
        ax.axhline(y=0, color='black', linewidth=0.8, linestyle=':', alpha=0.5)
        
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_xlabel('Relative Day', fontsize=11)
        ax.set_ylabel('Return', fontsize=11)
        ax.set_title(f'Event Study (Multiple Events): {sample_name}', 
                    fontsize=13, fontweight='bold')
        ax.legend(loc='best', frameon=True, fancybox=True, shadow=True)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_from_dataframe(
        self,
        df_aar: pd.DataFrame,
        event_window_col: str = 'event_window_str',
        day_col: str = 'relative_day_adj',
        car_col: str = 'CAAR',
        aar_col: str = 'AAR',
        std_car_col: str = 'Std. E. CAAR',
        std_aar_col: str = 'Std. E. AAR',
        title: str = 'Event Study Results',
        show_aar: bool = False,
        show_ci: bool = True,
        confidence: float = 0.90,
        figsize: Tuple[float, float] = None,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        从 DataFrame（event_analysis.py 的 df_aar）直接绘图
        
        Parameters
        ----------
        df_aar : pd.DataFrame
            包含 AAR/CAAR 等结果的 DataFrame
        event_window_col : str
            事件窗口列名
        day_col : str
            相对日期列名
        car_col : str
            CAR/CAAR 列名
        aar_col : str
            AAR 列名
        std_car_col : str
            CAR 标准误列名
        std_aar_col : str
            AAR 标准误列名
        title : str
            图表标题
        show_aar : bool
            是否显示 AAR 柱状图
        show_ci : bool
            是否显示置信区间
        confidence : float
            置信水平
        figsize : tuple
            图形大小
        save_path : str, optional
            保存路径
            
        Returns
        -------
        fig : matplotlib.figure.Figure
            图形对象
        """
        if figsize is None:
            figsize = self.figsize
        
        first_window = df_aar[event_window_col].iloc[0]
        subset = df_aar[df_aar[event_window_col] == first_window].copy()
        subset = subset.sort_values(day_col)
        
        time = subset[day_col].values
        CAAR = subset[car_col].values
        
        df_col = [col for col in df_aar.columns if 'df' in col.lower() and 'cum' in col.lower()]
        if df_col and len(subset) > 0:
            df_val = subset[df_col[0]].iloc[-1]
        else:
            df_val = len(subset) - 1
        
        fig, ax = plt.subplots(figsize=figsize)
        
        ax.plot(time, CAAR, 
                color='#d62728', 
                linewidth=2.5, 
                marker='s', 
                markersize=6,
                label='CAAR',
                zorder=3)
        
        if show_ci and std_car_col in subset.columns:
            delta = subset[std_car_col].values * t.ppf(confidence, df_val)
            ax.fill_between(time, CAAR - delta, CAAR + delta,
                           color='gray', alpha=0.2, 
                           label=f'{int(confidence*100)}% CI',
                           zorder=1)
        
        if show_aar and aar_col in subset.columns:
            ax.bar(time, subset[aar_col].values,
                  color='#ff7f0e', alpha=0.6, width=0.6,
                  label='AAR',
                  zorder=2)
        
        ax.axvline(x=0, color='black', linewidth=1.5, linestyle='--', label='Event Day')
        ax.axhline(y=0, color='black', linewidth=0.8, linestyle=':', alpha=0.5)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        
        ax.set_xlabel('Relative Day', fontsize=11)
        ax.set_ylabel('Cumulative Abnormal Return', fontsize=11)
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='best', frameon=True, fancybox=True, shadow=True)
        ax.grid(True, alpha=0.3)
        
        info_text = f'Event Window: {first_window}'
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
                fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f'Figure saved to: {save_path}')
        
        return fig
    
    def plot_comparison(
        self,
        results_dict: Dict[str, Dict],
        metric: str = 'CAAR',
        figsize: Tuple[float, float] = (14, 7),
        colors: List[str] = None,
        show_ci: bool = True,
        confidence: float = 0.90,
        title: str = 'Event Study Comparison',
    ) -> plt.Figure:
        """
        对比多个事件研究结果
        
        Parameters
        ----------
        results_dict : dict
            {名称: {time: array, metric: array, var: array, df: int/array}}
        metric : str
            要比较的指标 ('CAAR' 或 'AAR')
        figsize : tuple
            图形大小
        colors : list
            颜色列表
        show_ci : bool
            是否显示置信区间
        confidence : float
            置信水平
        title : str
            图表标题
            
        Returns
        -------
        fig : matplotlib.figure.Figure
            图形对象
        """
        if colors is None:
            colors = plt.cm.Set2(np.linspace(0, 1, len(results_dict)))
        
        fig, ax = plt.subplots(figsize=figsize)
        
        for idx, (name, data) in enumerate(results_dict.items()):
            time = data['time']
            values = data[metric]
            
            ax.plot(time, values, 
                   color=colors[idx], 
                   linewidth=2.5, 
                   marker='o' if len(results_dict) <= 5 else None,
                   markersize=4,
                   label=name,
                   zorder=3)
            
            if show_ci and 'var' in data and 'df' in data:
                var = data['var']
                df_val = data['df']
                delta = np.sqrt(var) * t.ppf(confidence, df_val)
                ax.fill_between(time, values - delta, values + delta,
                               color=colors[idx], alpha=0.15,
                               zorder=1)
        
        ax.axvline(x=0, color='black', linewidth=1.5, linestyle='--', label='Event Day')
        ax.axhline(y=0, color='black', linewidth=0.8, linestyle=':', alpha=0.5)
        
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_xlabel('Relative Day', fontsize=11)
        ax.set_ylabel(f'{metric}', fontsize=11)
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='best', frameon=True, fancybox=True, shadow=True)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_car_distribution(
        self,
        CAR_values: np.ndarray,
        event_date: str = '',
        figsize: Tuple[float, float] = (10, 6),
        bins: int = 20,
        kde: bool = True,
    ) -> plt.Figure:
        """
        绘制 CAR 分布直方图
        
        Parameters
        ----------
        CAR_values : np.ndarray
            CAR 值数组（多个公司的最终 CAR）
        event_date : str
            事件日期
        figsize : tuple
            图形大小
        bins : int
            直方图分箱数
        kde : bool
            是否显示核密度估计
            
        Returns
        -------
        fig : matplotlib.figure.Figure
            图形对象
        """
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        axes[0].hist(CAR_values, bins=bins, 
                    edgecolor='black', alpha=0.7, density=True)
        if kde:
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(CAR_values)
            x_range = np.linspace(CAR_values.min(), CAR_values.max(), 100)
            axes[0].plot(x_range, kde(x_range), color='red', linewidth=2, label='KDE')
        
        axes[0].axvline(x=0, color='black', linewidth=1, linestyle='--')
        axes[0].axvline(x=CAR_values.mean(), color='green', linewidth=2, 
                       linestyle='-', label=f'Mean: {CAR_values.mean():.4f}')
        axes[0].set_xlabel('CAR')
        axes[0].set_ylabel('Density')
        axes[0].set_title(f'CAR Distribution\n{event_date}')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        axes[1].boxplot(CAR_values, vert=True, patch_artist=True,
                       boxprops=dict(facecolor='lightblue'))
        axes[1].axhline(y=0, color='red', linewidth=1, linestyle='--')
        axes[1].set_ylabel('CAR')
        axes[1].set_title('CAR Boxplot')
        axes[1].grid(True, alpha=0.3)
        
        stats_text = f"""
        Mean: {CAR_values.mean():.4f}
        Std: {CAR_values.std():.4f}
        Min: {CAR_values.min():.4f}
        Max: {CAR_values.max():.4f}
        N: {len(CAR_values)}
        t-stat: {CAR_values.mean()/(CAR_values.std()/np.sqrt(len(CAR_values))):.3f}
        """
        axes[1].text(1.1, 0.5, stats_text, transform=axes[1].transAxes,
                    fontsize=9, verticalalignment='center',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        return fig
    
    def plot_significance_summary(
        self,
        df_aar: pd.DataFrame,
        event_window_col: str = 'event_window_str',
        day_col: str = 'relative_day_adj',
        car_col: str = 'CAAR',
        pval_col: str = 'CAAR_P-value',
        figsize: Tuple[float, float] = (12, 6),
    ) -> plt.Figure:
        """
        绘制显著性热图或气泡图
        
        Parameters
        ----------
        df_aar : pd.DataFrame
            结果 DataFrame
        event_window_col : str
            事件窗口列
        day_col : str
            日期列
        car_col : str
            CAR 列
        pval_col : str
            P 值列
        figsize : tuple
            图形大小
            
        Returns
        -------
        fig : matplotlib.figure.Figure
            图形对象
        """
        fig, axes = plt.subplots(1, 2, figsize=figsize, 
                                 gridspec_kw={'width_ratios': [3, 1]})
        
        pivot_df = df_aar.pivot_table(
            index=event_window_col, 
            columns=day_col, 
            values=car_col,
            aggfunc='first'
        )
        pivot_pval = df_aar.pivot_table(
            index=event_window_col, 
            columns=day_col, 
            values=pval_col,
            aggfunc='first'
        )
        
        ax1 = axes[0]
        im = ax1.imshow(pivot_df, aspect='auto', cmap='RdBu_r', 
                       vmin=-abs(pivot_df).max().max(), 
                       vmax=abs(pivot_df).max().max())
        ax1.set_xticks(range(len(pivot_df.columns)))
        ax1.set_xticklabels(pivot_df.columns)
        ax1.set_yticks(range(len(pivot_df.index)))
        ax1.set_yticklabels(pivot_df.index)
        ax1.set_xlabel('Relative Day')
        ax1.set_ylabel('Event Window')
        ax1.set_title('CAR Heatmap')
        
        for i in range(len(pivot_df.index)):
            for j in range(len(pivot_df.columns)):
                if not np.isnan(pivot_df.iloc[i, j]):
                    text = ax1.text(j, i, f'{pivot_df.iloc[i, j]:.3f}',
                                   ha="center", va="center",
                                   color="black" if abs(pivot_df.iloc[i, j]) < 0.02 else "white",
                                   fontsize=8)
        
        plt.colorbar(im, ax=ax1, label='CAR')
        
        ax2 = axes[1]
        windows = []
        sig_counts = []
        for window in pivot_pval.index:
            pvals = pivot_pval.loc[window].dropna()
            if len(pvals) > 0:
                sig_ratio = (pvals < 0.05).sum() / len(pvals) * 100
                windows.append(window)
                sig_counts.append(sig_ratio)
        
        y_pos = np.arange(len(windows))
        bars = ax2.barh(y_pos, sig_counts, color='steelblue', alpha=0.7)
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(windows)
        ax2.set_xlabel('Significant Days (%)')
        ax2.set_xlim(0, 100)
        ax2.set_title('Significance Ratio')
        ax2.axvline(x=50, color='red', linestyle='--', alpha=0.5)
        
        for i, (bar, count) in enumerate(zip(bars, sig_counts)):
            ax2.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                    f'{count:.1f}%', va='center', fontsize=9)
        
        ax2.invert_yaxis()
        
        plt.tight_layout()
        return fig
    
    def create_dashboard(
        self,
        df_aar: pd.DataFrame,
        car_values: Optional[np.ndarray] = None,
        title: str = 'Event Study Dashboard',
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        创建综合仪表板
        
        Parameters
        ----------
        df_aar : pd.DataFrame
            event_analysis.py 的结果
        car_values : np.ndarray, optional
            各公司 CAR 值分布
        title : str
            仪表板标题
        save_path : str, optional
            保存路径
            
        Returns
        -------
        fig : matplotlib.figure.Figure
            图形对象
        """
        fig = plt.figure(figsize=(16, 10))
        gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.3)
        
        ax1 = fig.add_subplot(gs[0, :])
        first_window = df_aar['event_window_str'].iloc[0]
        subset = df_aar[df_aar['event_window_str'] == first_window].sort_values('relative_day_adj')
        
        ax1.plot(subset['relative_day_adj'], subset['CAAR'], 
                color='#d62728', linewidth=2.5, marker='s', label='CAAR')
        ax1.fill_between(subset['relative_day_adj'], 
                        subset['CAAR'] - subset['Std. E. CAAR']*1.96,
                        subset['CAAR'] + subset['Std. E. CAAR']*1.96,
                        color='gray', alpha=0.2, label='95% CI')
        ax1.axvline(x=0, color='black', linestyle='--', linewidth=1.5)
        ax1.axhline(y=0, color='black', linestyle=':', linewidth=0.8)
        ax1.set_xlabel('Relative Day')
        ax1.set_ylabel('CAAR')
        ax1.set_title('Cumulative Average Abnormal Return', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
        
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.bar(subset['relative_day_adj'], subset['AAR'], 
               color='#ff7f0e', alpha=0.7, width=0.6)
        ax2.axvline(x=0, color='black', linestyle='--', linewidth=1.5)
        ax2.axhline(y=0, color='black', linestyle=':', linewidth=0.8)
        ax2.set_xlabel('Relative Day')
        ax2.set_ylabel('AAR')
        ax2.set_title('Average Abnormal Return')
        ax2.grid(True, alpha=0.3)
        ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
        
        ax3 = fig.add_subplot(gs[1, 1])
        if 'ACAR' in subset.columns:
            ax3.plot(subset['relative_day_adj'], subset['ACAR'], 
                    color='#2ca02c', linewidth=2.5, marker='o', label='ACAR')
            if 'Std. E. ACAR' in subset.columns:
                ax3.fill_between(subset['relative_day_adj'],
                                subset['ACAR'] - subset['Std. E. ACAR']*1.96,
                                subset['ACAR'] + subset['Std. E. ACAR']*1.96,
                                color='gray', alpha=0.2, label='95% CI')
        ax3.axvline(x=0, color='black', linestyle='--', linewidth=1.5)
        ax3.axhline(y=0, color='black', linestyle=':', linewidth=0.8)
        ax3.set_xlabel('Relative Day')
        ax3.set_ylabel('ACAR')
        ax3.set_title('Average CAR')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.xaxis.set_major_locator(MaxNLocator(integer=True))
        
        ax4 = fig.add_subplot(gs[1, 2])
        significance_data = []
        for col in ['AAR_P-value', 'ACAR_P-value', 'CAAR_P-value']:
            if col in subset.columns:
                sig = (subset[col] < 0.05).astype(int)
                significance_data.append(sig.values)
        
        if significance_data:
            sig_array = np.array(significance_data)
            days = subset['relative_day_adj'].values
            im = ax4.imshow(sig_array, aspect='auto', cmap='RdYlGn_r', 
                           vmin=0, vmax=1, extent=[days[0], days[-1], 0, 3])
            ax4.set_yticks([0.5, 1.5, 2.5])
            ax4.set_yticklabels(['AAR', 'ACAR', 'CAAR'][:len(significance_data)])
            ax4.set_xlabel('Relative Day')
            ax4.set_title('Significance (p < 0.05)')
            plt.colorbar(im, ax=ax4, ticks=[0, 1])
        
        if car_values is not None and len(car_values) > 0:
            ax5 = fig.add_subplot(gs[2, :])
            ax5.hist(car_values, bins=30, edgecolor='black', alpha=0.7, density=True)
            ax5.axvline(x=0, color='red', linestyle='--', linewidth=1.5)
            ax5.axvline(x=car_values.mean(), color='green', linestyle='-', linewidth=2,
                       label=f'Mean: {car_values.mean():.4f}')
            ax5.set_xlabel('CAR')
            ax5.set_ylabel('Density')
            ax5.set_title('CAR Distribution Across Companies')
            ax5.legend()
            ax5.grid(True, alpha=0.3)
            
            stats_text = f"""
            N: {len(car_values)}
            Mean: {car_values.mean():.4f}
            Std: {car_values.std():.4f}
            t-stat: {car_values.mean()/(car_values.std()/np.sqrt(len(car_values))):.3f}
            """
            ax5.text(0.98, 0.98, stats_text, transform=ax5.transAxes,
                    fontsize=9, verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
        
        fig.suptitle(title, fontsize=16, fontweight='bold', y=0.995)
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f'Dashboard saved to: {save_path}')
        
        return fig
    
    def save_figure(self, fig: plt.Figure, filename: str, dpi: int = 300, **kwargs):
        """保存图形"""
        fig.savefig(filename, dpi=dpi, bbox_inches='tight', **kwargs)
        print(f'Figure saved: {filename}')


def plot_from_event_analysis(
    df_aar: pd.DataFrame,
    event_window: Optional[str] = None,
    title: Optional[str] = None,
    show_aar: bool = False,
    car_type: str = 'CAAR',
    show_ci: bool = True,
    confidence: float = 0.90,
    figsize: Tuple[float, float] = (9, 6),
    save_path: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
) -> Union[plt.Figure, plt.Axes]:
    """
    从 event_analysis.py 的 df_aar 直接绘图（便捷函数）

    Parameters
    ----------
    df_aar : pd.DataFrame
        event_analysis.py 输出的 df_aar
    event_window : str, optional
        要绘制的事件窗口，None 表示自动选择第一个
    title : str, optional
        图表标题
    show_aar : bool
        是否显示 AAR 柱状图
    car_type : str, optional
        要绘制的 CAR 类型，可选 'CAAR'、'ACAR' 等，默认 'CAAR'
    show_ci : bool
        是否显示置信区间
    confidence : float
        置信水平
    figsize : tuple
        图形大小
    save_path : str, optional
        保存路径
    ax : plt.Axes, optional
        外部传入的坐标轴（用于子图）

    Returns
    -------
    fig or ax : 图形对象
    """
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns
    from scipy.stats import t
    from matplotlib.ticker import MaxNLocator



    # 获取所有可用的 event_window
    available_windows = df_aar['event_window_str'].unique()

    # 自动选择：默认使用第一个事件窗口
    if event_window is None:
        event_window = available_windows[0]
        print(f"自动选择事件窗口: {event_window}")

    if event_window not in available_windows:
        raise ValueError(f"指定的 event_window '{event_window}' 不存在。可用窗口: {list(available_windows)}")

    if title is None:
        title = f'Event Study Results: {car_type} ({event_window})'

    subset = df_aar[df_aar['event_window_str'] == event_window].sort_values('relative_day_adj').copy()

    # 验证 car_type 是否存在
    if car_type not in subset.columns:
        available_cols = [col for col in ['CAAR', 'ACAR'] if col in subset.columns]
        raise ValueError(f"car_type '{car_type}' 不存在于数据中。可用列: {available_cols}")

    # 关键：统一 x 坐标体系
    subset = subset.reset_index(drop=True)
    subset['x_pos'] = range(len(subset))

    # 处理坐标轴
    created_fig = False
    if ax is None:
        fig, ax_plot = plt.subplots(figsize=figsize)
        created_fig = True
    else:
        ax_plot = ax
        fig = ax_plot.figure

    # 折线图：CAAR / ACAR 等
    sns.lineplot(
        data=subset,
        x='x_pos',
        y=car_type,
        sort=False,
        color=line_color,
        linewidth=line_width,
        marker='o',
        ax=ax_plot,
        label=car_type
    )

    # 置信区间
    if show_ci and f'Std. E. {car_type}' in subset.columns and len(subset) > 0:
        df_val = subset['N_firms'].iloc[0] - 1 if 'N_firms' in subset.columns else len(subset) - 1
        delta = subset[f'Std. E. {car_type}'].values * t.ppf((1 + confidence) / 2, max(1, df_val))

        ax_plot.fill_between(
            subset['x_pos'].values,
            subset[car_type].values - delta,
            subset[car_type].values + delta,
            color=ci_color,
            alpha=ci_alpha,
            label=f'{int(confidence * 100)}% CI'
        )

    # 柱状图：AAR
    if show_aar and 'AAR' in subset.columns:
        sns.barplot(
            data=subset,
            x='x_pos',
            y='AAR',
            width=bar_width,
            color=bar_color,
            alpha=bar_alpha,
            ax=ax_plot,
            label='AAR'
        )

    # x轴标签显示真实 relative_day_adj
    ax_plot.set_xticks(subset['x_pos'])
    ax_plot.set_xticklabels(subset['relative_day_adj'].tolist())

    # 标记事件日
    if 0 in subset['relative_day_adj'].values:
        zero_pos = subset.loc[subset['relative_day_adj'] == 0, 'x_pos'].iloc[0]
        ax_plot.axvline(x=zero_pos, color='black', linestyle='--', linewidth=1, alpha=0.7)

    ax_plot.axhline(y=0, color='black', linestyle=':', linewidth=0.8, alpha=0.5)
    
    # ===== 手动控制 y 轴范围，避免 CI 被截断 =====
    y_values = [subset[car_type].values]

    if show_ci and f'Std. E. {car_type}' in subset.columns and len(subset) > 0:
        ci_upper = subset[car_type].values + delta
        ci_lower = subset[car_type].values - delta
        y_values.extend([ci_upper, ci_lower])

    if show_aar and 'AAR' in subset.columns:
        y_values.append(subset['AAR'].values)

    y_all = np.concatenate(y_values)
    y_min, y_max = y_all.min(), y_all.max()

    # 留一点上下边距
    padding = 0.08 * (y_max - y_min) if y_max > y_min else 0.02
    ax_plot.set_ylim(y_min - padding, y_max + padding)
    
    # 标签设置
    if xlabel is None:
        ax_plot.set_xlabel('Relative Day', fontsize=11)
    else:
        ax_plot.set_xlabel(xlabel, fontsize=11)

    if ylabel is not None:
        ax_plot.set_ylabel(ylabel, fontsize=11)
    elif ylabel == '':
        ax_plot.set_ylabel('')
        
    ax_plot.set_title(title, fontsize=13, fontweight='bold')

    # 图例去重
    handles, labels = ax_plot.get_legend_handles_labels()
    seen = set()
    unique_handles = []
    unique_labels = []
    for h, l in zip(handles, labels):
        if l not in seen:
            unique_handles.append(h)
            unique_labels.append(l)
            seen.add(l)

    ax_plot.legend(unique_handles, unique_labels, loc='upper left')

    ax_plot.xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f'Figure saved to: {save_path}')

    if created_fig:
        return fig
    return ax_plot


def plot_company_event(
    df_ar_by_company: pd.DataFrame,
    stockid: Optional[str] = None,
    eventdate: Optional[pd.Timestamp] = None,
    event_window: Optional[str] = None,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    show_ci: bool = True,
    confidence: float = 0.95,
    figsize: Tuple[float, float] = (9, 6),
    save_path: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
) -> None:
    """
    从 df_ar_by_company 中选择特定公司、事件日、事件窗口绘制 AR 和 CAR

    Parameters
    ----------
    df_ar_by_company : pd.DataFrame
        event_analysis.py 输出的 df_ar_by_company
    stockid : str, optional
        股票代码，None 表示自动选择第一个
    eventdate : pd.Timestamp, optional
        事件日期，None 表示自动选择第一个
    event_window : str, optional
        事件窗口（如 '[-2,2]'），None 表示自动选择第一个
    title : str, optional
        图表标题
    xlabel : str, optional
        x轴标签，None 表示显示默认 'Relative Day'，'' 表示隐藏
    ylabel : str, optional
        y轴标签，默认不显示
    show_ci : bool, optional
        是否显示置信区间，默认 True
    confidence : float, optional
        置信水平，默认 0.95
    figsize : tuple
        图形大小
    save_path : str, optional
        保存路径
    ax : plt.Axes, optional
        外部传入的坐标轴（用于子图）

    Returns
    -------
    None
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    from scipy.stats import t
    from matplotlib.ticker import MaxNLocator

    # 复制数据，避免修改原始 DataFrame
    df_ar_by_company = df_ar_by_company.copy()

    # 读取 CSV 后确保 stockid 为字符串（带前导零）
    df_ar_by_company['stockid'] = df_ar_by_company['stockid'].astype(str).str.zfill(6)

    # 转换 eventdate 为统一格式
    df_ar_by_company['eventdate'] = pd.to_datetime(df_ar_by_company['eventdate'])

    available_stockids = df_ar_by_company['stockid'].unique()
    available_windows = df_ar_by_company['event_window_str'].unique()
    available_eventdates = df_ar_by_company['eventdate'].unique()

    if stockid is None:
        stockid = available_stockids[0]
        print(f"自动选择股票: {stockid}")

    if event_window is None:
        event_window = available_windows[0]
        print(f"自动选择事件窗口: {event_window}")

    if eventdate is None:
        eventdate = available_eventdates[0]
        print(f"自动选择事件日: {pd.to_datetime(eventdate).strftime('%Y-%m-%d')}")

    # 用于比较
    eventdate_comp = pd.to_datetime(eventdate)

    subset = df_ar_by_company[
        (df_ar_by_company['stockid'] == stockid) &
        (df_ar_by_company['event_window_str'] == event_window) &
        (df_ar_by_company['eventdate'] == eventdate_comp)
    ].sort_values('relative_day_adj').copy()

    if len(subset) == 0:
        avail_eventdates = df_ar_by_company[
            (df_ar_by_company['stockid'] == stockid) &
            (df_ar_by_company['event_window_str'] == event_window)
        ]['eventdate'].unique()
        raise ValueError(
            f"没有找到匹配的数据: stockid={stockid}, eventdate={eventdate}, event_window={event_window}\n"
            f"该 stockid + event_window 下可用的 eventdate: {avail_eventdates}"
        )

    if title is None:
        title = f'AR and CAR: {stockid} on {eventdate_comp.strftime("%Y-%m-%d")} ({event_window})'

    # 记录是否为函数内部新建图形
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        created_fig = True
    else:
        fig = ax.figure

    # =========================
    # 关键修复：统一 x 坐标体系
    # =========================
    subset = subset.reset_index(drop=True)
    subset['x_pos'] = range(len(subset))

    # 柱状图：AR
    sns.barplot(
        data=subset,
        width=bar_width,
        x='x_pos',
        y='AR',
        ax=ax,
        alpha=bar_alpha,
        label='AR',
        color=bar_color,     # 更论文风
    )

    # 折线图：CAR
    sns.lineplot(
        data=subset,
        x='x_pos',
        y='CAR',
        sort=False,
        marker='o',
        linewidth=line_width,
        ax=ax,
        label='CAR',
        color=line_color,   # 深蓝（推荐）
    )

    # x轴标签显示真实事件日
    ax.set_xticks(subset['x_pos'])
    ax.set_xticklabels(subset['relative_day_adj'].tolist())

    # 置信区间：CAR
    if show_ci and 'Std. E. CAR' in subset.columns and len(subset) > 0:
        df_val = subset['df'].iloc[0] if 'df' in subset.columns else len(subset) - 1
        ci_multiplier = t.ppf((1 + confidence) / 2, max(1, df_val))
        delta = subset['Std. E. CAR'].values * ci_multiplier
        ci_label = f'{int(confidence * 100)}% CI'

        ax.fill_between(
            subset['x_pos'].values,
            subset['CAR'].values - delta,
            subset['CAR'].values + delta,
            alpha=ci_alpha,
            label=ci_label,
            color=ci_color,   # 浅灰
        )

    # 标记事件日和零线
    if 0 in subset['relative_day_adj'].values:
        zero_pos = subset.loc[subset['relative_day_adj'] == 0, 'x_pos'].iloc[0]
        ax.axvline(x=zero_pos, color='black', linestyle='--', linewidth=1, alpha=0.7)

    ax.axhline(y=0, color='black', linestyle=':', linewidth=0.8, alpha=0.5)

    # 标签设置
    if xlabel is None:
        ax.set_xlabel('Relative Day', fontsize=11)
    else:
        ax.set_xlabel(xlabel, fontsize=11)

    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=11)
    elif ylabel == '':
        ax.set_ylabel('')

    ax.set_title(title, fontsize=13, fontweight='bold')

    # 去重 legend
    handles, labels = ax.get_legend_handles_labels()
    seen = set()
    unique_handles = []
    unique_labels = []
    for h, l in zip(handles, labels):
        if l not in seen:
            unique_handles.append(h)
            unique_labels.append(l)
            seen.add(l)

    ax.legend(unique_handles, unique_labels, loc='upper left')

    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f'Figure saved to: {save_path}')
    # 改为：如果传入了外部 ax 则不关闭，否则关闭
    if ax is None:
        plt.close(fig)  # 不显示图形
    return None  # 或者返回 None

def plot_company_comparison(
    df_ar_by_company: pd.DataFrame,
    eventdate: pd.Timestamp,
    event_window_str: str = None,
    stockids: List[str] = None,
    figsize: Tuple[float, float] = (14, 7),
    colors: List[str] = None,
    title: str = None,
    save_path: str = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
) -> plt.Figure:
    """
    将不同公司的 CAR 曲线和 AR 柱状图画在同一图中对比

    Parameters
    ----------
    df_ar_by_company : pd.DataFrame
        包含 AR、CAR 等数据的 DataFrame
    eventdate : pd.Timestamp
        事件日期
    event_window_str : str, optional
        事件窗口字符串，如 "[-2,10]"
    stockids : list, optional
        要对比的股票代码列表，默认使用该事件日下所有股票
    figsize : tuple
        图形大小
    colors : list, optional
        颜色列表
    title : str, optional
        图表标题
    save_path : str, optional
        保存路径，如 "output.png"

    Returns
    -------
    fig : matplotlib.figure.Figure
        图形对象
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    from matplotlib.ticker import MaxNLocator

    df = df_ar_by_company.copy()
    df['eventdate'] = pd.to_datetime(df['eventdate'])
    df['stockid'] = df['stockid'].astype(str).str.zfill(6)

    if event_window_str is not None:
        df = df[df['event_window_str'] == event_window_str]

    df = df[df['eventdate'] == pd.to_datetime(eventdate)]

    if df.empty:
        raise ValueError(f"事件日 {eventdate} 在数据中未找到")

    if stockids is not None:
        stockids = [str(s).zfill(6) for s in stockids]
        df = df[df['stockid'].isin(stockids)]

    if df.empty:
        raise ValueError("筛选后没有可用数据")

    unique_stocks = list(df['stockid'].unique())
    n_stocks = len(unique_stocks)

    # 默认颜色
    if colors is None:
        palette = sns.color_palette("tab10", n_colors=n_stocks)
        colors = palette[:n_stocks]

    if title is None:
        title = (
            f'Comparison of AR/CAR for different companies '
            f'(Event Date: {pd.to_datetime(eventdate).strftime("%Y-%m-%d")})'
        )

    # 相对日 -> 统一坐标位置
    rel_days = sorted(df['relative_day_adj'].unique())
    x_map = {day: i for i, day in enumerate(rel_days)}

    figsize = figsize or (14, 7)
    fig, ax = plt.subplots(figsize=figsize)

    # 每个公司的单个柱宽
    single_bar_width = bar_width / max(n_stocks, 1)

    # 用于后续统一设置 y 轴范围
    y_values = []

    for idx, stockid in enumerate(unique_stocks):
        stock_data = df[df['stockid'] == stockid].sort_values('relative_day_adj').copy()

        # 统一 x 坐标
        stock_data['x_pos'] = stock_data['relative_day_adj'].map(x_map)

        car_values = stock_data['CAR'].values
        ar_values = stock_data['AR'].values
        x_pos = stock_data['x_pos'].values

        # ===== CAR 折线：使用 sns.lineplot =====
        sns.lineplot(
            data=stock_data,
            x='x_pos',
            y='CAR',
            sort=False,
            color=colors[idx],
            linewidth=2,
            marker='o',
            markersize=5,
            ax=ax,
            label=f'{stockid} CAR'
        )

        # ===== AR 柱状图：手动偏移，避免重叠 =====
        offset = (idx - n_stocks / 2 + 0.5) * single_bar_width
        bar_positions = x_pos + offset

        ax.bar(
            bar_positions,
            ar_values,
            width=single_bar_width * 0.9,
            color=colors[idx],
            alpha=bar_alpha,
            label=f'{stockid} AR',
            edgecolor='black',
            linewidth=0.3
        )

        y_values.append(car_values)
        y_values.append(ar_values)

    # 事件日与零线
    if 0 in x_map:
        ax.axvline(x=x_map[0], color='black', linestyle='--', linewidth=1, alpha=0.7)
    ax.axhline(y=0, color='black', linestyle=':', linewidth=0.8, alpha=0.5)

    # x轴标签显示真实 relative day
    ax.set_xticks(range(len(rel_days)))
    ax.set_xticklabels(rel_days)

    # 标题与标签
    ax.set_title(title, fontsize=13, fontweight='bold')
    # 标签设置
    if xlabel is None:
        ax.set_xlabel('Relative Day', fontsize=11)
    else:
        ax.set_xlabel(xlabel, fontsize=11)

    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=11)
    elif ylabel == '':
        ax.set_ylabel('')
    # 去重 legend
    handles, labels = ax.get_legend_handles_labels()
    seen = set()
    unique_handles = []
    unique_labels = []
    for h, l in zip(handles, labels):
        if l not in seen:
            unique_handles.append(h)
            unique_labels.append(l)
            seen.add(l)

    ax.legend(unique_handles, unique_labels, loc='best', frameon=True, ncol=2)

    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    # 手动设置 y 轴范围，避免贴边
    if y_values:
        y_all = np.concatenate(y_values)
        y_min, y_max = y_all.min(), y_all.max()
        padding = 0.08 * (y_max - y_min) if y_max > y_min else 0.02
        ax.set_ylim(y_min - padding, y_max + padding)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig

def list_available_selections(df_ar_by_company: pd.DataFrame) -> None:
    """列出 df_ar_by_company 中所有可用的选择"""
    # 确保 eventdate 列是 datetime 类型，避免比较问题
    df_ar_by_company = df_ar_by_company.copy()
    df_ar_by_company['eventdate'] = pd.to_datetime(df_ar_by_company['eventdate'])
    print("=" * 60)
    print("df_ar_by_company 中的可用选择：")
    print("=" * 60)
    print(f"\n股票数量: {df_ar_by_company['stockid'].nunique()}")
    print(f"事件窗口: {df_ar_by_company['event_window_str'].unique().tolist()}")
    print(f"事件日数量: {df_ar_by_company['eventdate'].nunique()}")
    print("\n示例:")
    sample = df_ar_by_company[['stockid', 'eventdate', 'event_window_str']].drop_duplicates().head(10)
    print(sample.to_string(index=False))
    print("\n用法: plot_company_event(df_ar_by_company, stockid='000001', eventdate=timestamp, event_window='[-2,2]')")