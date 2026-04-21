"""
event_study: 事件研究分析工具包
"""

from .event_analysis import event_study, mark_event_window
from .event_plot import (
    EventStudyPlotter,
    plot_from_event_analysis,
    plot_company_event,
    plot_company_comparison,
    list_available_selections,
)

__all__ = [
    'event_study',
    'mark_event_window',
    'EventStudyPlotter',
    'plot_from_event_analysis',
    'plot_company_event',
    'plot_company_comparison',
    'list_available_selections',
]