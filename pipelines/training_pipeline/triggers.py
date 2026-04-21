"""
Pipeline triggers and scheduling
"""

from prefect.schedules import Schedule
from prefect.schedules.intervals import IntervalSchedule
from prefect.schedules.cron import CronSchedule
from prefect.schedules.filters import on_days, at_time
from datetime import timedelta, time
from typing import Optional

# Daily schedule at 2 AM
daily_schedule = Schedule(
    cron="0 2 * * *",
    timezone="UTC"
)

# Weekly schedule on Monday at 3 AM
weekly_schedule = Schedule(
    cron="0 3 * * 1",
    timezone="UTC"
)

# Monthly schedule on 1st at 4 AM
monthly_schedule = Schedule(
    cron="0 4 1 * *",
    timezone="UTC"
)

# Interval schedule (every 6 hours)
interval_schedule = IntervalSchedule(
    interval=timedelta(hours=6)
)

# Weekday schedule (Monday to Friday, 9 AM)
weekday_schedule = Schedule(
    cron="0 9 * * 1-5",
    timezone="UTC"
)

# Event-based triggers (to be used with webhooks)
class EventTrigger:
    """Event-based trigger for pipeline"""
    
    def __init__(self, event_type: str):
        self.event_type = event_type
    
    def matches(self, event: dict) -> bool:
        """Check if event matches trigger"""
        return event.get('type') == self.event_type

# Data drift trigger
class DataDriftTrigger:
    """Trigger when data drift is detected"""
    
    def __init__(self, drift_threshold: float = 0.2):
        self.drift_threshold = drift_threshold
    
    def check_drift(self, drift_score: float) -> bool:
        """Check if drift exceeds threshold"""
        return drift_score > self.drift_threshold

# Performance degradation trigger
class PerformanceTrigger:
    """Trigger when model performance degrades"""
    
    def __init__(self, mae_threshold: float = 50000):
        self.mae_threshold = mae_threshold
    
    def check_performance(self, current_mae: float) -> bool:
        """Check if performance degraded"""
        return current_mae > self.mae_threshold

# New data trigger
class NewDataTrigger:
    """Trigger when new data is available"""
    
    def __init__(self, data_source: str, check_interval: int = 3600):
        self.data_source = data_source
        self.check_interval = check_interval
        self.last_check = None
    
    def check_new_data(self) -> bool:
        """Check if new data is available"""
        # Implement logic to check for new data
        return False

# Composite trigger
class CompositeTrigger:
    """Combine multiple triggers"""
    
    def __init__(self, triggers: list, operator: str = 'AND'):
        self.triggers = triggers
        self.operator = operator
    
    def should_run(self, context: dict) -> bool:
        """Check if any/all triggers are satisfied"""
        if self.operator == 'AND':
            return all(t.should_run(context) for t in self.triggers)
        elif self.operator == 'OR':
            return any(t.should_run(context) for t in self.triggers)
        return False

# Schedule factory
def create_schedule(
    frequency: str,
    time_str: Optional[str] = None,
    day_of_week: Optional[str] = None
) -> Schedule:
    """Create schedule based on parameters"""
    
    if frequency == 'daily' and time_str:
        hour, minute = map(int, time_str.split(':'))
        return Schedule(cron=f"{minute} {hour} * * *")
    
    elif frequency == 'weekly' and day_of_week and time_str:
        hour, minute = map(int, time_str.split(':'))
        day_map = {'mon': 1, 'tue': 2, 'wed': 3, 'thu': 4, 'fri': 5, 'sat': 6, 'sun': 0}
        return Schedule(cron=f"{minute} {hour} * * {day_map[day_of_week.lower()[:3]]}")
    
    elif frequency == 'hourly':
        return IntervalSchedule(interval=timedelta(hours=1))
    
    elif frequency == 'realtime':
        return IntervalSchedule(interval=timedelta(minutes=5))
    
    else:
        return IntervalSchedule(interval=timedelta(days=1))