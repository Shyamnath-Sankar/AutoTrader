"""
scheduler.py — APScheduler management for the Smart Money Trading Bot.

Handles:
  - Scheduling the next analysis scan at a fixed interval
  - Managing gate retry and rejection cooldown timers
  - Ensuring jobs don't stack or conflict

Key fix: ALL analysis jobs use the SAME job ID ("analysis_job") to prevent
concurrent analysis cycles from overlapping.
"""

from datetime import datetime, timedelta

import pytz
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.date import DateTrigger
from loguru import logger

from config import settings


class BotScheduler:
    """Manages APScheduler for the trading bot."""

    # Single job ID for ALL analysis scheduling — prevents stacking
    ANALYSIS_JOB_ID = "analysis_job"

    def __init__(self):
        self.scheduler = BackgroundScheduler(timezone=pytz.UTC)
        self._analysis_callback = None
        self._running = False

    def start(self, analysis_callback):
        """
        Start the scheduler.

        Args:
            analysis_callback: the function to call when a scheduled analysis fires.
                               This should be the main analysis pipeline function.
        """
        self._analysis_callback = analysis_callback
        self.scheduler.start()
        self._running = True
        logger.info("⏰ Scheduler started (UTC timezone)")

    def stop(self):
        """Gracefully shut down the scheduler."""
        if self._running:
            self.scheduler.shutdown(wait=False)
            self._running = False
            logger.info("⏰ Scheduler stopped")

    # ═══════════════════════════════════════════════════════════════════════
    # SCHEDULE A ONE-SHOT ANALYSIS — all use same job ID
    # ═══════════════════════════════════════════════════════════════════════

    def schedule_analysis(
        self,
        minutes_from_now: int,
        reason: str = "",
    ):
        """
        Schedule a one-shot analysis job at a specific time in the future.
        Used for gate retries, rejection cooldowns, and default scans.

        ALL analysis jobs use the same job ID to prevent stacking.
        If a job already exists, it is replaced (not duplicated).
        """
        if not self._analysis_callback:
            logger.error("Cannot schedule — no analysis callback set")
            return

        # Validate minutes
        if minutes_from_now <= 0:
            minutes_from_now = settings.DEFAULT_SCAN_INTERVAL_MINUTES

        run_time = datetime.now(pytz.UTC) + timedelta(minutes=minutes_from_now)

        # Remove existing job if present (prevent stacking)
        try:
            existing = self.scheduler.get_job(self.ANALYSIS_JOB_ID)
            if existing and existing.next_run_time:
                old_time = existing.next_run_time.strftime('%H:%M:%S')
                new_time = run_time.strftime('%H:%M:%S')
                logger.info(f"⏰ Replacing existing job (was {old_time} UTC) → now {new_time} UTC")
            self.scheduler.remove_job(self.ANALYSIS_JOB_ID)
        except Exception:
            pass

        self.scheduler.add_job(
            self._analysis_callback,
            trigger=DateTrigger(run_date=run_time),
            id=self.ANALYSIS_JOB_ID,
            name=f"Analysis ({reason[:50]})" if reason else "Scheduled Analysis",
            replace_existing=True,
            misfire_grace_time=120,  # allow 2 minutes late
        )

        logger.info(
            f"⏰ Next analysis in {minutes_from_now} min "
            f"(at {run_time.strftime('%H:%M:%S')} UTC)"
        )
        if reason:
            logger.info(f"   Reason: {reason}")

    def schedule_default_scan(self):
        """Schedule the next scan at the default interval."""
        self.schedule_analysis(
            minutes_from_now=settings.DEFAULT_SCAN_INTERVAL_MINUTES,
            reason="Default scan interval",
        )

    def schedule_gate_retry(self, skip_minutes: int, reason: str = ""):
        """Schedule a retry after a gate failure."""
        self.schedule_analysis(
            minutes_from_now=skip_minutes,
            reason=f"Gate retry: {reason}",
        )

    def schedule_cooldown(self, cooldown_minutes: int, reason: str = ""):
        """Schedule after a risk rejection with cooldown."""
        self.schedule_analysis(
            minutes_from_now=cooldown_minutes,
            reason=f"Rejection cooldown: {reason}",
        )

    # ═══════════════════════════════════════════════════════════════════════
    # UTILITIES
    # ═══════════════════════════════════════════════════════════════════════

    def get_pending_jobs(self) -> list[dict]:
        """Get a list of all pending scheduled jobs."""
        jobs = self.scheduler.get_jobs()
        return [
            {
                "id": job.id,
                "name": job.name,
                "next_run": job.next_run_time.isoformat() if job.next_run_time else None,
            }
            for job in jobs
        ]

    def cancel_all(self):
        """Cancel all pending jobs."""
        self.scheduler.remove_all_jobs()
        logger.info("⏰ All scheduled jobs cancelled")

    @property
    def is_running(self) -> bool:
        return self._running
