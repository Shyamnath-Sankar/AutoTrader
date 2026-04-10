"""
scheduler.py — APScheduler management for the Smart Money Trading Bot.

Handles:
  - Creating one-shot jobs when the AI says WATCH (schedule for later)
  - Managing the main scan loop
  - Ensuring jobs don't stack or conflict
"""

from datetime import datetime, timedelta

import pytz
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.date import DateTrigger
from loguru import logger

from config import settings


class BotScheduler:
    """Manages APScheduler for the trading bot."""

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
    # SCHEDULE A ONE-SHOT ANALYSIS
    # ═══════════════════════════════════════════════════════════════════════

    def schedule_analysis(
        self,
        minutes_from_now: int,
        reason: str = "",
        job_id: str = "ai_scheduled_analysis",
    ):
        """
        Schedule a one-shot analysis job at a specific time in the future.
        Used when the AI returns WATCH with next_check_minutes.

        If a job with the same ID already exists, it is replaced.
        """
        if not self._analysis_callback:
            logger.error("Cannot schedule — no analysis callback set")
            return

        # Validate minutes
        if minutes_from_now <= 0:
            minutes_from_now = settings.DEFAULT_SCAN_INTERVAL_MINUTES
        if minutes_from_now > settings.MAX_SCHEDULE_MINUTES:
            minutes_from_now = settings.MAX_SCHEDULE_MINUTES
            logger.warning(f"Schedule capped to {settings.MAX_SCHEDULE_MINUTES} minutes")

        run_time = datetime.now(pytz.UTC) + timedelta(minutes=minutes_from_now)

        # Remove existing job with same ID if present
        try:
            self.scheduler.remove_job(job_id)
        except Exception:
            pass

        self.scheduler.add_job(
            self._analysis_callback,
            trigger=DateTrigger(run_date=run_time),
            id=job_id,
            name=f"AI Analysis ({reason[:50]})" if reason else "AI Scheduled Analysis",
            replace_existing=True,
            misfire_grace_time=60,  # allow 60 seconds late
        )

        logger.info(
            f"⏰ Scheduled analysis in {minutes_from_now} min "
            f"(at {run_time.strftime('%H:%M:%S')} UTC)"
        )
        if reason:
            logger.info(f"   Reason: {reason}")

    def schedule_default_scan(self, job_id: str = "default_scan"):
        """Schedule the next scan at the default interval."""
        self.schedule_analysis(
            minutes_from_now=settings.DEFAULT_SCAN_INTERVAL_MINUTES,
            reason="Default scan interval",
            job_id=job_id,
        )

    def schedule_gate_retry(self, skip_minutes: int, reason: str = ""):
        """Schedule a retry after a gate failure."""
        self.schedule_analysis(
            minutes_from_now=skip_minutes,
            reason=f"Gate retry: {reason}",
            job_id="gate_retry",
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
