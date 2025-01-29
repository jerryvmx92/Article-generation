"""Trace logging for article generation."""

import os
import json
from datetime import datetime
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class TraceLogger:
    """Logger for recording article generation traces."""
    
    def __init__(self, trace_dir: Optional[str] = None):
        """Initialize the trace logger.
        
        Args:
            trace_dir: Directory to store traces. Defaults to './traces'
        """
        self.trace_dir = trace_dir or os.path.join(os.getcwd(), "traces")
        os.makedirs(self.trace_dir, exist_ok=True)
        
        # Create subdirectories
        self.success_dir = os.path.join(self.trace_dir, "success")
        self.error_dir = os.path.join(self.trace_dir, "error")
        os.makedirs(self.success_dir, exist_ok=True)
        os.makedirs(self.error_dir, exist_ok=True)
    
    def log_trace(
        self,
        title: str,
        keywords: list[str],
        prompt: str,
        response: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
        error: Optional[Exception] = None
    ) -> str:
        """Log a trace of the generation process.
        
        Args:
            title: The article title
            keywords: List of keywords used
            prompt: The prompt sent to the LLM
            response: The LLM's response
            metadata: Additional metadata about the generation
            error: Any error that occurred
            
        Returns:
            The path to the saved trace file
        """
        trace = {
            "timestamp": datetime.now().isoformat(),
            "title": title,
            "keywords": keywords,
            "prompt": prompt,
            "response": response,
            "metadata": metadata or {},
            "error": str(error) if error else None
        }
        
        # Create filename from sanitized title and timestamp
        safe_title = "".join(c if c.isalnum() else "_" for c in title.lower())
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{safe_title}_{timestamp}.json"
        
        # Save to appropriate directory
        save_dir = self.error_dir if error else self.success_dir
        save_path = os.path.join(save_dir, filename)
        
        try:
            with open(save_path, "w") as f:
                json.dump(trace, f, indent=2)
            logger.info(f"Saved trace to {save_path}")
            return save_path
        except Exception as e:
            logger.error(f"Failed to save trace: {e}")
            raise
    
    def get_traces(
        self,
        success_only: bool = True,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> list[Dict[str, Any]]:
        """Get traces matching the specified criteria.
        
        Args:
            success_only: Only return successful traces
            start_date: Only return traces after this date
            end_date: Only return traces before this date
            
        Returns:
            List of trace dictionaries
        """
        traces = []
        dirs = [self.success_dir]
        if not success_only:
            dirs.append(self.error_dir)
            
        for directory in dirs:
            for filename in os.listdir(directory):
                if not filename.endswith(".json"):
                    continue
                    
                filepath = os.path.join(directory, filename)
                try:
                    with open(filepath) as f:
                        trace = json.load(f)
                        
                    trace_date = datetime.fromisoformat(trace["timestamp"])
                    
                    if start_date and trace_date < start_date:
                        continue
                    if end_date and trace_date > end_date:
                        continue
                        
                    traces.append(trace)
                except Exception as e:
                    logger.error(f"Failed to load trace {filepath}: {e}")
                    
        return sorted(traces, key=lambda x: x["timestamp"], reverse=True) 