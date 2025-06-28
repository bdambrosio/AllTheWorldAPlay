import asyncio
import enum
from typing import Dict, Optional, Callable, Any
import logging

logger = logging.getLogger(__name__)

class ReplayState(enum.Enum):
    IDLE = "idle"
    PROCESSING = "processing"
    WAITING_SPEECH = "waiting_speech"
    PAUSED = "paused"
    ERROR = "error"

class ReplayStateManager:
    def __init__(self, session_id: str, websocket_send_func: Callable):
        self.session_id = session_id
        self.websocket_send = websocket_send_func
        
        self._state = ReplayState.IDLE
        self._current_operation = None
        self._replay_task: Optional[asyncio.Task] = None
        
        self._speech_complete_event = asyncio.Event()
        self._speech_timeout = 30
        
        self._event_delays = {
            'replay_event': 0.5,
            'show_update': 1.0,
            'world_update': 0.5,
            'character_update': 0.4,
            'command_ack': 0.1,
            'default': 0.2
        }
        
        self._state_change_callbacks = []
        
        self._state_lock = asyncio.Lock()
        
    @property
    def state(self) -> ReplayState:
        return self._state
        
    @property
    def is_processing(self) -> bool:
        return self._state in [ReplayState.PROCESSING, ReplayState.WAITING_SPEECH]
        
    @property
    def can_accept_commands(self) -> bool:
        return self._state in [ReplayState.IDLE, ReplayState.PAUSED]
        
    def add_state_change_callback(self, callback: Callable[[ReplayState, ReplayState], None]):
        self._state_change_callbacks.append(callback)
        
    async def _transition_state(self, new_state: ReplayState, operation: str = None):
        async with self._state_lock:
            old_state = self._state
            self._state = new_state
            self._current_operation = operation
            
            logger.info(f"State transition: {old_state.value} -> {new_state.value} (operation: {operation})")
            
            await self.websocket_send({
                "type": "state_update",
                "state": new_state.value,
                "operation": operation,
                "can_accept_commands": self.can_accept_commands,
                "is_processing": self.is_processing
            })
            
            for callback in self._state_change_callbacks:
                try:
                    callback(old_state, new_state)
                except Exception as e:
                    logger.error(f"Error in state change callback: {e}")
                    
    async def start_operation(self, operation: str) -> bool:
        if not self.can_accept_commands:
            logger.warning(f"Cannot start operation '{operation}' in state {self._state.value}")
            return False
            
        await self._transition_state(ReplayState.PROCESSING, operation)
        return True
        
    async def complete_operation(self, operation: str):
        if self._current_operation != operation:
            logger.warning(f"Operation mismatch: expected {self._current_operation}, got {operation}")
            
        await self._transition_state(ReplayState.IDLE)
        await self._send_command_ack(operation)
        
    async def pause_operation(self):
        if self._replay_task and not self._replay_task.done():
            self._replay_task.cancel()
            try:
                await self._replay_task
            except asyncio.CancelledError:
                pass
            self._replay_task = None
            
        await self._transition_state(ReplayState.PAUSED)
        await self._send_command_ack("pause")
        
    async def resume_from_pause(self, operation: str):
        if self._state != ReplayState.PAUSED:
            logger.warning(f"Cannot resume from state {self._state.value}")
            return False
            
        await self._transition_state(ReplayState.PROCESSING, operation)
        return True
        
    async def enter_error_state(self, error_msg: str):
        await self._transition_state(ReplayState.ERROR)
        await self.websocket_send({
            "type": "error",
            "message": error_msg
        })
        
    async def recover_from_error(self):
        await self._transition_state(ReplayState.IDLE)
        
    async def start_speech_wait(self):
        if self._state != ReplayState.PROCESSING:
            logger.warning(f"Cannot start speech wait from state {self._state.value}")
            return
            
        await self._transition_state(ReplayState.WAITING_SPEECH)
        self._speech_complete_event.clear()
        
    async def wait_for_speech_complete(self) -> bool:
        if self._state != ReplayState.WAITING_SPEECH:
            return True
            
        try:
            await asyncio.wait_for(
                self._speech_complete_event.wait(), 
                timeout=self._speech_timeout
            )
            return True
        except asyncio.TimeoutError:
            logger.warning("Speech timeout occurred")
            await self.websocket_send({
                "type": "speech_timeout",
                "message": "Audio playback timed out"
            })
            return False
        except asyncio.CancelledError:
            return False
            
    async def signal_speech_complete(self):
        self._speech_complete_event.set()
        if self._state == ReplayState.WAITING_SPEECH:
            await self._transition_state(ReplayState.PROCESSING)
            
    async def safe_task_transition(self, new_task_coro):
        if self._replay_task and not self._replay_task.done():
            self._replay_task.cancel()
            try:
                await self._replay_task
            except asyncio.CancelledError:
                pass
                
        self._replay_task = asyncio.create_task(new_task_coro)
        return self._replay_task
        
    async def get_event_delay(self, event_type: str) -> float:
        return self._event_delays.get(event_type, self._event_delays['default'])
        
    async def _send_command_ack(self, command: str):
        await self.websocket_send({
            "type": "command_ack",
            "command": command
        })
        
    def cleanup(self):
        if self._replay_task and not self._replay_task.done():
            self._replay_task.cancel()
