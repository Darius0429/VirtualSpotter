import os
import csv
import time
import logging
import threading
from datetime import datetime
from typing import Dict, Any, List, Optional
import h5py
import numpy as np
from ._base import DataModule
from ..module_info import minfo
from ..api_control import api
from .. import validator as val

logger = logging.getLogger(__name__)

class BinaryTelemetryLogger:
    """Binary HDF5 logger for telemetry data"""
    
    def __init__(self, filepath: str, filename: str, group_name: str = "data"):
        self.filepath = filepath
        self.filename = filename
        self.full_path = os.path.join(filepath, filename)
        self.group_name = group_name
        self.file = None
        self.buffer = {}
        self.buffer_size = 100  # Number of samples before writing to disk
        self.row_count = 0
        os.makedirs(filepath, exist_ok=True)
        
    def open(self):
        """Open the HDF5 file for writing"""
        if self.file is None:
            self.file = h5py.File(self.full_path, 'a')
            # Create group if it doesn't exist
            if self.group_name not in self.file:
                self.file.create_group(self.group_name)
        return self.file
    
    def write_dict_row(self, data: Dict[str, Any]):
        """Add a dictionary row to the buffer"""
        # Initialize buffer arrays if needed
        for key, value in data.items():
            if key not in self.buffer:
                self.buffer[key] = []
            
            # Add value to buffer
            self.buffer[key].append(value)
        
        self.row_count += 1
        
        # Flush buffer if it gets too large
        if self.row_count >= self.buffer_size:
            self.flush_buffer()
    
    def flush_buffer(self):
        """Write buffered data to HDF5 file"""
        if not self.buffer or self.row_count == 0:
            return
            
        file = self.open()
        group = file[self.group_name]
        
        # Process each column
        for key, values in self.buffer.items():
            if len(values) == 0:
                continue
                
            # Convert to numpy array with appropriate type
            try:
                if isinstance(values[0], str):
                    # For string data
                    data_array = np.array(values, dtype=h5py.special_dtype(vlen=str))
                else:
                    # For numeric data
                    data_array = np.array(values)
            except Exception:
                # Fallback for mixed data
                data_array = np.array(values, dtype=h5py.special_dtype(vlen=str))
            
            # Create or extend dataset
            if key in group:
                # Get current size
                current_size = group[key].shape[0]
                # Resize
                group[key].resize((current_size + len(data_array),))
                # Add new data
                group[key][current_size:] = data_array
            else:
                # Create new dataset with unlimited size
                group.create_dataset(
                    key, 
                    data=data_array,
                    maxshape=(None,),  # Unlimited length
                    chunks=True,       # Enable chunking
                    compression="gzip", 
                    compression_opts=4  # Balance between speed and compression
                )
        
        # Clear buffer
        self.buffer = {}
        self.row_count = 0
        
        # Ensure data is written to disk
        file.flush()
    
    def close(self):
        """Close the HDF5 file"""
        if self.file:
            self.flush_buffer()
            self.file.close()
            self.file = None

class Realtime(DataModule):
    """Simplified Data Logger for TinyPedal"""

    def __init__(self, config, module_name):
        super().__init__(config, module_name)
        
        # Configure logger settings
        self.log_directory = self.mcfg.get("log_directory", "logs/telemetry")
        self.compress_logs = False  # Explicitly disable compression
        
        # Configure data collection intervals
        self.log_high_freq_interval = max(self.mcfg.get("log_high_freq_interval", 0.01), 0.001)
        self.log_static_interval = max(self.mcfg.get("log_static_interval", 5.0), 10.0)
        self.log_vehicles_interval = max(self.mcfg.get("log_vehicles_interval", 0.25), 0.2)
        
        # Create logging directory
        self.log_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), 
            self.log_directory
        )
        os.makedirs(self.log_path, exist_ok=True)
        
        # Session tracking
        self.last_session_id = ("",-1,-1,-1)
        self.current_session_info = {}
        
        # Writers
        self.high_freq_writer = None
        self.static_data_writer = None
        self.vehicles_data_writer = None
        
        # Threading
        self._high_freq_thread = None
        self._static_data_thread = None
        self._vehicles_data_thread = None
        self._stop_logging = threading.Event()
        
        # Timers
        self.last_high_freq_time = 0
        self.last_static_data_time = 0
        self.last_vehicles_data_time = 0
        
        # Lap completed tracking
        self.last_lap_number = 0
        
    def update_data(self):
        """Update module data"""
        reset = False
        update_interval = self.active_interval

        while not self._event.wait(update_interval):
            if self.state.active:
                if not reset:
                    reset = True
                    update_interval = self.active_interval
                    logger.info("Telemetry Logger module activated")
                    
                    # Reset session tracking to force creation of new log files
                    self.last_session_id = ("",-1,-1,-1)
                    
                    # Start data collection threads
                    self._stop_logging.clear()
                    
                    self._high_freq_thread = threading.Thread(
                        target=self._run_high_freq_logging, 
                        daemon=True,
                        name="HighFrequencyLogger"
                    )
                    self._high_freq_thread.start()
                    
                    self._static_data_thread = threading.Thread(
                        target=self._run_static_data_logging, 
                        daemon=True,
                        name="StaticDataLogger"
                    )
                    self._static_data_thread.start()
                    
                    self._vehicles_data_thread = threading.Thread(
                        target=self._run_vehicles_data_logging, 
                        daemon=True,
                        name="VehiclesDataLogger"
                    )
                    self._vehicles_data_thread.start()
                    
                    logger.info("Telemetry logging threads started")
                    
            else:
                if reset:
                    reset = False
                    update_interval = self.idle_interval
                    
                    # Stop logging threads
                    self._stop_logging.set()
                    
                    # Wait for threads to finish
                    if self._high_freq_thread:
                        self._high_freq_thread.join(timeout=2.0)
                    if self._static_data_thread:
                        self._static_data_thread.join(timeout=2.0)
                    if self._vehicles_data_thread:
                        self._vehicles_data_thread.join(timeout=2.0)
                        
                    self._high_freq_thread = None
                    self._static_data_thread = None
                    self._vehicles_data_thread = None
                    
                    # Close any open writers
                    self._close_writers()
                    logger.info("Telemetry logging threads stopped")

    def _run_high_freq_logging(self):
        """Thread for logging high-frequency vehicle telemetry data"""
        logger.info("Starting high-frequency logging thread")
        self.last_high_freq_time = time.time()
        
        try:
            while not self._stop_logging.is_set():
                current_time = time.time()
                
                # Check if we should log based on interval
                if current_time - self.last_high_freq_time >= self.log_high_freq_interval:
                    # Check if we're in a valid session
                    if api is not None and api.state and self._should_log_data():
                        # Log high-frequency data
                        self._log_high_frequency_data()
                        self.last_high_freq_time = current_time
                        
                        # Check for lap completion
                        self._check_lap_completion()
                
                # Sleep just enough to maintain desired frequency
                sleep_time = max(0.001, self.log_high_freq_interval - (time.time() - current_time))
                time.sleep(sleep_time)
                
        except Exception as e:
            logger.error(f"Error in high-frequency logging thread: {str(e)}", exc_info=True)

    def _run_static_data_logging(self):
        """Thread for logging static session data"""
        logger.info("Starting static data logging thread")
        self.last_static_data_time = time.time()
        
        try:
            while not self._stop_logging.is_set():
                current_time = time.time()
                
                # Check if we should log based on interval
                if current_time - self.last_static_data_time >= self.log_static_interval:
                    # Check if we're in a valid session
                    if api is not None and api.state and self._should_log_data():
                        # Log static data
                        self._log_static_data()
                        self.last_static_data_time = current_time
                
                # Sleep to maintain desired frequency (less critical for static data)
                time.sleep(1.0)
                
        except Exception as e:
            logger.error(f"Error in static data logging thread: {str(e)}", exc_info=True)

    def _run_vehicles_data_logging(self):
        """Thread for logging multi-vehicle dataset"""
        logger.info("Starting vehicles data logging thread")
        self.last_vehicles_data_time = time.time()
        
        try:
            while not self._stop_logging.is_set():
                current_time = time.time()
                
                # Check if we should log based on interval
                if current_time - self.last_vehicles_data_time >= self.log_vehicles_interval:
                    # Check if we're in a valid session
                    if api is not None and api.state and self._should_log_data():
                        # Log vehicles data
                        self._log_vehicles_data()
                        self.last_vehicles_data_time = current_time
                
                # Sleep to maintain desired frequency
                sleep_time = max(0.01, self.log_vehicles_interval - (time.time() - current_time))
                time.sleep(sleep_time)
                
        except Exception as e:
            logger.error(f"Error in vehicles data logging thread: {str(e)}", exc_info=True)

    def _should_log_data(self) -> bool:
        """Check if we should be logging data (valid session)"""
        if not api:
            return False
        
        combo_id = api.read.check.combo_id()
        session_id = api.read.check.session_id()
        session_type = api.read.session.session_type()
        
        # Skip if not in a valid session
        if session_type == 0:  # Idle
            return False
            
        # Check if we've changed session
        if not val.same_session(combo_id, session_id, self.last_session_id):
            # Finalize any previous session
            self._close_writers()
            self._open_new_writers(combo_id, session_id, session_type)
            self.last_session_id = (combo_id, *session_id)
            
            # Store session info for correlation between files
            self.current_session_info = {
                "session_id": session_id,
                "combo_id": combo_id,
                "session_type": session_type,
                "track_name": api.read.check.track_id() if api else "unknown_track",
                "vehicle_name": api.read.check.vehicle_id() if api else "unknown_vehicle",
                "start_time": time.time()
            }
            
            # Reset lap tracking
            self.last_lap_number = 0
            
            logger.info(f"Started logging for new session: {self._get_session_type_name(session_type)}")
            
        return True

    def _check_lap_completion(self):
        """Check if a lap has been completed"""
        if not api:
            return
            
        try:
            current_lap = api.read.lap.number()
            if current_lap > self.last_lap_number:
                # Lap completed, log info
                lap_data = {
                    "lap_number": current_lap,
                    "timestamp": time.time(),
                    "session_time": api.read.session.elapsed(),
                    "lap_time": api.read.timing.last_laptime()
                }
                
                # Log lap completion marker
                if hasattr(self, "lap_marker_writer") and self.lap_marker_writer:
                    self.lap_marker_writer.write_dict_row(lap_data)
                
                logger.info(f"Lap {current_lap-1} completed in {lap_data['lap_time']:.3f}s")
                self.last_lap_number = current_lap
        except Exception as e:
            logger.debug(f"Error checking lap completion: {str(e)}")

    def _log_high_frequency_data(self):
        """Log high-frequency vehicle telemetry data"""
        if not api or not self.high_freq_writer:
            return
        
        try:
            # Build timestamp and core data
            timestamp = time.time()
            session_time = api.read.session.elapsed()
            
            data = {
                "timestamp": timestamp,
                "session_time": session_time,
                "lap_number": api.read.lap.number(),
                "lap_distance": api.read.lap.distance(),
                "lap_progress": api.read.lap.progress(),
                "sector": api.read.lap.sector_index()
            }
            
            # Add vehicle physics data
            position_xyz = api.read.vehicle.position_xyz()
            data.update({
                "speed": api.read.vehicle.speed(),
                "position_x": position_xyz[0],
                "position_y": position_xyz[1],
                "position_z": position_xyz[2],
                "orientation_yaw": api.read.vehicle.orientation_yaw_radians(),
                "accel_lat": api.read.vehicle.accel_lateral(),
                "accel_long": api.read.vehicle.accel_longitudinal(),
                "accel_vert": api.read.vehicle.accel_vertical()
            })
            
            # Add input data
            data.update({
                "throttle": api.read.inputs.throttle(),
                "brake": api.read.inputs.brake(),
                "clutch": api.read.inputs.clutch(),
                "steering": api.read.inputs.steering()
            })
            
            # Add engine data
            data.update({
                "rpm": api.read.engine.rpm(),
                "gear": api.read.engine.gear(),
                "fuel": api.read.vehicle.fuel()
            })
                
            # Add tire temperatures if available
            if hasattr(api.read, "tyre"):
                tire_temps = api.read.tyre.surface_temperature_avg()
                data.update({
                    "temp_tire_fl": tire_temps[0],
                    "temp_tire_fr": tire_temps[1],
                    "temp_tire_rl": tire_temps[2],
                    "temp_tire_rr": tire_temps[3]
                })
            
            # Add to CSV
            self.high_freq_writer.write_dict_row(data)
            
        except Exception as e:
            logger.error(f"Error logging high-frequency data: {str(e)}", exc_info=True)

    def _log_static_data(self):
        """Log static session data that changes infrequently"""
        if not api or not self.static_data_writer:
            return
        
        try:
            # Build timestamp and core data
            timestamp = time.time()
            session_time = api.read.session.elapsed()
            
            data = {
                "timestamp": f"{timestamp:.3f}",
                "session_time": f"{session_time:.3f}",
                "session_id": str(self.current_session_info.get("session_id", "")),
                "session_type": api.read.session.session_type()
            }
            
            # Add track information
            data.update({
                "track_name": api.read.check.track_id(),
                "track_length": api.read.lap.track_length()
            })
            
            # Add vehicle information
            data.update({
                "vehicle_name": api.read.vehicle.vehicle_name(),
                "vehicle_class": api.read.vehicle.class_name(),
                "driver_name": api.read.vehicle.driver_name()
            })
            
            # Add session information
            data.update({
                "session_type_name": self._get_session_type_name(api.read.session.session_type()),
                "session_remaining": f"{api.read.session.remaining():.2f}" if hasattr(api.read.session, "remaining") else 0,
                "total_vehicles": api.read.vehicle.total_vehicles()
            })
            
            # Add weather information if available
            if hasattr(api.read.session, "track_temperature"):
                data.update({
                    "track_temp": f"{api.read.session.track_temperature():.2f}",
                    "ambient_temp": f"{api.read.session.ambient_temperature():.2f}"
                })
            
            # Add to CSV
            self.static_data_writer.write_dict_row(data)
            
        except Exception as e:
            logger.error(f"Error logging static data: {str(e)}", exc_info=True)

    def _log_vehicles_data(self):
        """Log multi-vehicle dataset"""
        if not api or not self.vehicles_data_writer:
            return
        
        try:
            # Get total number of vehicles
            total_vehicles = api.read.vehicle.total_vehicles()
            
            if total_vehicles <= 0:
                return
                
            # Get player index
            player_index = api.read.vehicle.player_index() if hasattr(api.read.vehicle, "player_index") else 0
            
            # Common timestamp and session info
            timestamp = time.time()
            session_time = api.read.session.elapsed()
            
            # Add data for each vehicle
            for i in range(total_vehicles):
                try:
                    # Create basic vehicle data
                    vehicle_data = {
                        "timestamp": f"{timestamp:.3f}",
                        "session_time": f"{session_time:.3f}",
                        "index": i,
                        "is_player": api.read.vehicle.is_player(i),
                        "position": api.read.vehicle.place(i),
                        "driver_name": api.read.vehicle.driver_name(i),
                        "vehicle_name": api.read.vehicle.vehicle_name(i),
                        "vehicle_class": api.read.vehicle.class_name(i)
                    }
                    
                    # Add lap data
                    vehicle_data.update({
                        "lap_number": api.read.lap.number(i),
                        "lap_distance": f"{api.read.lap.distance(i):.2f}",
                        "lap_progress": f"{api.read.lap.progress(i):.2f}",
                        "last_laptime": f"{api.read.timing.last_laptime(i):.3f}",
                        "best_laptime": f"{api.read.timing.best_laptime(i):.3f}"
                    })
                    
                    # Add position data
                    veh_x, veh_y, veh_z = api.read.vehicle.position_xyz(i)
                    vehicle_data.update({
                        "world_x": f"{veh_x:.3f}",
                        "world_y": f"{veh_y:.3f}",
                        "world_z": f"{veh_z:.3f}",
                        "speed": f"{api.read.vehicle.speed(i):.3f}"
                    })
                    
                    # Add relative to player data if not the player
                    if not vehicle_data["is_player"] and player_index >= 0:
                        player_x, player_y, player_z = api.read.vehicle.position_xyz(player_index)
                        vehicle_data.update({
                            "relative_x": f"{veh_x - player_x:.3f}",
                            "relative_y": f"{veh_y - player_y:.3f}",
                            "relative_z": f"{veh_z - player_z:.3f}"
                        })
                    
                    # Add status data
                    vehicle_data.update({
                        "in_pit": api.read.vehicle.in_pits(i),
                        "pit_state": api.read.vehicle.pit_state(i) if hasattr(api.read.vehicle, "pit_state") else 0,
                        "num_pitstops": api.read.vehicle.number_pitstops(i) if hasattr(api.read.vehicle, "number_pitstops") else 0
                    })
                    
                    # Add to CSV
                    self.vehicles_data_writer.write_dict_row(vehicle_data)
                    
                except Exception as e:
                    logger.debug(f"Error collecting data for vehicle {i}: {str(e)}")
            
        except Exception as e:
            logger.error(f"Error logging vehicles data: {str(e)}", exc_info=True)
    
    def _open_new_writers(self, combo_id: str, session_id: tuple, session_type: int):
        """Create new HDF5 writers for the current session"""
        # Create a timestamped filename base
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_type_name = self._get_session_type_name(session_type)
        
        # Get track and vehicle names
        track_name = api.read.check.track_id() if api else "unknown_track"
        vehicle_name = api.read.check.vehicle_id() if api else "unknown_vehicle"
        
        # Sanitize names for filenames
        track_abbr = self._sanitize_filename(track_name)[:10]
        vehicle_abbr = self._sanitize_filename(vehicle_name)[:10]
        
        # Create session directory
        session_dir = os.path.join(self.log_path, f"{timestamp}_{track_abbr}_{vehicle_abbr}_{session_type_name}")
        os.makedirs(session_dir, exist_ok=True)
        
        # Create a single HDF5 file for all data
        telemetry_file = os.path.join(session_dir, "telemetry.h5")
        
        # Store session metadata in the file
        with h5py.File(telemetry_file, 'w') as f:
            meta = f.create_group('metadata')
            meta.attrs['timestamp'] = timestamp
            meta.attrs['combo_id'] = combo_id
            meta.attrs['session_id'] = str(session_id)
            meta.attrs['session_type'] = session_type
            meta.attrs['session_type_name'] = session_type_name
            meta.attrs['track_name'] = track_name
            meta.attrs['vehicle_name'] = vehicle_name
        
        # Create writers for different data types
        self.high_freq_writer = BinaryTelemetryLogger(
            filepath=session_dir,
            filename="telemetry.h5",
            group_name="telemetry_data"
        )
        
        self.static_data_writer = BinaryTelemetryLogger(
            filepath=session_dir,
            filename="telemetry.h5",
            group_name="static_data"
        )
        
        self.vehicles_data_writer = BinaryTelemetryLogger(
            filepath=session_dir,
            filename="telemetry.h5",
            group_name="vehicles_data"
        )
        
        # Create lap marker writer
        self.lap_marker_writer = BinaryTelemetryLogger(
            filepath=session_dir,
            filename="telemetry.h5",
            group_name="lap_markers"
        )
        
        logger.info(f"Created new telemetry log files in: {session_dir}")
    
    def _close_writers(self):
        """Close all active writers"""
        writers = [
            self.high_freq_writer,
            self.static_data_writer,
            self.vehicles_data_writer,
            getattr(self, "lap_marker_writer", None)
        ]
        
        for writer in writers:
            if writer:
                try:
                    writer.close()
                except Exception as e:
                    logger.error(f"Error closing writer: {str(e)}")
        
        self.high_freq_writer = None
        self.static_data_writer = None
        self.vehicles_data_writer = None
        self.lap_marker_writer = None
    
    def _get_session_type_name(self, session_type: int) -> str:
        """Convert session type number to a readable name"""
        session_types = {
            0: "idle",
            1: "practice",
            2: "qualify",
            3: "race_prep",
            4: "race",
            5: "time_trial"
        }
        return session_types.get(session_type, f"session_{session_type}")
    
    def _sanitize_filename(self, name: str) -> str:
        """Remove invalid characters from filename"""
        invalid_chars = '\\/:*?"<>|'
        for char in invalid_chars:
            name = name.replace(char, '_')
        return name

    def stop(self):
        """Stop the module"""
        # Stop logging threads
        self._stop_logging.set()
        
        # Wait for threads to finish
        if self._high_freq_thread:
            self._high_freq_thread.join(timeout=2.0)
        if self._static_data_thread:
            self._static_data_thread.join(timeout=2.0)
        if self._vehicles_data_thread:
            self._vehicles_data_thread.join(timeout=2.0)
        
        # Close writers
        self._close_writers()
        
        # Reset session tracking to force a new session on restart
        self.last_session_id = ("",-1,-1,-1)
        
        # Call parent stop method
        super().stop()