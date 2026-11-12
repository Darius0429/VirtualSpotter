"""
Vehicles module
"""
import pandas as pd

import array
import logging
import threading
from collections import namedtuple

from .module_info import minfo
from .api_control import api 
from . import calculation as calc
from . import validator as val
import csv
import os
import array
from typing import Callable
from collections import defaultdict
import time

chknm = val.infnan2zero 
cs2py = val.cbytes2str


MODULE_NAME = "delta_vehicles"


import os
from mistralai import Mistral

index_name = 'le-mans-ultimate'
mistral_api_key = os.environ["MISTRAL_API_KEY"]
model = "mistral-embed"


# logger = logging.getLogger(__name__)

class TelemetryCollector:
    def __init__(self, interval=0.1, output_file="telemetry_data.csv"):
        self.interval = interval  # Interval between reads in seconds
        self.data = defaultdict(lambda: defaultdict(list))  # Store time series data
        # self.data = pd
        self.running = False
        self.output_file = output_file
        self.thread = threading.Thread()

    def start_collecting(self):
        """Start collecting telemetry data from all methods."""
        self.running = True
        self.thread.start()
        for func_name, func in self.get_all_methods():
            self._collect(func_name, func)

    def stop_collecting(self):
        """Stop the data collection threads."""
        self.running = False
        self.thread.join()

        # Once data collection stops, save data to file
        self.save_to_file()

    def _collect(self, func_name: str, func: Callable):
        """Thread function to periodically collect data from a function."""
        while self.running:
            try:
                # Call the telemetry function and store the result
                result = func()
                timestamp = time.time()
                self.data[func_name].append(result)  # Store only the result, omit the timestamp
                print(f"{func_name}: {result}")
            except Exception as e:
                print(f"Error collecting {func_name}: {e}")
            
            # Sleep for the interval before the next read
            time.sleep(self.interval)

    def get_all_methods(self):
        """Get all telemetry functions from the reader, excluding Python special methods."""
        methods = []
        for category_name in dir(api.read):
            if category_name.startswith('__'):  # Skip any Python special attributes
                continue
            category = getattr(api.read, category_name)
            if callable(category):  # Skip non-callable attributes
                continue
            for method_name in dir(category):
                if method_name.startswith('__'):  # Skip special attributes like __dict__, __doc__, etc.
                    continue
                method = getattr(category, method_name)
                if callable(method):
                    methods.append((f"{category_name}.{method_name}", method))
        return methods

    def save_to_file(self):
        """Save collected telemetry data to a CSV file."""
        if not self.data:
            print("No data to save!")
            return

        # Get the maximum number of rows for each function (in case some functions were called more often)
        max_rows = max(len(values) for values in self.data.values())

        # Create a CSV file and write the headers and all data rows
        with open(self.output_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            
            # Write headers (function names as the first row)
            headers = list(self.data.keys())
            writer.writerow(headers)

            # Write data rows
            for i in range(max_rows):
                row = []
                for func_name in headers:
                    # Append the data if available, otherwise leave it empty (for missing data points)
                    if i < len(self.data[func_name]):
                        row.append(self.data[func_name][i])
                    else:
                        row.append("")  # Empty if no data available at this row for that function
                writer.writerow(row)

        print(f"Data successfully saved to {self.output_file}")


class StrategyCollector:
    """Vehicles info"""
    module_name = MODULE_NAME

    def __init__(self, interval=0.1, output_file="strategy_data.csv",):
        self.active = False
        self.race = False
        self.event = threading.Event()
        self.pit_time_list = array.array("f", [0,-1,0] * 128)
        self.data = defaultdict(lambda: defaultdict(list))
        # self.data = pd.DataFrame() 
        self.interval = interval
        self.output_file = output_file
        self.client = Mistral(api_key=mistral_api_key)
        # self.pc_client = Pinecone(api_key=pc_api_key)
        # self.namespace = "strategydata"
        # self.pc_client.create_index(
        #     name=index_name,
        #     dimension=1024,
        #     metric="cosine",
        #     spec=ServerlessSpec(
        #         cloud='aws', 
        #         region='us-east-1'
        #     ) 
        # ) 
        # while not self.pc_client.describe_index(index_name).status['ready']:
        #     time.sleep(1)
        
        # self.index = self.pc_client.Index(index_name)

    def start(self):
        # print()
        # self.race = reader.read.session.session_type() > 9
        """Start update thread"""
        if not self.active:# and self.race:
            self.active = True
            self.event.clear()
            threading.Thread(target=self.__update_data, daemon=True).start()
            self.output_file = time.strftime("%Y%m%d") + f"_{self.output_file}"
            # logger.info("ACTIVE: module vehicles")

    def stop(self):
        """Stop thread"""
        self.event.set()
        # self.pc_client.delete_index(index_name)

    def retrieve_data(self,label):
        match int(label):
            case 1:
                data = {
                    "system_prompt":
                    """
                    # Instructions:
                        Figure out the most relevant information for the driver, and respond briefly with the correct respond.
                    # Context about the data of the driver, every datapoint is when starting a new lap:
                        'fuel_add': the amount of fuel needed to add in a pitstop till the end
                        'fuel_current': current level of fuel
                        'fuel_consumption': the amount of fuel consumed a lap
                    """,
                    "data":{
                        "Balazs Szakter": {
                            # "elapsed_time": [1200.3, 1220.5, 1240.1],
                            # "total_laps": [1, 2, 3],
                            # "pos_abs": [5, 4, 3],
                            # "time_gap": [0.0, 0.0, 0.0],
                            # "last_laptime": [85.7, 85.9, 85.6],
                            # "best_laptime": [85.7, 85.7, 85.6],
                            # "in_pit": [False, False, False],
                            # "pit_count": [0, 0, 0]
                            "fuel_add": minfo.fuel.neededRelative,
                            "fuel_current": minfo.fuel.amountCurrent,
                            "fuel_consumption": minfo.fuel.estimatedConsumption,
                        },
                    },
                }
            case 2:
                data = {}
            case 3:
                data = {}
        #     "Damon Hill": {
        #         "elapsed_time": [1201.0, 1221.7, 1241.4],
        #         "total_laps": [1, 2, 3],
        #         "pos_abs": [6, 5, 4],
        #         "time_gap": [2.3, 2.1, 1.9],
        #         "last_laptime": [86.0, 85.9, 85.8],
        #         "best_laptime": [86.0, 85.9, 85.8],
        #         "in_pit": [False, False, False],
        #         "pit_count": [0, 0, 0]
        #     }
        
        return data

    def __update_data(self):
        """Update module data."""
        reset = False
        active_interval = self.interval  # Interval for updates
        update_interval = active_interval
        last_lap_stime = [-1] * minfo.vehicles.total  # last lap start time
        veh_total = max(minfo.vehicles.total, 1)
        (class_pos_list, place_index_list, is_multi_class) = self.__class_position_list(veh_total)
        minfo.relative.classes = class_pos_list
        vehicles_data = tuple(self.__update_vehicle_data(minfo.relative.classes))
        
        while not self.event.wait(update_interval):
            veh_total = max(minfo.vehicles.total, 1)
            (class_pos_list, place_index_list, is_multi_class) = self.__class_position_list(veh_total)
            minfo.relative.classes = class_pos_list
            vehicles_data = tuple(self.__update_vehicle_data(minfo.relative.classes))
            
            # Update driver data per lap
            for idx in range(veh_total):
                lap_stime = api.read.timing.start(idx)
                # print(last_lap_stime[idx])
                if lap_stime > last_lap_stime[idx] != -1:
                    telemetry = self.get_data(idx, vehicles_data)
                    driver_name = telemetry.get("drivername")  # Extract driver name from telemetry data
                    
                    if not driver_name:
                        print(f"Error: Telemetry data is missing 'drivername' for vehicle at index {idx}")
                        continue  # Skip this driver if we can't retrieve the name

                    # Store the timestamp and telemetry data for each key
                    timestamp = time.time()
                    for key, value in telemetry.items():
                        if key != "drivername":  # Exclude drivername from telemetry data
                            self.data[driver_name][key].append(value)
            
                    # self.save_to_file()
                last_lap_stime[idx] = lap_stime
            # Save the data to a file at regular intervals
            #self.process_and_store_telemetry_data(model, self.client, self.index,self.namespace)    

        self.active = False


    def __update_vehicle_data(self, class_pos_list):
        """Update vehicle data"""
        # Additional data
        veh_total = max(minfo.vehicles.total, 1)
        track_length = api.read.lap.track_length()
        in_race = api.read.session.in_race()
        class_list_size = len(class_pos_list)

        # Local player data
        plr_total_laps = api.read.lap.total_laps()
        plr_lap_distance = api.read.lap.distance()
        plr_percentage_distance = calc.percentage_distance(plr_lap_distance, track_length)
        plr_speed = api.read.vehicle.speed()
        plr_pos_xz = (api.read.vehicle.pos_longitudinal(),
                      api.read.vehicle.pos_lateral())
        plr_ori_rad = api.read.vehicle.orientation_yaw_radians()

        # Generate data list from all vehicles in current session
        for index in range(veh_total):
            is_player = api.read.vehicle.is_player(index)
            slot_id = api.read.vehicle.slot_id(index)
            position = api.read.vehicle.place(index)
            driver_name = api.read.vehicle.driver_name(index)
            vehicle_name = api.read.vehicle.vehicle_name(index)
            vehicle_class = api.read.vehicle.class_name(index)
            
            if index < class_list_size:
                position_in_class = class_pos_list[index][1]
                laptime_session_best = class_pos_list[index][3]
                laptime_class_best = class_pos_list[index][4]
                opt_index_ahead = class_pos_list[index][5]
            else:
                position_in_class = 0
                laptime_session_best = 99999
                laptime_class_best = 99999
                opt_index_ahead = -1

            laptime_best = api.read.timing.best_laptime(index)
            laptime_last = api.read.timing.last_laptime(index)
            lap_etime = api.read.timing.elapsed(index)
            speed = api.read.vehicle.speed(index)

            # Distance & time
            total_laps = api.read.lap.total_laps(index)
            lap_distance = api.read.lap.distance(index)

            percentage_distance = calc.percentage_distance(lap_distance, track_length)
            relative_distance = calc.circular_relative_distance(
                track_length, plr_lap_distance, lap_distance
                ) if not is_player else 0
            relative_time_gap = calc.relative_time_gap(
                relative_distance, speed, plr_speed
                ) if not is_player else 0

            gap_behind_next_in_class = self.__calc_gap_behind_next_in_class(
                opt_index_ahead, track_length, speed, total_laps, percentage_distance)
            gap_behind_next = self.__calc_gap_behind_next(index)
            gap_behind_leader = self.__calc_gap_behind_leader(index)

            is_lapped = 0 if is_player or not in_race else calc.lap_difference(
                total_laps + percentage_distance,
                plr_total_laps + plr_percentage_distance,
                3,
                3
                )
            is_yellow = speed < 8

            # Pit
            in_garage = api.read.vehicle.in_garage(index)
            in_pit = api.read.vehicle.in_pits(index)
            num_pit_stops = api.read.vehicle.number_pitstops(index)
            pit_state = api.read.vehicle.pit_state(index)
            pit_time = self.__calc_pit_time(
                index, in_pit, in_garage, laptime_last, lap_etime,
                in_pit * 1000 + slot_id)
            tire_compound = api.read.tyre.compound(index)
            

            # Position data
            pos_xz = (api.read.vehicle.pos_longitudinal(index),
                      api.read.vehicle.pos_lateral(index))
            orientation_xz_radians = api.read.vehicle.orientation_yaw_radians(index)
            relative_rotated_pos_xz = calc.rotate_pos(
                plr_ori_rad - 3.14159265,   # plr_ori_rad, rotate view
                pos_xz[0] - plr_pos_xz[0],  # x position related to player
                pos_xz[1] - plr_pos_xz[1]   # y position related to player
                ) if not is_player else (0,0)
            relative_orientation_xz_radians = (
                orientation_xz_radians - plr_ori_rad
                ) if not is_player else 0
            relative_straight_distance = calc.distance(
                plr_pos_xz, pos_xz
                ) if not is_player else 0
            yield DataSet(
                position = position,
                driverName = driver_name,
                vehicleName = vehicle_name,
                vehicleClass = vehicle_class,
                positionInClass = position_in_class,
                sessionBestLapTime = laptime_session_best,
                classBestLapTime = laptime_class_best,
                bestLapTime = laptime_best,
                lastLapTime = laptime_last,
                isPlayer = is_player,
                isNotPlayer = not is_player,
                percentageDistance = percentage_distance,
                relativeDistance = relative_distance,
                relativeTimeGap = relative_time_gap,
                gapBehindNextInClass = gap_behind_next_in_class,
                gapBehindNext = gap_behind_next,
                gapBehindLeader = gap_behind_leader,
                isLapped = is_lapped,
                isYellow = is_yellow,
                inGarage = in_garage,
                inPit = in_pit,
                numPitStops = num_pit_stops,
                pitState = pit_state,
                pitTime = pit_time,
                tireCompound = tire_compound,
                posXZ = pos_xz,
                relativeOrientationXZRadians = relative_orientation_xz_radians,
                relativeRotatedPosXZ = relative_rotated_pos_xz,
                relativeStraightDistance = relative_straight_distance,
                elapsedTime = lap_etime,
                totalLaps = total_laps,
                #slotID = slot_id,
                #speed = speed,
                #totalLaps = total_laps,
                #lapDistance = lap_distance,
                #orientationXZRadians = orientation_xz_radians,
                )

    def __calc_pit_time(self, index, in_pit, in_garage, laptime_last, lap_etime, pit_status):
        """Calculate lap & pit time"""
        index *= 3
        idx_inpit, idx_start, idx_timer = index, index + 1, index + 2
        # Pit status check
        if pit_status != self.pit_time_list[idx_inpit]:
            self.pit_time_list[idx_inpit] = pit_status  # last pit status
            self.pit_time_list[idx_start] = lap_etime  # last etime stamp
        # Ignore pit timer in garage
        if in_garage:
            self.pit_time_list[idx_start] = -1
            self.pit_time_list[idx_timer] = 0
            return 0
        # Calculating pit time while in pit
        if in_pit:
            if self.pit_time_list[idx_start] >= 0:
                self.pit_time_list[idx_timer] = min(
                    max(lap_etime - self.pit_time_list[idx_start], 0), 999.9)
        # Reset pit time if made a valid lap time after pitting
        else:
            if self.pit_time_list[idx_timer] and laptime_last > 0:
                self.pit_time_list[idx_timer] = 0
        return self.pit_time_list[idx_timer]

    @staticmethod
    def __calc_gap_behind_next_in_class(
        opt_index, track_length, speed, total_laps, percentage_distance):
        """Calculate interval behind next in class"""
        if opt_index < 0:
            return 0.0
        opt_total_laps = api.read.lap.total_laps(opt_index)
        opt_lap_distance = api.read.lap.distance(opt_index)
        opt_percentage_distance = calc.percentage_distance(opt_lap_distance, track_length)
        lap_diff = abs(opt_total_laps + opt_percentage_distance - total_laps - percentage_distance)
        if lap_diff > 1:
            return int(lap_diff)
        return calc.relative_time_gap(
            lap_diff * track_length, api.read.vehicle.speed(opt_index), speed
        )

    @staticmethod
    def __calc_gap_behind_next(index):
        """Calculate interval behind next"""
        laps_behind_next = api.read.lap.behind_next(index)
        if laps_behind_next > 0:
            return laps_behind_next
        return api.read.timing.behind_next(index)

    @staticmethod
    def __calc_gap_behind_leader(index):
        """Calculate interval behind leader"""
        laps_behind_leader = api.read.lap.behind_leader(index)
        if laps_behind_leader > 0:
            return laps_behind_leader
        return api.read.timing.behind_leader(index)
    
    def __class_position_list(self, veh_total):
        """Create vehicle class position list"""
        raw_veh_class = list(self.__raw_class_data(veh_total))
        split_veh_list = tuple(zip(*raw_veh_class))
        # Multi-class check
        is_multi_class = len(set(split_veh_list[0])) > 1
        # Create overall vehicle place, player index list
        place_index_list = sorted(zip(split_veh_list[1], split_veh_list[2]))
        # Create class position list
        raw_veh_class.sort()  # sort by vehicle class
        class_position_list = sorted(self.__sort_class_data(raw_veh_class))
        return class_position_list, place_index_list, is_multi_class
    
    @staticmethod
    def __raw_class_data(veh_total):
        """Raw vehicle class data"""
        for index in range(veh_total):
            vehclass = api.read.vehicle.class_name(index)
            position = api.read.vehicle.place(index)
            laptime_best = api.read.timing.best_laptime(index)
            yield (
                vehclass,  # 0 vehicle class name
                position,  # 1 overall position/place
                index,     # 2 player index
                laptime_best if laptime_best > 0 else 99999, # 3 best lap time
            )

    @staticmethod
    def __sort_class_data(sorted_veh_class):
        """Sort vehicle class position data"""
        laptime_session_best = calc.session_best_laptime(sorted_veh_class, 3)
        laptime_class_best = 99999
        initial_class = sorted_veh_class[0][0]
        position_in_class = 0
        player_index_ahead = -1
        player_index_behind = -1
        total_veh = len(sorted_veh_class)

        for idx, veh_sort in enumerate(sorted_veh_class):
            if veh_sort[0] == initial_class:
                position_in_class += 1
            else:
                initial_class = veh_sort[0]  # reset init name
                position_in_class = 1  # reset position counter

            if position_in_class == 1:
                laptime_class_best = veh_sort[3]
                player_index_ahead = -1  # no player ahead

            if (idx + 1 < total_veh  # check next index within range
                and sorted_veh_class[idx + 1][0] == veh_sort[0]):  # next player is in same class
                player_index_behind = sorted_veh_class[idx + 1][2]
            else:
                player_index_behind = -1

            yield (
                veh_sort[2],       # 0 - 2 player index
                position_in_class,  # 1 - position in class
                veh_sort[0],       # 2 - 0 class name
                laptime_session_best,  # 3 session best
                laptime_class_best,  # 4 classes best
                player_index_ahead,  # 5 player index ahead
                player_index_behind,  # 6 player index behind
            )
            player_index_ahead = veh_sort[2]


    def get_data(self, index, vehicles_data):
        """Relative data"""
        # Prevent index out of range
        if vehicles_data and 0 <= index < len(vehicles_data):

            # check whether is lapped
            is_lapped = vehicles_data[index].isLapped

            # 0 Is player
            is_player = vehicles_data[index].isPlayer

            total_laps = vehicles_data[index].totalLaps

            # 1 Vehicle in pit
            in_pit = vehicles_data[index].inPit

            elapsed_time = vehicles_data[index].elapsedTime
            print(elapsed_time)
            # 2 Driver position
            position = (f"{vehicles_data[index].position:02d}", is_lapped)

            pos_abs = int(position[0])
            # 3 Driver name
            drv_name = (vehicles_data[index].driverName, is_lapped)

            # 4 Vehicle name
            veh_name = (vehicles_data[index].vehicleName, is_lapped)

            # 5 Vehicle position in class
            pos_class = f"{vehicles_data[index].positionInClass:02d}"
            # 6 Vehicle class
            veh_class = vehicles_data[index].vehicleClass

            # 7 Time gap
            time_gap = (f"{vehicles_data[index].relativeTimeGap:.02f}", is_lapped)

            # 8 Tyre compound index
            tire_idx = vehicles_data[index].tireCompound
            compound_front = tire_idx[0]
            compound_rear = tire_idx[1]

            # 9 Lap time
            last_laptime = f"{vehicles_data[index].lastLapTime:.02f}"
            laptime = self.set_laptime(
                vehicles_data[index].inPit,
                vehicles_data[index].lastLapTime,
                vehicles_data[index].pitTime
            )
            best_laptime = f"{vehicles_data[index].bestLapTime:.02f}"

            # 10 Pitstop count
            pit_count = (vehicles_data[index].numPitStops,
                         vehicles_data[index].pitState)

            # return (is_player, in_pit, pos_abs, drv_name, veh_name, pos_class, veh_class,
            #         time_gap, tire_idx, laptime, pit_count,best_laptime)
        
            return {
                "drivername": drv_name[0],
                "elapsed_time": elapsed_time,
                "total_laps": total_laps,
                "is_player": is_player,
                "veh_class": veh_class,
                "pos_abs": pos_abs,
                "pos_class": pos_class,
                "time_gap": time_gap[0],
                "last_laptime": last_laptime,
                "best_laptime": best_laptime,
                "is_lapped": is_lapped,
                "in_pit": in_pit,
                "pit_state": pit_count[1],
                "pit_count": pit_count[0],
                # "compound_front": compound_front,
                # "compound_rear": compound_rear
            }
        
    @staticmethod
    def set_laptime(inpit, laptime_last, pit_time):
        """Set lap time"""
        if inpit:
            return "PIT" + f"{pit_time:.01f}"[:5].rjust(5) if pit_time > 0 else "-:--.---"
        if laptime_last <= 0:
            return "OUT" + f"{pit_time:.01f}"[:5].rjust(5) if pit_time > 0 else "-:--.---"
        return calc.sec2laptime_full(laptime_last)[:8].rjust(8)
    
    def save_to_file(self):
        """Save collected telemetry data for multiple drivers to a CSV file."""
        if not self.data:
            # print("No data to save!")
            return

        # Ensure "elapsed_time" exists for each driver
        if "elapsed_time" not in self.data[0]:
            print("No elapsed time data found!")
            return

        # Create a CSV file and write the headers and all data rows
        with open(self.output_file, mode='w', newline='') as file:
            writer = csv.writer(file)

            # Collect all telemetry metrics across all drivers to ensure we capture all headers
            all_metrics = set()
            for driver_data in self.data.values():
                all_metrics.update(driver_data.keys())

            # Convert set to sorted list for consistent column ordering
            all_metrics = sorted(all_metrics)

            # Write headers: 'Driver', 'Elapsed Time', followed by all telemetry metric names
            headers = ['Driver', 'Elapsed Time'] + all_metrics
            writer.writerow(headers)

            # Write data rows for each driver and each telemetry metric
            for driver, telemetry_data in self.data.items():
                if driver == "elapsed_time":  # Skip if we're on the elapsed_time key
                    continue

                # Get the time series (elapsed_time) from the telemetry data
                elapsed_time = self.data["elapsed_time"]

                # Iterate over each time step and collect telemetry data for this driver
                max_rows = len(elapsed_time)  # Use elapsed_time as the row count baseline

                for i in range(max_rows):
                    row = [driver]  # First column is the driver name

                    # Get the elapsed time for this time step
                    if i < len(elapsed_time):
                        time_value = elapsed_time[i]
                    else:
                        time_value = ""

                    row.append(time_value)

                    # Collect data for each metric in the row
                    for metric in all_metrics:
                        if metric in telemetry_data and i < len(telemetry_data[metric]):
                            value = telemetry_data[metric][i]
                            row.append(value)
                        else:
                            row.append("")  # No data for this metric at this time step

                    writer.writerow(row)

        print(f"Data successfully saved to {self.output_file}")

    def process_and_store_telemetry_data(self, model, client, pinecone_index,namespace):
        # Step 1: Flatten the telemetry data
        flattened_data = flatten_telemetry_data(self.data)
        
        # Step 2: Get embeddings for the flattened data
        embeddings = get_mistral_embeddings(flattened_data, model, client)
        
        # Step 3: Store embeddings in Pinecone
        store_embeddings_in_pinecone(embeddings, flattened_data, pinecone_index,namespace)

def flatten_telemetry_data(data):
    """Flatten the nested defaultdict structure into a list of strings for embedding."""
    flattened_data = []
    
    for driver_name, telemetry_data in data.items():
        # Concatenate key-value pairs for each driver into a single string
        for key, values in telemetry_data.items():
            for value in values:
                # Create a text representation for embedding
                text_to_embed = f"Driver: {driver_name}, {key}: {value}"
                flattened_data.append(text_to_embed)
    
    return flattened_data
import random
def get_mistral_embeddings(text_data, model, client, batch_size=10, max_retries=5):
    """Get Mistral embeddings for the flattened text data with retry logic and store embeddings."""
    embeddings = []
    
    for i in range(0, len(text_data), batch_size):
        batch = text_data[i:i + batch_size]
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                response = client.embeddings.create(model=model, inputs=batch)
                embeddings += [d.embedding for d in response.data]
                break  # Exit retry loop on success
            except Exception as e:
                # Handle rate limit and retry after a delay
                if "429" in str(e):
                    retry_count += 1
                    wait_time = 2 ** retry_count + random.uniform(0, 1)
                    print(f"Rate limit exceeded, retrying in {wait_time:.2f} seconds...")
                    time.sleep(wait_time)
                else:
                    raise e
    
    return embeddings

import pinecone

def store_embeddings_in_pinecone(embeddings, text_data, pinecone_index, namespace):
    """Store the embeddings in a Pinecone index."""
    for i, embedding in enumerate(embeddings):
        metadata = {
            "original_text": text_data[i]  # Store the original flattened text for reference
        }
        
        # Upsert embedding with metadata into Pinecone
        pinecone_index.upsert(
            vectors=[{
                "id": f"embed_{i}",  # Unique ID for the embedding
                "values": embedding,
                "metadata": metadata
            }],
            namespace=namespace
        )





def nearest_distance_data(
    data_list: list,
    dist_line: int = 999999,
    dist_time: int = 999999,
    dist_yellow: int = 999999):
    """Calculate nearest distance data"""
    for data in data_list:
        # Find nearest straight line distance
        if not data.isPlayer and data.relativeStraightDistance < dist_line:
            dist_line = data.relativeStraightDistance
        # Find nearest traffic time gap
        if 0 == data.inPit > data.relativeDistance and data.relativeTimeGap < dist_time:
            dist_time = data.relativeTimeGap
        # Find nearest yellow flag (on track) distance
        if data.isYellow:
            rel_dist = abs(data.relativeDistance)
            if rel_dist < dist_yellow:
                dist_yellow = rel_dist
    return dist_line, dist_time, dist_yellow


DataSet = namedtuple(
    "DataSet",
    [
        # Sorting priority section (for track map display, set reverse sort)
        "isNotPlayer",
        "inGarage",
        "inPit",
        "position",
        "driverName",
        "vehicleName",
        "vehicleClass",
        "positionInClass",
        "sessionBestLapTime",
        "classBestLapTime",
        "bestLapTime",
        "lastLapTime",
        "isPlayer",
        "percentageDistance",
        "relativeDistance",
        "relativeTimeGap",
        "gapBehindNextInClass",
        "gapBehindNext",
        "gapBehindLeader",
        "isLapped",
        "isYellow",
        "numPitStops",
        "pitState",
        "pitTime",
        "tireCompound",
        "posXZ",
        "relativeOrientationXZRadians",
        "relativeRotatedPosXZ",
        "relativeStraightDistance",
        "elapsedTime",  # Add elapsedTime here
        "totalLaps"     # Add totalLaps here
    ]
)

# reader.connect("Le Mans Ultimate")
# reader.start()
# client = StrategyCollector(interval=1)
# client.start()
# time.sleep(8*60)
# client.stop()
# reader.stop()
