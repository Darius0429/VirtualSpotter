# from ..delta_info import dinfo
from ..data_reader import reader
import threading
from dataclasses import dataclass, field
from ..import calculation as calc
from .. import validator as val
from collections import namedtuple

chknm = val.infnan2zero
cs2py = val.cbytes2str

@dataclass
class DriverInfo:
    """Driver-specific information"""
    drivername: str = ""
    is_player: bool = False
    veh_class: str = ""
    pos_class: int = 0
    sector_times: dict = field(default_factory=dict)  # Relative gap to the next driver
    best_lap: float = 9999.0

@dataclass
class DriverData:
    """Delta module output data"""
    drivers: list[DriverInfo] = field(default_factory=list)

@dataclass
class RelativeInfo:
    """Relative module output data"""
    relative: list = field(default_factory=list)
    standings: list = field(default_factory=list)
    classes: list = field(default_factory=list)

@dataclass
class StrategyInfo:
    strategy: str = ""


class ModuleInfo:
    """Modules output data"""

    def __init__(self):
        self.driver = DriverInfo()
        self.relative = RelativeInfo()
        self.delta = DriverData()

dinfo = ModuleInfo()

class FastestLapSet():
    def __init__(self):
        self.active = False
        self.event = threading.Event()
        self.latest_fastest_driver = None  # Store the latest fastest lap driver info
        self.compare_lap = None
        print("initialised")
        self.lock = threading.Lock()

    def start(self):
        if not self.active:
            self.active = True
            self.event.clear()
            threading.Thread(target=self.__update_data, daemon=True).start()
            print("started thread")

    def stop(self):
        self.event.set()

    def __update_data(self):
        """Update module data"""
        try:
            reset = False
            active_interval = 1  # Interval for updates
            update_interval = active_interval
            last_lap_stime = [-1] * reader.read.vehicle.total_vehicles()  # last lap start time
            veh_total = max(reader.read.vehicle.total_vehicles(), 1)
            print("total vehs: ", veh_total)
            (class_pos_list, place_index_list, is_multi_class) = self.__class_position_list(veh_total)
            dinfo.relative.classes = class_pos_list

            vehicles_data = tuple(self.__update_vehicle_data(dinfo.relative.classes))
            while not self.event.wait(update_interval):
                update_interval = active_interval
                if not reset:
                    reset = True
                    # Initialize drivers at the start
                    for idx in range(reader.read.vehicle.total_vehicles()):
                        try:
                            (is_player, in_pit, pos_abs, drv_name, veh_name, 
                            pos_class, veh_class, time_gap, tire_idx, last_lap, pit_count, best_lap, best_sectors) = self.get_data(idx, vehicles_data)
                            # Ensure driver is not already in the list
                            if drv_name[0] not in [driver.drivername for driver in dinfo.delta.drivers]:
                                dinfo.delta.drivers.append(DriverInfo(
                                    drivername=drv_name[0],
                                    is_player=is_player,
                                    veh_class=veh_class,
                                    pos_class=pos_class,
                                    sector_times=best_sectors,
                                    best_lap=best_lap
                                ))
                        except Exception as e:
                            print(f"Exception in initializing driver data: {e}")
                veh_total = max(reader.read.vehicle.total_vehicles(), 1)
                (class_pos_list, place_index_list, is_multi_class) = self.__class_position_list(veh_total)
                dinfo.relative.classes = class_pos_list
                vehicles_data = tuple(self.__update_vehicle_data(dinfo.relative.classes))
                # Update driver data per lap
                for idx in range(veh_total):
                    if idx > len(last_lap_stime):
                        last_lap_stime.append()  # Append default value
                    try:
                        lap_stime = reader.read.timing.start(idx)
                        (is_player, in_pit, pos_abs, drv_name, veh_name, 
                            pos_class, veh_class, time_gap, tire_idx, last_lap, pit_count, best_lap, best_sectors) = self.get_data(idx, vehicles_data)
                        
                        # if idx >= len(last_lap_stime):
                        #     last_lap_stime.append(-1)  # Append default value
                        # Find driver
                        driver_found = next((driver for driver in dinfo.delta.drivers if driver.drivername == drv_name[0]), None)
                        if driver_found:
                            if lap_stime > last_lap_stime[idx] != -1:
                                # Update data for existing driver
                                if best_lap < driver_found.best_lap:
                                    print("supposed to trigger")
                                    driver_found.best_lap = best_lap
                                    driver_found.sector_times = best_sectors
                                    driver_found.pos_class = int(pos_class)
                                    with self.lock:
                                        self.latest_fastest_driver = driver_found

                                    if best_lap < self.find_myself_as_driver().best_lap:
                                        self.compare_lap = driver_found

                        else:
                            # Add new driver
                            dinfo.delta.drivers.append(DriverInfo(
                                drivername=drv_name[0],
                                is_player=is_player,
                                veh_class=veh_class,
                                pos_class=pos_class,
                                sector_times=best_sectors,
                                best_lap=best_lap
                            ))

                        # Update lap start time
                        last_lap_stime[idx] = lap_stime
                    except Exception as e:
                        print(f"Exception during lap data update for vehicle index {idx}: {e}")

        except Exception as e:
            print(f"Exception in thread: {e}")


    def find_myself_as_driver(self):
        # Search for the driver where is_player is True
        my_driver = next((driver for driver in dinfo.delta.drivers if driver.is_player), None)
        
        if my_driver:
            print(f"Found myself as driver: {my_driver.drivername}")
            return my_driver
        else:
            print("Could not find myself as driver.")
            return None

    def check_event(self):
        """Check if a new fastest lap has been set and return the driver info."""
        if self.latest_fastest_driver:
            return self.latest_fastest_driver  # Return the latest driver who set the fastest lap
        return None
    def sector_comparison(self):
        """Check if a new fastest lap has been set and return the driver info."""
        if self.compare_lap:
            return self.compare_lap,self.find_myself_as_driver() # Return the latest driver who set the fastest lap
        return None

    def __update_vehicle_data(self, class_pos_list):
        """Update vehicle data"""
        # Additional data
        
        veh_total = max(reader.read.vehicle.total_vehicles(), 1)
        track_length = reader.read.lap.track_length()
        in_race = reader.read.session.in_race()
        class_list_size = len(class_pos_list)
        
        # Local player data
        plr_total_laps = reader.read.lap.total_laps()
        plr_lap_distance = reader.read.lap.distance()
        plr_percentage_distance = calc.percentage_distance(plr_lap_distance, track_length)
        plr_speed = reader.read.vehicle.speed()
        plr_pos_xz = (reader.read.vehicle.pos_longitudinal(),
                      reader.read.vehicle.pos_lateral())
        plr_ori_rad = reader.read.vehicle.orientation_yaw_radians()

        # Generate data list from all vehicles in current session
        for index in range(veh_total):
            is_player = reader.read.vehicle.is_player(index)
            slot_id = reader.read.vehicle.slot_id(index)
            position = reader.read.vehicle.place(index)
            driver_name = reader.read.vehicle.driver_name(index)
            vehicle_name = reader.read.vehicle.vehicle_name(index)
            vehicle_class = reader.read.vehicle.class_name(index)
            
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

            laptime_best = reader.read.timing.best_laptime(index)
            best_sector1 = reader.read.timing.best_sector1(index)
            best_sector2 = reader.read.timing.best_sector2(index)-best_sector1
            best_sector3 = laptime_best-best_sector2-best_sector1
            best_sectors = {
                "sector1": best_sector1,
                "sector2": best_sector2,
                "sector3": best_sector3
            }

            laptime_last = reader.read.timing.last_laptime(index)
            lap_etime = reader.read.timing.elapsed(index)
            speed = reader.read.vehicle.speed(index)

            # Distance & time
            total_laps = reader.read.lap.total_laps(index)
            lap_distance = reader.read.lap.distance(index)

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
            in_garage = reader.read.vehicle.in_garage(index)
            in_pit = reader.read.vehicle.in_pits(index)
            num_pit_stops = reader.read.vehicle.number_pitstops(index)
            pit_state = reader.read.vehicle.pit_state(index)
            # pit_time = self.__calc_pit_time(
            #     index, in_pit, in_garage, laptime_last, lap_etime,
            #     in_pit * 1000 + slot_id)
            pit_time = 5.0
            tire_compound = reader.read.tyre.compound(index)

            # Position data
            pos_xz = (reader.read.vehicle.pos_longitudinal(index),
                      reader.read.vehicle.pos_lateral(index))
            orientation_xz_radians = reader.read.vehicle.orientation_yaw_radians(index)
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
                bestSectorTimes = best_sectors,
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
        opt_total_laps = reader.read.lap.total_laps(opt_index)
        opt_lap_distance = reader.read.lap.distance(opt_index)
        opt_percentage_distance = calc.percentage_distance(opt_lap_distance, track_length)
        lap_diff = abs(opt_total_laps + opt_percentage_distance - total_laps - percentage_distance)
        if lap_diff > 1:
            return int(lap_diff)
        return calc.relative_time_gap(
            lap_diff * track_length, reader.read.vehicle.speed(opt_index), speed
        )

    @staticmethod
    def __calc_gap_behind_next(index):
        """Calculate interval behind next"""
        laps_behind_next = reader.read.lap.behind_next(index)
        if laps_behind_next > 0:
            return laps_behind_next
        return reader.read.timing.behind_next(index)

    @staticmethod
    def __calc_gap_behind_leader(index):
        """Calculate interval behind leader"""
        laps_behind_leader = reader.read.lap.behind_leader(index)
        if laps_behind_leader > 0:
            return laps_behind_leader
        return reader.read.timing.behind_leader(index)
    
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
            vehclass = reader.read.vehicle.class_name(index)
            position = reader.read.vehicle.place(index)
            laptime_best = reader.read.timing.best_laptime(index)
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

            # 1 Vehicle in pit
            in_pit = vehicles_data[index].inPit

            # 2 Driver position
            position = (f"{vehicles_data[index].position:02d}", is_lapped)
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

            # 9 Lap time
            laptime = self.set_laptime(
                vehicles_data[index].inPit,
                vehicles_data[index].lastLapTime,
                vehicles_data[index].pitTime
            )
            best_laptime = f"{vehicles_data[index].bestLapTime:.03f}"

            best_sectors = vehicles_data[index].bestSectorTimes
            best_sectors = {key: round(value, 3) for key, value in best_sectors.items()}
            # best_sectors = {}
            # for sector,i in zip(vehicles_data[index].bestSectorTimes,range(len(vehicles_data[index].bestSectorTimes))):
            #     best_sectors[f"sector{i+1}"] = f"{sector:.03f}"

            # 10 Pitstop count
            pit_count = (vehicles_data[index].numPitStops,
                         vehicles_data[index].pitState)

            return (is_player, in_pit, position, drv_name, veh_name, pos_class, veh_class,
                    time_gap, tire_idx, laptime, pit_count,best_laptime,best_sectors)
        
    @staticmethod
    def set_laptime(inpit, laptime_last, pit_time):
        """Set lap time"""
        if inpit:
            return "PIT" + f"{pit_time:.01f}"[:5].rjust(5) if pit_time > 0 else "-:--.---"
        if laptime_last <= 0:
            return "OUT" + f"{pit_time:.01f}"[:5].rjust(5) if pit_time > 0 else "-:--.---"
        return calc.sec2laptime_full(laptime_last)[:8].rjust(8)

# class FastestLapSet():
#     def __init__(self):
#         self.active = False
#         self.event = threading.Event()
#         self.latest_fastest_driver = None  # Store the latest fastest lap driver info
#         self.compare_lap = None
#         print("initialised")
#         self.lock = threading.Lock()

#     def start(self):
#         if not self.active:
#             self.active = True
#             self.event.clear()
#             threading.Thread(target=self.__update_data, daemon=True).start()
#             print("started thread")

#     def stop(self):
#         self.event.set()


DataSet = namedtuple(
    "DataSet",
    [
    # Sorting priority section (for track map display, set reverse sort)
    "isNotPlayer",
    "inGarage",
    "inPit",
    "position",
    # End of sorting priority section
    "driverName",
    "vehicleName",
    "vehicleClass",
    "positionInClass",
    "sessionBestLapTime",
    "classBestLapTime",
    "bestLapTime",
    "bestSectorTimes",
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
    #"slotID",
    #"speed",
    #"totalLaps",
    #"lapDistance",
    #"orientationXZRadians",
    ]
)
