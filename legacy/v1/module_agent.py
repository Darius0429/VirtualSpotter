from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage,AnyMessage, SystemMessage
from langchain_mistralai.chat_models import ChatMistralAI
from typing import TypedDict, Literal,Annotated, Dict
import os
import operator
from langgraph.graph import StateGraph, MessagesState, END
import asyncio
from code.LLM.events import FastestLapSet
import time
import pygame
import os
from io import BytesIO
from elevenlabs import VoiceSettings
from elevenlabs.client import ElevenLabs
from dataclasses import asdict
import json
from ai_race_engineer.game_controller import GameController
from ai_race_engineer.S2T import Speech2Text
from ai_race_engineer.logging import StrategyCollector
from pydantic import BaseModel, Field
from tinypedal.tinypedal.api_control import api
from tinypedal.tinypedal.module_info import *

# Define the state for the graph
class RaceEngineerState(MessagesState):
    session_type: int
    events: Annotated[list[str], operator.add] = []  # Ensure this is initialized as an empty list
    event_details: Annotated[list[str], operator.add] = []  # Ensure this is initialized as an empty list
    event_type: str
    # practice: bool
    # qualifying_prep: bool
    # qualifying: bool
    # race_prep: bool
    # race: bool
    data: Annotated[list[str], operator.add] = []  # Ensure this is initialized as an empty list
    data_type: int
    messages: Annotated[list[AnyMessage], operator.add] = []
    active: bool
    triggered: bool
    race_start: bool 

class RaceEngineerAgent:
    def __init__(self):

        self.fastest_lap_event = FastestLapSet()

        self.model_interaction= ChatMistralAI(api_key=os.getenv("MISTRAL_API_KEY"),model_name="open-mistral-7b")
        self.model_response= ChatMistralAI(api_key=os.getenv("MISTRAL_API_KEY2"))
        self.model_data = ChatMistralAI(api_key=os.getenv("MISTRAL_API_KEY3"))
        # self.model_data = self.model_data.with_structured_output(code, include_raw=False)

        self.t2s_client = ElevenLabs(api_key=os.environ["TTS_API_KEY"])

        self.data_client = StrategyCollector(interval=1)

        self.mixer = pygame.mixer

        self.mixer.init()

        self.wheel = GameController()

        self.s2t = Speech2Text()

        # Define the graph
        self.graph = StateGraph(RaceEngineerState)
        self.start = False
        self.setup_graph() 
        self.session_type = 0
        

    def setup_graph(self):
        # Add nodes to the graph
        self.graph.add_node("LLM Interaction", self.llm_interaction_node)
        self.graph.add_node("Session Checker", self.session_checker_node)

        self.graph.add_node("Idle",self.idle_node)
        self.graph.add_node("Practice Session", self.practice_session_node)
        self.graph.add_node("Qualifying Session", self.qualifying_session_node)
        self.graph.add_node("Race Preparation", self.race_preparation_node)
        self.graph.add_node("Race Session", self.race_session_node)
        self.graph.add_node("Event Handling", self.event_handling_node)

        self.graph.add_node("Data Fetching", self.fetch_data_node)
        self.graph.add_node("Data Storing", self.data_storer_node)


        self.graph.add_node("LLM Response", self.llm_response_node)
        # Define the entry point
        self.graph.set_entry_point("Session Checker")

        self.graph.add_edge("Session Checker", "LLM Interaction")

        # self.graph.add_conditional_edges("Session Checker", lambda state: bool(state["query"]),{
        #     True: "LLM Interaction",
        #     False: END
        # })

        # Define the edges based on session type and events
        self.graph.add_conditional_edges("Session Checker", lambda state: state["session_type"], {
            0: "Idle",
            1: "Practice Session",
            2: "Qualifying Session",
            3: "Race Preparation",
            4: "Race Session"
        })

        self.graph.add_conditional_edges("LLM Interaction", lambda state: state["data_type"],{
            "0": "Data Fetching",
            "1": "LLM Response",
            "2": END
        })


        self.graph.add_edge("Practice Session", "Event Handling")
        self.graph.add_edge("Qualifying Session", "Event Handling")
        self.graph.add_edge("Race Preparation", "Event Handling")
        self.graph.add_edge("Race Session", "Event Handling")
        self.graph.add_edge("Race Session", "Data Storing")
        # self.graph.add_conditional_edges("Race Session",lambda state:state.get("race_start", False),{
        #     False: "Data Storing"
        # }) 

        # Define the stopping criteria
        self.graph.add_edge("Data Fetching","LLM Response")
        # graph.add_edge("Event Handling", "LLM Response")
        self.graph.add_conditional_edges("Event Handling",lambda state: state["triggered"],{
            True: "LLM Response",
            False: END
        })

        self.graph.add_edge("LLM Response",END)

        # Compile the graph
        self.race_engineer_agent = self.graph.compile()

    async def llm_interaction_node(self,state: RaceEngineerState):

        if state["active"]:
            # Central node for user interaction
            system= f"""
                You are a professional Race Engineer to me Balazs Szakter. We are currently in a race session. I am driving and I ask you questions about the others or my strategy.

                # Instructions:
                1. Your task is to classify the user prompt in 2 categories:
                    "0": if the user question is about data, strategy, telemetry.
                    "1": if the user question is not about the session and is more general (e.g. conversational questions). 
                2. You must respond the classified label only, nothing else!
            """
            time.sleep(1.1)
            messages = state["messages"]
            messages = [SystemMessage(content=system)] + messages
            response = await self.model_interaction.ainvoke(input=messages)
            print(response.content)
            return {"data_type": response.content}
        else:
            # print("Not active")
            return {"data_type": "2"}

    def get_telemetry_data(self):
        print("getting telemetry data")
        data1 = 1
        data_string = f"My best lap is {data1}"  # Example formatting
        return data_string

    async def fetch_data_node(self,state: RaceEngineerState):
        self.data = self.data_client.retrieve_data()
        system = f"""
        You are a professional Race Data Fetcher to me Balazs Szakter. We are currently in a race session. The user wants to get data from the current session.

        # Instructions:
        1. Your task is to answer to user query.
        2. You must get the precise answer to the question, if it means that you have to do calculations.
        3. You must respond very briefly so skip calculations, just give the result.

        # Context about the data of the driver, every datapoint is when starting a new lap:
            'elapsed_time': Time that the lap started from the start of the session.
            'total_laps': Number of laps completed, the index of the lap
            'pos_abs': Which position they are in when starting the lap, across all the field
            'pos_class': Which position they are in when starting the lap, relative to the class (MORE IMPORTANT THAN 'pos_abs')
            'time_gap': The driver relative gap to me (Balazs Szakter) at the end of the lap
            'last_laptime': The laptime after crossing the finish line
            'best_laptime': Their best laptime so far
            'is_lapped': If they are lapped by the leader
            'in_pit': If they are in the pits
            'pit_state': If they pitted on the lap
            'pit_count': How many pits they completed so far

        # Data:
        {self.data_client.retrieve_data()}

        # My data with drivername "Balazs Szakter"
        """
        # print(system)
        time.sleep(1.1)
        messages = state["messages"]
        messages = [SystemMessage(content=system)] + messages
        # data = await self.fetch_and_execute_code(messages=messages,state=RaceEngineerState)
        # time.sleep(1.5)
        response = await self.model_data.ainvoke(input=messages)
        
        return {"data": [response.content]}

    async def data_storer_node(self,state: RaceEngineerState):
        if not self.start:
            self.start = True
            
            self.data_client.start()
            # return {"race_start": True}

    async def session_checker_node(self,state: RaceEngineerState):
        # print("The session type is: ", state["session_type"])
        # print(state["messages"],state["data"])
        if state["messages"][-1].content:
            print("this one is activated")
            return {"active": True}
        
        else:
            return {"active": False}
        # if state["session_type"] == 1:
        #     if state["query"]:
        #         print("Practice is started, question asked")
        #         return {"query": state["query"]}
        #     # else:
        #         # print("no question")
        # elif state["session_type"] == 2:
        #     if state["query"]:
        #         print("Race is started, question asked")
        #         return {"query": state["query"]}
        #     # else:
        #     #     print("no question")
        # elif state["session_type"] == 3:
        #     if state["query"]:
        #         print("Race is started, question asked")
        #         return {"query": state["query"]}
        #     else:
        #         print("no question")
        # elif state["session_type"] == 4:
        #     if state["query"]:
        #         print("Race is started, question asked")
        #         return {"query": state["query"]}
            # else:
            #     print("no question")

    async def idle_node(self,state:RaceEngineerState):                
        events = []
        return {"events": events}

    async def practice_session_node(self,state: RaceEngineerState):
        events = ["fastest_lap_set","sector_comparison"]
        # print("practise")
        return {"events": events}

    async def qualifying_session_node(self,state: RaceEngineerState):
        events = ["setup_reminder","fastest_lap_set","sector_comparison","fuel"]
        return {"events": events}

    async def race_preparation_node(self,state: RaceEngineerState):
        events = ["setup_reminder","fuel","strategy"]
        return {"events": events}

    async def race_session_node(self,state: RaceEngineerState):
        events = ["pitstop","fastest_lap_set","flags","race_summary","strategy"]
        return {"events": events}

    async def event_handling_node(self,state: RaceEngineerState):
        # Handle events based on session type and specific events
        # if state["race"]:
        #     print("It is a race")
        #     return {"data": [{"drivername": "Balazs", "laptime": "1:26.3"}]}
        # elif state["race_prep"]:
        #     print("It is a race_prep")
        # elif state["qualifying"]:
        #     print("It is a qualifying")
        # elif state["practice"]:
        #     print("It is a practice")
        # else:
        #     print(f"error in {event_handling_node}")
        if not self.fastest_lap_event.active:
            self.fastest_lap_event.start()

        for event in state["events"]:
            match event:
                case []:
                    return {"triggered": False}
                
                case "fastest_lap_set":
                    driver_info = self.fastest_lap_event.check_event()
                    if driver_info:
                        drivername = driver_info.drivername
                        laptime = driver_info.best_lap
                        sector_times = driver_info.sector_times

                        # Handle the event, e.g., log or pass to the next node
                        # print(f"Fastest Lap Event! Driver: {drivername}, Laptime: {laptime}, Sector Times: {sector_times}")
                        # Reset the fastest driver after handling the event
                        self.fastest_lap_event.latest_fastest_driver = None
                        
                        return {"event_type": str(event),"event_details": [f"Fastest lap: {laptime}",f"Driver: {drivername}"],"triggered": True}
                    else: return {"triggered": False}
                case "sector_comparison":
                    driver_info,my_info = self.fastest_lap_event.sector_comparison()
                    if driver_info:
                        drivername = driver_info.drivername
                        laptime = driver_info.best_lap
                        sector_times = driver_info.sector_times
                        my_sectors = my_info.sector_times

                        # Handle the event, e.g., log or pass to the next node
                        print(f"Fastest Lap Event! Driver: {drivername}, Laptime: {laptime}, Sector Times: {sector_times}")
                        # Reset the fastest driver after handling the event
                        self.fastest_lap_event.compare_lap = None
                        return {"event_type": str(event),"event_details": [f"Driver: {drivername}",f"Sector times: {sector_times}",f"My sector times: {my_sectors}"],"triggered": True}
                    else: return {"triggered": False}
            
            return {"triggered": False}    

    async def llm_response_node(self,state: RaceEngineerState):

        if state["active"]:
            # Central node for user interaction
            system= f"""
                You are a professional Race Engineer to me Balazs Szakter. We are currently in a race. The user asks about telemetry data available, or something else.

                # Instructions:
                1. You must respond how a race engineer would in the race in a brief consise manner.
                2. Based on the query above an agent might have gathered data or not. 
                3. If data was gathered summarize the data how a race engineer would.
                2. If no data was gathered below, then respond to the user query above.
                
                # Data gathered by agent:
                {state["data"]}

                # Response:
            """
            time.sleep(1.1)
            messages = state["messages"]
            messages = [SystemMessage(content=system)] + messages
            response = await self.model_response.ainvoke(input=messages)
            print(response.content)
            await self.speak_text(response.content)
            return {"active": False}

        elif state["triggered"]:
            print(state["event_type"])
            match state ["event_type"]:
                case "fastest_lap_set":
                    event = state["event_details"]
                    # Central node for user interaction
                    system= f"""
                        You are a professional Race Engineer to me Balazs Szakter. We are currently in a session. Someone set their fastest lap of the session.

                        # Instructions:
                        1. Tell the user who did the fastest lap and what laptime they did in a coherent response.
                        2. Only tell the information and nothing else.
                        3. Always convert float type time to minutes and seconds.
                        
                        # Data explanation:
                        "Driver: ": The name of the driver
                        "Fastest lap: ": The laptime of the driver, it is a type "float" so be sure to convert it to time (e.g 89.123 seconds to 1:29.123)

                        # Incoming data:
                        "{event}"

                        # Response:
                    """
                    time.sleep(1.1)
                    messages = state["messages"]
                    messages = [SystemMessage(content=system)] + messages
                    response = await self.model_response.ainvoke(input=messages)
                    print(response.content)
                    await self.speak_text(response.content)

                case "sector_comparison":
                    event = state["event_details"]
                    # Central node for user interaction
                    system= f"""
                        You are a professional Race Engineer to me Balazs Szakter. We are currently in a session. Someone set a faster laptime than us.

                        # Instructions:
                        1. Tell the user what sectors are faster and what sectors are slower than our best sectors.
                        2. Tell it with the exact margin between the sectors.
                        
                        # Data explanation:
                        "Driver: ": The name of the driver
                        "Sector times: ": The sector times of the driver
                        "Our sector times: " Our sector times for comparison

                        # Incoming data:
                        "{event}"

                        # Response:
                    """
                    time.sleep(1.1)
                    messages = state["messages"]
                    messages = [SystemMessage(content=system)] + messages
                    response = await self.model_response.ainvoke(input=messages)
                    print(response.content)
                    await self.speak_text(response.content)
            
    async def speak_text(self,text):
        """Speak the text using text-to-speech."""
        response = self.t2s_client.text_to_speech.convert(
            voice_id="pNInz6obpgDQGcFmaJgB",  # Voice ID
            output_format="mp3_22050_32",
            text=text,
            model_id="eleven_multilingual_v2",
        )

        audio_stream = BytesIO()
        for chunk in response:
            audio_stream.write(chunk)
        audio_stream.seek(0)

        self.mixer.music.load(audio_stream, 'mp3')
        self.mixer.music.play()

    async def run_query(self,question):
        messages = [HumanMessage(content=question)]
        await self.race_engineer_agent.ainvoke({"session_type": api.read.session.session_type(), "messages": messages})

    async def start_interface(self):
        """Main loop for handling user input in the terminal."""
        
        await self.speak_text("radio check!")
        try:
            while True:
                if api is not None:
                    print(api)
                    self.session_type = api.read.session.session_type()
                    if self.wheel.joystick:
                        if self.wheel.check_button_press() and not self.s2t.listening:
                            self.s2t.listening = True
                            query = self.s2t.transcribe_speech()

                            if query:
                                print(f"Recognized Speech: {query}")
                                await self.run_query(query)
                        
                        elif not self.wheel.check_button_press() and self.s2t.listening:
                            self.s2t.listening = False
                            
                        
                        else:
                            query = ""
                            await self.run_query(query)

                pygame.time.wait(100)  # Poll every 100ms

        except KeyboardInterrupt:
            print("Stopping...")
            self.fastest_lap_event.stop()
            self.data_client.stop()
        
        # except Exception as e:
        #     print(f"problem: {e}")
        finally:
            pygame.quit()

    async def stop_interface(self):
        self.fastest_lap_event.stop()
        self.data_client.stop()


# question="What is my best lap?"
# asyncio.run(run_query(question))
agent = RaceEngineerAgent()