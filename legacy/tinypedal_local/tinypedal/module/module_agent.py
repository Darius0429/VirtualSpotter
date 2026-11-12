"""
Race Engineer Agent module
"""

import os
import time
import pygame
import logging
import asyncio
import threading
from io import BytesIO
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AnyMessage, SystemMessage
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_anthropic import ChatAnthropic
import anthropic
from typing import TypedDict, Literal, Annotated, Dict
import operator
from langgraph.graph import StateGraph, MessagesState, END
from elevenlabs import VoiceSettings
from elevenlabs.client import ElevenLabs
from dataclasses import asdict

from ._base import DataModule
from ..module_info import minfo
from ..api_control import api
from ..utilities.game_controller import GameController
from ..utilities.S2T import Speech2Text
from ..logging import StrategyCollector
from ..utilities.events import FastestLapSet

logger = logging.getLogger(__name__)

# Define the state for the graph
class RaceEngineerState(MessagesState):
    session_type: int
    events: Annotated[list[str], operator.add] = []
    event_details: Annotated[list[str], operator.add] = []
    event_type: str
    data: Annotated[list[str], operator.add] = []
    data_type: int
    messages: Annotated[list[AnyMessage], operator.add] = []
    active: bool
    triggered: bool
    race_start: bool


class Realtime(DataModule):
    """Race Engineer Agent data"""

    def __init__(self, config, module_name):
        super().__init__(config, module_name)
        
        # Initialize LLM models
        self.model_interaction = ChatMistralAI(api_key=os.getenv("MISTRAL_API_KEY"), model_name="open-mistral-7b")
        self.model_response1 = ChatAnthropic(model='claude-3-5-haiku-20241022')
        self.model_response2 = ChatMistralAI(api_key=os.getenv("MISTRAL_API_KEY2"), model_name="open-mistral-7b")
        
        # Initialize TTS
        self.t2s_client = ElevenLabs(api_key=os.environ.get("TTS_API_KEY", ""))
        
        # Initialize data client and events
        self.data_client = StrategyCollector(interval=1)
        self.fastest_lap_event = FastestLapSet()
        
        # Initialize pygame for audio
        self.mixer = pygame.mixer
        if not pygame.get_init():
            pygame.init()
        self.mixer.init()
        
        # Initialize game controller and speech-to-text
        self.wheel = GameController()
        self.s2t = Speech2Text()
        
        # Setup the graph
        self.graph = StateGraph(RaceEngineerState)
        self.start_graph = False
        self.session_type = 0
        self.setup_graph()
        
        # Thread for agent loop
        self._agent_thread = None
        self._stop_agent = threading.Event()

    def setup_graph(self):
        """Setup the LangGraph components"""
        # Add nodes to the graph
        self.graph.add_node("LLM Interaction", self.llm_interaction_node)
        self.graph.add_node("Session Checker", self.session_checker_node)
        self.graph.add_node("Idle", self.idle_node)
        self.graph.add_node("Practice Session", self.practice_session_node)
        self.graph.add_node("Qualifying Session", self.qualifying_session_node)
        self.graph.add_node("Race Preparation", self.race_preparation_node)
        self.graph.add_node("Race Session", self.race_session_node)
        self.graph.add_node("Event Handling", self.event_handling_node)
        self.graph.add_node("Data Fetching", self.fetch_data_node)
        self.graph.add_node("Data Storing", self.data_storer_node)
        self.graph.add_node("LLM Response", self.llm_response_node)
        
        # Set entry point
        self.graph.set_entry_point("Session Checker")
        
        # Define edges
        self.graph.add_edge("Session Checker", "LLM Interaction")
        
        self.graph.add_conditional_edges("Session Checker", lambda state: state["session_type"], {
            0: "Idle",
            1: "Practice Session",
            2: "Qualifying Session",
            3: "Race Preparation",
            4: "Race Session"
        })
        
        self.graph.add_conditional_edges("LLM Interaction", lambda state: state["data_type"], {
            "2": "Data Fetching",
            "1": "Data Fetching",
            "3": "LLM Response",
            "0": END
        })
        
        self.graph.add_edge("Practice Session", "Event Handling")
        self.graph.add_edge("Qualifying Session", "Event Handling")
        self.graph.add_edge("Race Preparation", "Event Handling")
        self.graph.add_edge("Race Session", "Event Handling")
        self.graph.add_edge("Race Session", "Data Storing")
        
        self.graph.add_edge("Data Fetching", "LLM Response")
        
        self.graph.add_conditional_edges("Event Handling", lambda state: state["triggered"], {
            True: "LLM Response",
            False: END
        })
        
        self.graph.add_edge("LLM Response", END)
        
        # Compile the graph
        self.race_engineer_agent = self.graph.compile()

    async def llm_interaction_node(self, state: RaceEngineerState):
        """Handle LLM interactions to classify user query"""
        if state["active"]:
            system = """
                You are a professional Race Engineer to me Balazs Szakter. We are currently in a race session. I am driving and I ask you questions about the others or my strategy.

                # Instructions:
                1. Your task is to classify the user prompt in 2 categories:
                    "1": if the user question is about strategy. Keywords: fuel, pit, pitstop, tire change, fuel save
                    "2": if the user question is about timing, positional data. Keywords: leader, guy ahead, laptime, lap
                    "3": if the user question is not about the session and is more general (e.g. conversational questions). 
                2. You must respond the classified label only, nothing else!
            """
            time.sleep(1.1)
            messages = state["messages"]
            messages = [SystemMessage(content=system)] + messages
            response = await self.model_interaction.ainvoke(input=messages)
            logger.debug("LLM Interaction response: %s", response.content)
            return {"data_type": response.content}
        else:
            return {"data_type": "0"}

    async def fetch_data_node(self, state: RaceEngineerState):
        """Fetch telemetry data for user query"""

        return {"data": [self.data_client.retrieve_data(label=state["data_type"])]}

# 'elapsed_time': Time that the lap started from the start of the session.
#             'total_laps': Number of laps completed, the index of the lap
#             'pos_abs': Which position they are in when starting the lap, across all the field
#             'pos_class': Which position they are in when starting the lap, relative to the class (MORE IMPORTANT THAN 'pos_abs')
#             'time_gap': The driver relative gap to me (Balazs Szakter) at the end of the lap
#             'last_laptime': The laptime after crossing the finish line
#             'best_laptime': Their best laptime so far
#             'is_lapped': If they are lapped by the leader
#             'in_pit': If they are in the pits
#             'pit_state': If they pitted on the lap
#             'pit_count': How many pits they completed so far
    async def data_storer_node(self, state: RaceEngineerState):
        """Start data collection if not already started"""
        if not self.start_graph:
            self.start_graph = True
            self.data_client.start()

    async def session_checker_node(self, state: RaceEngineerState):
        """Check session state and user query"""
        if state["messages"] and state["messages"][-1].content:
            logger.debug("Session checker activated with query")
            return {"active": True}
        else:
            return {"active": False}

    async def idle_node(self, state: RaceEngineerState):
        """Handle idle state"""
        events = []
        return {"events": events}

    async def practice_session_node(self, state: RaceEngineerState):
        """Handle practice session state"""
        events = ["fastest_lap_set", "sector_comparison"]
        return {"events": events}

    async def qualifying_session_node(self, state: RaceEngineerState):
        """Handle qualifying session state"""
        events = ["setup_reminder", "fastest_lap_set", "sector_comparison", "fuel"]
        return {"events": events}

    async def race_preparation_node(self, state: RaceEngineerState):
        """Handle race preparation state"""
        events = ["setup_reminder", "fuel", "strategy"]
        return {"events": events}

    async def race_session_node(self, state: RaceEngineerState):
        """Handle race session state"""
        events = ["pitstop", "fastest_lap_set", "flags", "race_summary", "strategy"]
        return {"events": events}

    async def event_handling_node(self, state: RaceEngineerState):
        """Handle events based on session type"""
        if not self.fastest_lap_event.active:
            self.fastest_lap_event.start()
        
        # Process each potential event
        for event in state["events"]:
            if event == "fastest_lap_set":
                driver_info = self.fastest_lap_event.check_event()
                if driver_info:
                    drivername = driver_info.drivername
                    laptime = driver_info.best_lap
                    sector_times = driver_info.sector_times
                    self.fastest_lap_event.latest_fastest_driver = None
                    
                    logger.debug(f"Fastest lap event triggered: {drivername}, {laptime}")
                    
                    return {
                        "event_type": event,
                        "event_details": [f"Fastest lap: {laptime}", f"Driver: {drivername}"],
                        "triggered": True
                    }
            
            elif event == "sector_comparison":
                try:
                    # The error is happening here - calling sector_comparison() is returning None
                    comparison_result = self.fastest_lap_event.sector_comparison()
                    
                    # Check if we got a valid result (not None)
                    if comparison_result is None:
                        logger.debug("Sector comparison returned None")
                        continue
                        
                    # Now safely unpack the result
                    try:
                        driver_info, my_info = comparison_result
                    except (ValueError, TypeError) as e:
                        logger.error(f"Error unpacking sector comparison result: {e}")
                        continue
                    
                    # Verify both parts of the tuple contain data
                    if not driver_info or not my_info:
                        logger.debug(f"Missing data in comparison: driver_info={driver_info}, my_info={my_info}")
                        continue
                    
                    drivername = driver_info.drivername
                    laptime = driver_info.best_lap
                    sector_times = driver_info.sector_times
                    my_sectors = my_info.sector_times
                    
                    logger.debug(f"Sector comparison triggered: Driver: {drivername}, Sectors: {sector_times}, My sectors: {my_sectors}")
                    self.fastest_lap_event.compare_lap = None
                    
                    return {
                        "event_type": event,
                        "event_details": [
                            f"Driver: {drivername}",
                            f"Sector times: {sector_times}",
                            f"My sector times: {my_sectors}"
                        ],
                        "triggered": True
                    }
                except Exception as e:
                    logger.error(f"Error processing sector comparison: {str(e)}")
                    # Continue to next event on error
                    continue
            
            # Add handling for other event types...
        
        # If no events were triggered
        return {"triggered": False}

    async def run_query(self, question):
        """Run a query through the agent"""
        try:
            # Only process non-empty questions
            if question.strip():
                messages = [HumanMessage(content=question)]
                
                # Get session type safely
                session_type = 0
                if api is not None:
                    try:
                        session_type = api.read.session.session_type()
                    except Exception as e:
                        logger.error(f"Error reading session type: {str(e)}")
                
                # Run the agent
                await self.race_engineer_agent.ainvoke({
                    "session_type": session_type,
                    "messages": messages,
                    "active": True,
                    "triggered": False,
                    "events": [],
                    "event_details": [],
                    "data": [],
                    "race_start": False
                })
            else:
                # For empty queries, just check for events
                session_type = 0
                if api is not None:
                    try:
                        session_type = api.read.session.session_type()
                    except Exception as e:
                        logger.error(f"Error reading session type: {str(e)}")
                
                # Run with minimal state for event checks
                await self.race_engineer_agent.ainvoke({
                    "session_type": session_type,
                    "messages": [],
                    "active": False,
                    "triggered": False,
                    "events": [],
                    "event_details": [],
                    "data": [],
                    "race_start": False
                })
        except Exception as e:
            logger.error(f"Agent error: {str(e)}")

    async def llm_response_node(self, state: RaceEngineerState):
        """Generate and speak LLM responses"""
        # Only process one type of response at a time - giving priority to active queries
        if state["active"]:
            system = f"""
                You are a professional Race Engineer to me Balazs Szakter. We are currently in a race. The user asks about telemetry data available, or something else.

                # Instructions:
                1. You must respond how a race engineer would in the race in a brief consise manner.
                2. Based on the query above an agent might have gathered data or not. 
                3. If data was gathered summarize the data how a race engineer would.
                4. If no data was gathered below, then respond to the user query above.
                5. Keep your response under 50 words.
                
                # Data gathered by agent:
                {state["data"] if state["data"] else "No data gathered."}
            """
            try:
                time.sleep(0.5)  # Small delay to ensure state is consistent
                messages = state["messages"]
                
                # Add system message first
                system_msg = SystemMessage(content=system)
                all_messages = [system_msg] + messages
                
                # Process with LLM
                response = await self.model_response1.ainvoke(input=all_messages)
                logger.debug("LLM Response: %s", response.content)
                
                # Speak the response
                await self.speak_text(response.content)
                return {"active": False}
            except Exception as e:
                logger.error(f"Error in active response: {str(e)}")
                await self.speak_text("Sorry, I couldn't process that request.")
                return {"active": False}

        elif state["triggered"]:
            logger.debug("Event triggered: %s", state["event_type"])
            try:
                event_type = state["event_type"]
                event_details = state["event_details"]
                
                if event_type == "fastest_lap_set":
                    system = f"""
                        You are a professional Race Engineer to me Balazs Szakter. We are currently in a session. Someone set their fastest lap of the session.

                        # Instructions:
                        1. Tell the user who did the fastest lap and what laptime they did in a coherent response.
                        2. Only tell the information and nothing else.
                        3. Always convert float type time to minutes and seconds.
                        4. Keep your response under 30 words.
                        
                        # Data explanation:
                        "Driver: ": The name of the driver
                        "Fastest lap: ": The laptime of the driver, it is a type "float" so be sure to convert it to time (e.g 89.123 seconds to 1:29.123)

                        # Incoming data:
                        {event_details}
                    """
                    
                    # Create message for LLM
                    system_msg = SystemMessage(content=system)
                    empty_msg = HumanMessage(content="Process the fastest lap event")
                    
                    # Use second model for events
                    response = await self.model_response2.ainvoke(input=[system_msg, empty_msg])
                    logger.debug("Fastest lap event response: %s", response.content)
                    await self.speak_text(response.content)

                elif event_type == "sector_comparison":
                    system = f"""
                        You are a professional Race Engineer to me Balazs Szakter. We are currently in a session. Someone set a faster laptime than us.

                        # Instructions:
                        1. Tell the user what sectors are faster and what sectors are slower than our best sectors.
                        2. Tell it with the exact margin between the sectors.
                        3. Keep your response under 40 words.
                        
                        # Data explanation:
                        "Driver: ": The name of the driver
                        "Sector times: ": The sector times of the driver
                        "My sector times: " Our sector times for comparison

                        # Incoming data:
                        {event_details}
                    """
                    
                    # Create message for LLM
                    system_msg = SystemMessage(content=system)
                    empty_msg = HumanMessage(content="Process the sector comparison event")
                    
                    # Use second model for events
                    response = await self.model_response2.ainvoke(input=[system_msg, empty_msg])
                    logger.debug("Sector comparison response: %s", response.content)
                    await self.speak_text(response.content)
                
                return {"triggered": False}  # Reset trigger after processing
            
            except Exception as e:
                logger.error(f"Error in event response: {str(e)}")
                return {"triggered": False}
        
        # If neither active nor triggered
        return {}
            
    async def speak_text(self, text):
        """Speak the text using text-to-speech."""
        try:
            response = self.t2s_client.text_to_speech.convert(
                voice_id="pNInz6obpgDQGcFmaJgB",
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
        except Exception as e:
            logger.error("TTS error: %s", str(e))

    # async def run_query(self, question):
    #     """Run a query through the agent"""
    #     messages = [HumanMessage(content=question)]
    #     try:
    #         await self.race_engineer_agent.ainvoke({
    #             "session_type": api.read.session.session_type() if api else 0, 
    #             "messages": messages
    #         })
    #     except Exception as e:
    #         logger.error("Agent error: %s", str(e))

    async def agent_loop(self):
        """Main agent loop for handling controller input"""
        await self.speak_text("Radio check, engineer online!")
        try:
            while not self._stop_agent.is_set():
                try:
                    if api is not None:
                        self.session_type = api.read.session.session_type()
                        if self.wheel.joystick:
                            if self.wheel.check_button_press() and not self.s2t.listening:
                                self.s2t.listening = True
                                query = self.s2t.transcribe_speech()

                                if query:
                                    logger.info("Recognized Speech: %s", query)
                                    await self.run_query(query)
                            
                            elif not self.wheel.check_button_press() and self.s2t.listening:
                                self.s2t.listening = False
                                
                            else:
                                await self.run_query("")
                except Exception as inner_e:
                    logger.error("Error in agent loop: %s", str(inner_e))
                
                await asyncio.sleep(0.1)  # Poll every 100ms

        except Exception as e:
            logger.error("Agent loop error: %s", str(e))
        
    def agent_thread_wrapper(self):
        """Wrapper to run agent loop in a thread"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self.agent_loop())
        loop.close()

    def update_data(self):
        """Update module data"""
        reset = False
        update_interval = self.active_interval

        while not self._event.wait(update_interval):
            if self.state.active:

                if not reset:
                    reset = True
                    update_interval = self.active_interval
                    
                    # Start agent thread if not running
                    if self._agent_thread is None or not self._agent_thread.is_alive():
                        self._stop_agent.clear()
                        self._agent_thread = threading.Thread(target=self.agent_thread_wrapper, daemon=True)
                        self._agent_thread.start()
                        logger.info("Agent thread started")

            else:
                if reset:
                    reset = False
                    update_interval = self.idle_interval
                    
                    # Stop agent thread
                    if self._agent_thread and self._agent_thread.is_alive():
                        self._stop_agent.set()
                        self._agent_thread.join(timeout=2.0)
                        self._agent_thread = None
                        logger.info("Agent thread stopped")
                    
                    # Stop events and data collection
                    if self.fastest_lap_event.active:
                        self.fastest_lap_event.stop()
                    
                    if self.start_graph:
                        self.data_client.stop()
                        self.start_graph = False

    def stop(self):
        """Stop the module"""
        # Stop agent thread
        if self._agent_thread and self._agent_thread.is_alive():
            self._stop_agent.set()
            self._agent_thread.join(timeout=2.0)
            self._agent_thread = None
        
        # Stop events and data collection
        if self.fastest_lap_event.active:
            self.fastest_lap_event.stop()
        
        if self.start_graph:
            self.data_client.stop()
            self.start_graph = False
            
        # Call parent stop method
        super().stop()
