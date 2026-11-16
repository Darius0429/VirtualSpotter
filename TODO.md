 What the model sees today

  - Before any data: about 3,700 tokens of instructions and 16 tool descriptions on every question.
  - Per question: it picks one tool and gets raw output such as "energy_to_add_to_finish_percent": 51.69488588968911 next to "fuel_to_add_to_finish_l":
    32.41, with 17 fields for a simple question.
  - Result: it read back whichever field matched your words ("fuel" gave litres), and each answer needed two model calls.

  Proposal: one short race card instead of 16 tools

  The card would be put in front of the model on every question, already rounded and in your words, so most answers need one model call and no tool. These
  are the stats I'd put on it; this is the list to correct:

  - Race: lap and laps left, time left, class place.
  - Energy: percent on board, percent per lap, laps on board, pit lap, percent to add, and short or surplus to the finish.
  - Fuel: shown only if it runs out before energy.
  - Pace: average of clean laps, last lap, best lap.
  - Car ahead and car behind: name, gap, pace difference per lap, laps until the gap closes, laps of energy they have, whether they have pitted.
  - Rivals by predicted finish: name, predicted gap at the flag, stops left.
  - Traffic: the next other-class car you catch or that catches you, with lap and place on the lap.
  - Pit stop now: time lost and rejoin place.
  - Tyres: your own wear only; the game doesn't publish other cars' wear.
  - Last few timeline entries: for example "lost P2 to X on lap 34", so a remark like "he spun me" has context.

  Five tools would remain for detail: sector comparison, saving check, another car's pit stop, the timeline, and adding a note.

  Decisions for you

  1. Is the stat list right? What is missing or not needed?
  2. Voxtral through the API for speech, or stay local?
  3. Should I build the replay comparison of agent models on tonight's 12 questions?

  Sources:
  - Deepgram Nova-3 review (keyterm prompting) (https://neurlcreators.substack.com/p/nova-3-deepgram-review)
  - Best STT Providers 2026 (Coval) (https://coval.ai/blog/best-speech-to-text-providers-in-2026-independent-benchmarks-and-how-to-choose)
  - Voxtral Transcribe 2 (context biasing) (https://aitoolsatlas.ai/tools/voxtral-transcribe-2)
  - Voxtral Mini Transcribe 3 pricing (https://www.llmreference.com/model/voxtral-mini-transcribe-3/mistral-ai-api)
  - Mistral Voxtral announcement (https://mistral.ai/fr/news/voxtral)
  - Parakeet vs Whisper vs Nemotron (https://openwhispr.com/blog/parakeet-vs-whisper-vs-nemotron)
  - Offline Speech-to-Text 2026 (https://diyai.io/ai-tools/speech-to-text/offline-speech-to-text/)
  - Function calling latency comparison, May 2026 (https://callsphere.ai/blog/llm-comparison-function-calling-tool-orchestration-lowest-latency-may-2026)
  - Best LLM for function calling comparison (https://deploybase.ai/articles/best-llm-for-function-calling-tool-use-comparison)

❯ Okay now that I have run my first test I reached a conclusion. For v1 it made sense to have different agents doing different stuff. But for v2 I need one agent to respond when I ask. I want to use claude models to do reasoning agentic decisions. I want to use a lightweight classifier that determins what type of reasoning needed, low-medium-high. I want to clean the data as mentioned in a way that the tool can access it instead of raw data in very high frequency. I need to autodetect corners based on gps data like MoTeC does for example. Then I need to put my data in there like deltas for me and my opponents. This way I can track where I am faster or losing time. This should include braking point before the corner, and throttle point after, apex distance, lowest gear, minimum speed. Like MoTeC would analyse it. And then I need sector times and laptimes, fuel/energy usage. I also need strategy with my predicted racetimes based on my pace and my energy vs energy needed so is pitstop needed etc. Do you think this is structured and enough for race prediction? I would use scripts to calculate these for the models to reduce token count and also less noise and less context for the prediction models. 