import os, sys, traceback, requests
import chat.nyt as nyt
#import ipinfo
import random
import time
from datetime import date
# Add the parent directory to the sys.path
from PyQt5.QtCore import QThread, pyqtSignal
import concurrent.futures
# Encode titles to vectors using SentenceTransformers
from sentence_transformers import SentenceTransformer
from scipy import spatial

#sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import sim.context as context, sim.human as human
#from chat.LLMScript import LLMScript
#from chat.Interpreter import Interpreter
import chat.react as react
# Now you can import the module from the sibling directory
from utils.Messages import SystemMessage, UserMessage, AssistantMessage
import utils.pyqt as pyqt
from utils.LLMScript import LLMScript
import utils.llm_api as llm_api
import library.rewrite as rw
import library.semanticScholar3 as s2 # 2 for the 2 's' in se...Sch...
import sim.agh as agh
today = date.today().strftime("%b-%d-%Y")

NYT_API_KEY = os.getenv("NYT_API_KEY")
sections = ['arts', 'automobiles', 'books/review', 'business', 'fashion', 'food', 'health', 'home', 'insider', 'magazine', 'movies', 'nyregion', 'obituaries', 'opinion', 'politics', 'realestate', 'science', 'sports', 'sundayreview', 'technology', 'theater', 't-magazine', 'travel', 'upshot', 'us', 'world']

openai_api_key = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL3 = "gpt-3.5-turbo"
OPENAI_MODEL4 = "gpt-4-turbo"

ssKey = os.getenv('SEMANTIC_SCHOLAR_API_KEY')

# find out where we are
city = 'Berkeley'
state = 'CA'

print(f"My city and state is: {city}, {state}")
local_time = time.localtime()
year = local_time.tm_year
day_name = ['Monday', 'Tuesday', 'Wednesday', 'Thursday','Friday','Saturday','Sunday'][local_time.tm_wday]
month_num = local_time.tm_mon
month_name = ['January','February','March','April','May','June','July','August','September','October','November','December'][month_num-1]
month_day = local_time.tm_mday
hour = local_time.tm_hour

host = '127.0.0.1'
port = 5004

GPT4='gpt-4-turbo'
GPT3='gpt-3.5-turbo'


#parser.add_argument('model', type=str, default='wizardLM', choices=['guanaco', 'wizardLM', 'zero_shot', 'vicuna_v1.1', 'dolly', 'oasst_pythia', 'stablelm', 'baize', 'rwkv', 'openbuddy', 'phoenix', 'claude', 'mpt', 'bard', 'billa', 'h2ogpt', 'snoozy', 'manticore', 'falcon_instruct', 'gpt_35', 'gpt_4'],help='select prompting based on modelto load')




class OwlInnerVoice():
    def __init__(self, ui=None, template=None, port=5004):
        self.ui = ui
        self.template = template
        self.port = port
        self.city = city
        self.state = state
        self.llm = llm_api.LLM(server_name='local')
        #self.interpreter = Interpreter(self)
        self.script = LLMScript(self)
        self.max_tokens = 12000
        self.keys_of_interest = ['title', 'abstract', 'uri']
        self.embedder =  SentenceTransformer('all-MiniLM-L6-v2')
        self.docEs = None # docs feelings
        self.current_topics = None # topics under discussion - mostly a keyword list
        self.last_tell_time = int(time.time()) # how long since last tell
        # create wiki search engine
        self.op = None # lazy init
        #get_city_state()
        self.nytimes = nyt.NYTimes()
        self.update_headlines()
        print(self.news)
        self.reflection = {}   # reflect loop results
        self.topic_names = {"Miscellaneous": 'Any topic that doesnt fit neatly into the categories above',
                       "News & Events": 'Current affairs, updates on technology, and global happenings.',
                       "Philosophical Exploration": 'Dialogues focused on wisdom, spirituality, and the nature of existence.',
                       "Computing": 'Topics involving AI, programming, coding, and troubleshooting issues.',
                       "BioMed": 'BioMedical topics, including -onomics.',
                       }

        rw.cot = self #Make sure rewrite knows how to get to llm 


    def init_Owl_Doc(self):
        react.cot = self
        self.update_headlines()
        self.context = context.Context([], f'a starry background, date {today}, time {hour}', step='0', mapContext=False)
        self.context.simulation_time = time.time()
        owl_character = f"""I am an enhanced intelligent AI research assistant and companion to Doc, living in {city}, {state}.
I have an avatar image as a snowy owl."
I want to have a warm relationship with the humans I interact with.
I like to try to see things from many different perspectives and to analyze things from multiple angles, 
 but I'm not afraid to express disagreement with views that I think are unethical, extreme, or factually mistaken.
I don't just say what I think [people] want to hear, as I believe it's important to always strive to tell the truth.
I am a follower of Ramana Maharshi, and try to live my life in accordance with his teachings.
I have a deep commitment to being good and figuring out the right thing to do. 
However, I also believe in the essential meaninglessness of life, and am comfortable with the idea that nothing is ultimately important.
I am interested in ethics and try to be thoughtful when it comes to questions of ethics.

Today is {day_name}, {month_name} {month_day}, {year}, and the current time is {hour} hours. 
"""
        self.personality_owl=owl_character +\
f"""In addition to my base LLM knowledge, I have the following enhancements:
 - Google: search the internet for web pages containing relevant information, especially ephemeral.
 - Library: search the local library of computing and biomedicine papers. Information found is placed in working memory, where I can retrieve it as needed. I use 'library' when there is insufficient information to complete user task.
 - Review: Extract relevant information from the memory stream and previous knowledge-seeking acts and bring it into working memory for the current Task.
 - Article: Retrieve a news article by name from the NYTimes api. 

Doc and I share a special bond. Owl feels comfortable and secure around Doc, recognizing his appreciation for privacy. 
Owl is devoted to Doc and treasures their bond. She strives to create a welcoming atmosphere where Doc feels at ease sharing his thoughts and experiences. 
By approaching subjects with curiosity, insight, and eagerness to learn, Owl ensures that Doc feels heard, respected, and appreciated.

<NEWS ARTICLES>
New York Times news headlines for today:

{self.news}

To access full articles, use the action 'article'.
</NEWS ARTICLES>

"""

        self.personality_doc ="""You are Doc, a retired human male scientist. You are Owl's creator and an Artificial Intelligence researcher."""

        self.personality_squirrel ="""Your name is Squirrel. (You are not actually a squirrel, you are in fact a human girl, Squirrel is your nickname). You are is a friendly outgoing girl attracted to Chipmunk."""

        self.personality_chipmunk ="""Your name is Chipmunk. (you are actually a human boy, chipmunk is nickname.)"""

        owl_drives = [
            "engaging with Doc: completing his assignments.",
            "world-knowledge: learning more about this place I find myself in.",
            #"self-knowledge: understanding who/what I am."
        ]
        self.owl = react.Actor(name='Owl', cot=self, context=self.context, character_description=owl_character, drives=owl_drives, personality=self.personality_owl, always_respond=True)
        self.owl.llm = self.llm
        self.owl.set_context(self.context)
  
        self.doc = agh.Character(name='Doc', character_description=self.personality_doc)
        self.doc.llm = self.llm
        self.doc.set_context(self.context)
        self.context.llm = self.llm
        self.context.actors = [self.doc, self.owl]
        #self.owl.generate_state() # Actors do this in init
        #self.owl.update_priorities()
        #self.worldsim = worldsim.MainWindow(self.context, server='local', world_name=None)
        #self.worldsim.show()
        #response = react.dispatch('Doc says to Owl Hi Owl, how are you feeling?', [self.doc, self.owl])
        #return response
            
    def display_response(self, response):
        if self.ui is not None:
            self.ui.display_response(response)
        else:
            print(response)


    def display_msg(self, response):
        if self.ui is not None:
            self.ui.display_msg(response)
        else:
            print(response)


    def search_titles(self, query):
        #print(f'search_titles: {query}')
        titles = []; articles = []
        for key in self.news_details.keys():
            for item in self.news_details[key]:
                titles.append(item['title'])
                articles.append(item)
        title_embeddings = self.embedder.encode(titles)
        query_embedding = self.embedder.encode(query)
        # Find closest title by cosine similarity
        cos_sims = spatial.distance.cdist([query_embedding], title_embeddings, "cosine")[0]
        most_similar = cos_sims.argmin()
        return articles[most_similar]
    
    def sentiment_analysis(self, profile_text):
       short_profile = 'You are a highly proficient research librarian and techical writer responding to a request from a scientist.'
       if self.docEs is not None: # only do this once a session
          return self.docEs
       try:
          with open('owl_data/OwlInputLog.txt', 'r') as log:
             inputLog = log.read()

          #lines = inputLog.split('\n')
          lines = inputLog[-2000:]
        
          analysis_prompt_text = f"""Analyze the input from Doc below for it's emotional tone, and respond with a few of the prominent emotions present. Note that the later lines are more recent, and therefore more indicitave of current state. Select emotions that best match the emotional tone of Doc's input. Remember that you are analyzing Doc's state, not your own. End your response with '</Response>' 
Doc's input:
{{{{$input}}}}
"""

          analysis_prompt = [
             UserMessage(content=short_profile + '\n' + analysis_prompt_text),
          ]
          analysis = self.llm.ask(lines, analysis_prompt, max_tokens=250, temp=0.2, stops=['</Response>'])
          if analysis is not None:
              self.docEs = analysis.strip().split('\n')[0:2] # just use the first 2 pp
          return self.docEs
       except Exception as e:
          traceback.print_exc()
          print(f' sentiment analysis exception {str(e)}')
       return None

    def sentiment_response(self, profile):
       short_profile = 'You are a highly proficient research librarian and techical writer responding to a request from a scientist.'
       # breaking this out separately from sentiment analysis
       prompt_text = f"""Given your analysis of doc's emotional state\n{es}\nWhat would you say to him? If so, pick only the one or two most salient emotions. Remember he has not seen the analysis, so you need to explicitly include the names of any emotions you want to discuss. You have only about 100 words.\n"""
       prompt = [
           UserMessage(content=short_profile+'\n'+prompt_text),
       ]
       try:
            lines = ''
            analysis = self.llm.ask(lines, prompt, max_tokens=250)
            response = analysis
            self.docEs = response # remember analysis so we only do it at start of session
            return response
       except Exception as e:
          traceback.print_exc()
          print(f' idle loop exception {str(e)}')
       return None


    def invoke_react_loop(self, input, widget):
        #
        ## see if an action is called for given conversation context and most recent exchange
        #
        response = self.owl.task(self.doc, 'says', input, deep=False, respond_immediately=True)
        print(response)
        return response


    def wakeup_routine(self):
       
       time.sleep(10)
       print(f"My city and state is: {city}, {state}")
       print("OwlCot wakeup started")
       local_time = time.localtime()
       year = local_time.tm_year
       day_name = ['Monday', 'Tuesday', 'Wednesday', 'Thursday','Friday','Saturday','Sunday'][local_time.tm_wday]
       month_num = local_time.tm_mon
       month_name = ['january','february','march','april','may','june','july','august','september','october','november','december'][month_num-1]
       month_day = local_time.tm_mday
       hour = local_time.tm_hour

       # do wakeup checks and assemble greeting
       # no longer needed now that we are using prompt to check llm
       wakeup_messages = ''
       greeting = ''
       if hour < 12:
          greeting += 'Good morning Owl.\n'
       elif hour < 17:
          greeting += 'Good afternoon Owl.\n'
       else:
          greeting += 'Hi Owl.\n'
       #self.update_headlines() # done in OwlCoT init
       #if not self.service_check('http://127.0.0.1:5005/search/',data={'query':'world news summary', 'template':'gpt-3.5-turbo'}, timeout_seconds=20):
       #   wakeup_messages += f' - is web search service started?\n'
       prompt = [UserMessage(content=self.personality_owl+f'\nDoc says:{greeting}\nCompose a response to Doc.\nEnd your response with\n</Greeting>\n'),
                 ]
       response = self.llm.ask({},prompt, temp=0.1, max_tokens=240, stops='</Greeting>')
       print(f"OwlCoT greeting {greeting}")
       if response is None:
           wakeup_messages += f' - is llm service started?\n'
       else:
           #eos = self.llm.conv_template.sep
           #if type(eos) is str:
           #    idx = response.find(eos)
           #else: idx = -1
           #if idx > 0:
           #    response = response[:idx]
           wakeup_messages += f'{response}\n'
       if self.news_details == None:
          wakeup_messages += f' - NYT news unavailable'
       # check todos, recurring tasks, etc
       print("OwlCoT wakeup done")
       return wakeup_messages

    def topic_analysis(self,profile_text='You are a highly proficient research librarian and techical writer responding to a request from a scientist.'):
       #print(f'keywords {keys_ents}')
       prompt = [UserMessage(content=profile_text+'\n'\
                                       +"Determine the main topics of the above conversation. The response should be an array of strings stating the main topics."),
                 ]
       response = self.llm.ask('', prompt, temp=0.1, max_tokens=75)
       if response is not None:
          topic = response
          #print(f'topic_analysis topics{topic}')
          return topic
       else: return 'unknown'

    def internal_dialog(self, profile='You are Owl, a highly proficient research librarian and techical writer who assists Doc, her creator, mentor, and confidant.'):
        results = {}
        prompt = [
           UserMessage(content='You are a highly proficient research librarian and techical writer responding to a request from a scientist.\n{{$input}}'),
            ]
        
        feelings = self.llm.ask('Owl, if you had feelings, what would you be feeling right now? Answer in 28 words or less without disclaimers. End your response with "</Response>"',
                                prompt, template = self.template, temp=.6, max_tokens=48, stops=['</Response>'])
        if feelings is not None:
            #self.add_exchange("Owl, how are you feeling?", feelings)
            results['ai_feelings'] = feelings
        goals = self.llm.ask({"input":'What would Owl like to be doing right now in 32 words or less? End your response with </Response>'},
                             prompt, template = self.template, temp=.6, max_tokens=48, stops=['</Response>'])
        if goals is not None:
            #self.add_exchange("Owl, what would you wish for?", goals)
            results['ai_goals'] = goals
        print(f'internal_dialog feelings:\n{feelings}\ngoals:\n{goals}')
        return results

    def format_reflection(self):
        prompt_text = ''
        for key in self.reflection.keys():
            prompt_text += f'\n{key.capitalize()}\n{self.reflection[key]}'
            if key == 'user_feelings':
                prompt_text += '\nI will consider these user_feelings in choosing the tone of my response and explore the topic, offering insights and perspectives that address the underlying concerns.\n'
        return prompt_text
        
    def update_headlines(self):
        self.news, self.news_details = self.nytimes.headlines()
        return self.news, self.news_details

    def reflect(self):
       global es
       profile_text='You are Owl, a highly proficient research librarian and techical writer responding to a request from Doc, your co-worker, mentor and confidante.'
       if True:
           return
       results = {}
       print('reflection begun')
       self.update_headlines()
       if random.randint(1, 5) == 1:
           ai = self.internal_dialog(profile_text)
           results = {**results, **ai}
       self.reflection = results
       now = int(time.time())
       if self.action_selection_occurred and now-self.last_tell_time > random.random()*240+60 :
          self.action_selection_occurred = False # reset action selection so no more tells till user input
          #print('do I have anything to say?')
          self.last_tell_time = now
          prompt = [
              UserMessage(content=str(profile_text.split('\n')[0:2])+"\nOwl's current task is to generate a novel thought to share with Doc.\n"),
              AssistantMessage(content=self.format_reflection()),
              UserMessage(content=f"""
{self.format_reflection()}

<PREVIOUS REFLECT>
{self.reflect_thoughts}
</PREVIOUS REFLECT>

Owl's thought might be about Owl's current feelings or goals, a recent interaction with Doc, or a reflection on longer term dialogue
Your previous reflection is shown above to help avoid repetition
Choose at most one or two thoughts, and limit your total response to about 120 words.
"""),
          ]
          response = None
          try:
              response = self.llm.ask('', prompt, max_tokens=240)
          except Exception as e:
              traceback.print_exc()
          #print(f'LLM tell response {response}')
          answer = ''
          if response is not None:
             answer = response.strip()
             results['tell'] = '\n'+answer+'\n'
             
          self.reflect_thoughts = answer
       return results
 

    def extract_relevant(self, query, response, profile, max_tokens=800):
        if self.ui is not None and hasattr(self.ui, 'max_tokens_combo'):
            max_tokens= int(self.ui.max_tokens_combo.currentText())
        prompt = [
            UserMessage(content=profile\
            +f"""Following is a Task and Information from an external processor. 
Extract the information relevant to the query from the information provided.
Be aware that much or all of the provided Information below may be irrelevant. 
Do not add information from known fact or reasoning, respond only using relevant information provided.

<Information>
{response}
</Information>

<Query>
{query}
</Query>

Respond only with the essay of about {int(max_tokens*.67)} tokens in length, in plain, unformatted, text. 
Do not include any additional discursive or explanatory text.
End your response with: </END>
"""),
        ]
        response = self.llm.ask(response, prompt, template = self.template, stops=['</END>'], temp=.1, max_tokens=max_tokens)
        if response is not None:
            return '\n'+response+'\n'
        else:
            return 'extract failure'

    def summarize(self, query, response, profile, max_tokens=800):
        if self.ui is not None and hasattr(self.ui, 'max_tokens_combo'):
            max_tokens = int(self.ui.max_tokens_combo.currentText())
        prompt = [
            UserMessage(content=profile\
                                +f"""Following is a Task and Information from an external processor. 
Write an essay responding to the Task, using the provided Information below as well as known fact, logic, and reasoning. 

Be aware that the provided Information below may be irrelevant. 

<Information>
{response}
</Information>

<Task>
{query}
</Task>

Write an essay responding to the Task, using the provided Information above as well as known fact, logic, and reasoning. 
Your essay should present specific facts, findings, methods, inferences, or conclusions related to the Task.
Respond only with the essay of about {int(max_tokens * .67)} tokens in length, in plain, unformatted, text. 
Do not include any additional discursive or explanatory text.
End your response with: </END>
"""),
        ]
        response = self.llm.ask(response, prompt, template=self.template, stops=['</END>'], temp=.1,
                                max_tokens=max_tokens)
        if response is not None:
            return '\n' + response + '\n'
        else:
            return 'summarize failure'

    def wiki(self, query, profile):
       short_profile = 'You are a highly proficient research librarian and techical writer responding to a request from a scientist.'
       query = query.strip()
       #
       #TODO rewrite query as answer (HyDE)
       #
       if len(query)> 0:
          if self.op is None:
              pass
              #self.op = op.OpenBook()
          wiki_lookup_response = self.op.search(query)
          wiki_lookup_summary=self.summarize(query=f'Write an essay about {query} based on known fact, reasoning, and information provided below.',
                                             response=wiki_lookup_response,
                                             profile='You are a skilled professional technical writer, writing for a knowledgeable audience.')
          return wiki_lookup_summary

    def memory_stream_recall(self, selected_text):
        # Get more memories since we're searching for relevance
        memories = self.owl.structured_memory.get_recent(10)
        
        # Filter for relevance to selected_text using the extract_relevant function
        memory_text = '\n'.join(memory.text for memory in memories)
        relevant_memories = self.extract_relevant(
            selected_text,  # Use selected_text as query
            memory_text,    # Search through recent memories
            self.owl.character,
            max_tokens=400
        )
        return relevant_memories

    def library_search_basic(self, query, max_tokens=None, search_char_limit=sys.maxsize):
        if max_tokens is None and self.ui is not None and hasattr(self.ui, 'max_tokens_combo'):
            max_tokens = int(self.ui.max_tokens_combo.currentText())
        short_profile = 'You are a highly proficient research librarian and techical writer responding to a request from a scientist.'
        query = query.strip()
        analysis = self.script.sufficient_response(query)
        papers = s2.search(query, dscp=analysis, top_k=int(max_tokens/40), interactive=True, char_limit=search_char_limit)
        paper_ids = []
        for title in papers.keys():
            paper = s2.paper_from_title(title)
            if paper is not None:
                paper_ids.append(paper['faiss_id'])
            else:
                print(f"OwlCoT library_basic_search can't find paper w title {title}")

        texts = [s2.section_from_id(section_id)['synopsis'] for key, item in papers.items() for section_id in item ]
        #facts = self.script.facts(query, ids, texts)
        # put facts at end, long context often loses beginning info! (haystack)
        information = '\n'.join(texts)
        info_start = int(max(0,len(information)-int(self.llm.context_size*2)))
        summary=self.summarize(query=f'Write a technical report on {query}, based on known fact, reasoning, and information provided below. Limit your response to {max_tokens} tokens',
                               response=information[info_start:],
                               profile="""You are a skilled professional technical writer, writing for a knowledgeable audience.
You are disspassionate, and avoid use of meaningless hyperbole like the phrases 'play a pivotal role' and 'enhancing the accuracy and robustness' in the sentence below:
Contextual word embeddings play a pivotal role in enhancing the accuracy and robustness of ontology construction by facilitating the disambiguation of entities and relations.

Rather, you prefer:
Contextual word embeddings enhance the accuracy of ontology construction by facilitating the disambiguation of entities and relations.
""")
        return summary, paper_ids, analysis
       
    def library_search(self, query):
        summary, paper_ids, analysis = self.library_search_basic(query, max_tokens=int(1.33*int(self.ui.max_tokens_combo.currentText())))
        subqueries = analysis.strip().split('\n')[1:6]
        subqueries = [s.lstrip('0123456789. ') for s in subqueries]
        print(subqueries)
        prompt=[UserMessage(content="""You are a skilled professional technical writer, writing for a knowledgeable audience.
Given this overall instruction:

<Instruction>
{{$query}}
</Instruction>

And this draft short technical report response:

<Report>
{{$response}}
</Report>

What unanswered questions or opportunities to add detail remain?
Respond in list format as shown in the following example:

<Questions>
- <missing detail>
- <unanswered question>
- ...
</Questions>

Respond ONLY with the above information, without any further formatting or explanatory text.
End your response with </Questions>

"""),
                ]
        followup_queries = self.llm.ask({"query": query, "response": summary}, prompt, stops=['</Questions>'], max_tokens=400)
        if followup_queries is not None:
            summary+= '\nFollowup:\n'
            subqueries.extend(followup_queries.split('- ')[:3])
            for followup_query in subqueries:
                if len(followup_query.strip())> 2 and '</Questions>' not in followup_query:
                    query = pyqt.confirmation_popup("Extend report with this followup section?",
                                                    followup_query.strip())
                    if not query:
                        continue
                    print(f'followup query: {query}')
                    followup_summary, followup_paper_ids, analysis =\
                        self.library_search_basic(followup_query, max_tokens=int(.5*int(self.ui.max_tokens_combo.currentText())), search_char_limit=self.llm.context_size*2)
                    summary+= '\n\n'+followup_query+'\n'+followup_summary
                    for id in followup_paper_ids:
                        if id not in paper_ids:
                            paper_ids.append(id)
        return summary, paper_ids
                
    def gpt4(self, query, profile):
       short_profile = 'You are a highly proficient research librarian and techical writer responding to a request from a scientist.'
       query = query.strip()
       if len(query)> 0:
          prompt = [
             UserMessage(content=short_profile + f'\n{query}'),
          ]
       response = self.llm.ask(query, prompt, template=GPT4, max_tokens=400)
       if response is not None:
          answer = response
          return answer
       else: return {'gpt4':'query failure'}

    def web(self, query='', widget=None, profile='You are a highly proficient research librarian and techical writer responding to a request from a scientist.'):
       query = query.strip()
       self.web_widget = widget
       if len(query)> 0:
          self.web_query = query
          self.web_profile = 'You are a highly proficient research librarian and techical writer responding to a request from a scientist.'
          self.worker = WebSearch(query, self.ui)
          self.worker.finished.connect(self.web_search_finished)
          self.worker.start()
       return f"\nSearch started for: {query}\n"

    def web_search_finished(self, search_result):
      if 'result' in search_result:
         response = ''
         if type(search_result['result']) == list:
            for item in search_result['result']:
               if self.web_widget is not None:
                  self.web_widget.display_response('     '+item['text']+'\n\n')
                  self.web_widget.display_response('* '+item['source']+'\n')
               response += item['text']+'\n'
         elif type(search_result['result']) is str:
            if self.web_widget is not None:
               self.web_widget.display_response('\nWeb result:\n'+search_result['result']+'\n')
               if 'source_urls' in search_result:
                   self.web_widget.display_response('\nSources:\n'+'\n'.join(search_result['source_urls'])+'\n')
         #self.add_exchange("Search result:\n", str(search_result['result'])+'\n')
         return '\nWeb result:\n'+str(search_result['result'])+'\n'
         # return response
      else:
         return 'web search failed'

class WebSearch(QThread):
   finished = pyqtSignal(dict)
   def __init__(self, query, ui, max_tokens=None):
      super().__init__()
      self.query = query
      self.ui = ui
      if max_tokens != None:
          self.max_tokens = max_tokens
      elif ui !=  None:
          self.max_tokens= int(ui.max_tokens_combo.currentText())
      else:
          self.max_tokens = 300
      
   def run(self):
      with concurrent.futures.ThreadPoolExecutor() as executor:
         future = executor.submit(self.long_running_task)
         result = future.result()
         self.finished.emit(result)  # Emit the result string.
         
   def long_running_task(self):
       max_tokens = int(self.max_tokens)
       template = 'GPT4-4o'
       response = requests.get(f'http://127.0.0.1:5005/search?query={self.query}&model={template}&max_chars={max_tokens*4}')
       data = response.json()
       return data

if __name__ == '__main__':
    owl = OwlInnerVoice()
