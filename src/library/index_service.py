import sys, os
from pathlib import Path
from fastapi import FastAPI, BackgroundTasks
from starlette.concurrency import run_in_threadpool
from starlette.config import Config
from starlette.datastructures import CommaSeparatedStrings
import asyncio
import traceback

from typing import Any
import subprocess
import json

app = FastAPI()
job_id=0
import asyncio
import json
HOME_PATH = Path.home() / '.local/share/AllTheWorld/owl_data/'
HOME_PATH.mkdir(parents=True, exist_ok=True)

class PersistentQueue(asyncio.Queue):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.persistence_file = HOME_PATH / "indexing_service_queue.json"
        self._items_list = []  # Mirror list for serialization
        self.lock = asyncio.Lock()
        self.serialize_lock = asyncio.Lock()
        
    async def put(self, item):
        print(f'put acquiring lock')
        await self.lock.acquire()
        try:
            await super().put(item)
            self._items_list.append(item)  # Keep the list in sync
        finally:
            print(f'put releasing lock')
            self.lock.release()
            
    async def get(self):
        item=None
        item = await super().get()
        try:
            print(f'get acquiring lock')
            await self.lock.acquire()
            self._items_list.remove(item)  # Keep the list in sync
        finally:
            print(f'get releasing lock')
            self.lock.release()
        return item

    async def serialize(self):
        print(f'serialize acquiring lock')
        await self.serialize_lock.acquire()
        try:
            entries = [json.dumps(item) for item in self._items_list]
            with open(self.persistence_file, "w") as f:
                json.dump(entries, f)
        finally:
            print(f'serialize releasing lock')
            self.serialize_lock.release()

    async def deserialize(self):
        print(f'serialize acquiring lock')
        await self.serialize_lock.acquire()
        try:
            with open(self.persistence_file, "r") as f:
                entries = json.load(f)
                for item in entries:
                    await self.put(json.loads(item))  # Use put to keep the list in sync
        finally:
            print(f'serialize releasing lock')
            self.serialize_lock.release()

            
task_queue = PersistentQueue()

@app.on_event("startup")
async def startup_event():
    global task_queue
    #config = Config(".env")  # Load configuration from a .env file
    config = Config()  # Load configuration from a .env file
    # make sure we only run 1 job at a time
    max_threads = config("MAX_THREADS", cast=int, default=1)
    await task_queue.deserialize()
    print(f'Index Service Starting, {task_queue.qsize()} jobs in queue')

@app.post("/submit_paper/")
async def submit_paper(paper: str, background_tasks: BackgroundTasks):
    global task_queue, job_id
    job_id = job_id + 1
    await task_queue.put({"id": job_id, "type": "dict", "paper": paper})
    await task_queue.serialize()
    return {"message": "Job submitted", "job_id": job_id}

@app.post("/submit_url/")
async def submit_url(url: str, background_tasks: BackgroundTasks):
    global task_queue, job_id
    job_id = job_id + 1
    await task_queue.put({"id": job_id, "type": "url", "url": url})
    await task_queue.serialize()
    return {"message": "Job submitted", "job_id": job_id}

@app.post("/submit_file/")
async def submit_file(filepath: str, background_tasks: BackgroundTasks):
    global task_queue, job_id
    job_id = job_id + 1
    await task_queue.put({"id": job_id, "type": "file", "filepath": filepath})
    await task_queue.serialize()
    return {"message": "Job submitted", "job_id": job_id}

async def process_tasks():
    global task_queue
    while True:
        spec = await task_queue.get()  # Wait for tasks
        print(f'Starting new task {spec}')
        try:
            if type(spec)  is str:
                spec = json.loads(spec)
                
            if spec['type'] == 'url':
                try:
                    await run_in_threadpool(s2.index_url, spec['url'])
                except Exception as e:
                    print(f"fail to url file {spec}\n{str(e)}")
                    traceback.print_exc()
            if spec['type'] == 'dict':
                try:
                    paper_json = json.loads(spec['paper'])
                    await run_in_threadpool(s2.index_dict, paper_json)
                except Exception as e:
                    print(f"fail to parse paper descriptor as json {spec['paper']}\n{str(e)}")
                    traceback.print_exc()
            if spec['type'] == 'file':
                try:
                    await run_in_threadpool(s2.index_file, spec['filepath'])
                except Exception as e:
                    print(f"fail to index file {spec}\n{str(e)}")
                    traceback.print_exc()
        except Exception as e:
            traceback.print_exc()
        task_queue.task_done()
        await task_queue.serialize()

asyncio.create_task(process_tasks())
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import chat.OwlCoT as cot
cot = cot.OwlInnerVoice(None)
# set cot for rewrite so it can access llm
import library.semanticScholar3 as s2
s2.cot = cot
s2.rw.cot = cot


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=5006)
