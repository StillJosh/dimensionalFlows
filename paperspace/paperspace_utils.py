# paperspace_utils.py
# Description: A brief description of what this file does.
# Author: Joshua Stiller
# Date: 17.03.24


from gradient import NotebooksClient, ResourceFetchingError

from datetime import datetime
from time import sleep
import random


def start_machine(lock, machine_type, id, api_key, with_sleep = None):

    notebooks_client = NotebooksClient(api_key)
    n = datetime.now()
    while True:
        try:
            notebooks_client.start(id=id, machine_type=machine_type, shutdown_timeout=6)
            msg = f'{id} - The {machine_type} machine has been acquired for Paperspace Gradient Notebook.'
            with lock:
                print(f"The time it took to complete the request is {datetime.now() - n}")
                print(msg)
            return
        except ResourceFetchingError as e:
            if str(e).find('out of capacity') != -1:
                with lock:
                    print(f"{id} - {machine_type}: {e}")
                if with_sleep is not  None:
                    if with_sleep == "random":
                        sleep(random.random()*50)
                    if with_sleep == "constant":
                        sleep(10)
            else:
                with lock:
                    print(f"Aborting {id} - {machine_type}: {e}")
                return