# paperspace_handler.py
# Description: A brief description of what this file does.
# Author: Joshua Stiller
# Date: 17.03.24


from gradient import NotebooksClient, ResourceFetchingError

from datetime import datetime
from multiprocessing import Lock, Process
from paperspace_utils import start_machine
from dotenv import dotenv_values

if __name__ == "__main__":
    config = dotenv_values('../.env')
    PAPERSPACE_API_KEY = config['PAPERSPACE_API_KEY']

    lock = Lock()
    # The API key of your Paperspace account.
    machine_types = ['Free-GPU']
    id = 'nbyeow8rlr'  # The id of the Gradient notebook.

    #for machine_type in machine_types:
    #    Process(target=start_machine, args=(lock, machine_type, id, PAPERSPACE_API_KEY)).start()

    start_machine(lock, 'Free-GPU', id, PAPERSPACE_API_KEY, with_sleep="random")

    # Get all Notebooks
    notebooks_client = NotebooksClient(PAPERSPACE_API_KEY)
    notebooks = notebooks_client.list(tags=[])

    # Get id of running notebook
    for notebook in notebooks:
        if notebook.state != "Cancelled":
            print(notebook.id)