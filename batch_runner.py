import threading, queue
import time
q = queue.Queue()

def worker():
    while True:
        item = q.get()
        if item == 5:
            time.sleep(5)
        print(f'Working on {item}')
        print(f'Finished {item}')
        q.task_done()
        print(threading.activeCount())
# turn-on the worker thread

# send thirty task requests to the worker
for item in range(30):
    q.put(item)
print('All task requests sent\n', end='')

threading.Thread(target=worker).start()
q.join()
# block until all tasks are done
print('All work completed')