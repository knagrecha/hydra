import curses
import time


def task_progress(name, epoch_count, minibatch, minibatch_count, m_time, t_time, y, x):
    progress = int(minibatch / minibatch_count)
    stdscr.addstr(y, x, "{} | Epochs Remaining {}".format(name, epoch_count))
    stdscr.addstr(y+1, x, "Epoch Progress: [{1:10}] {0}%".format(progress * 10, "#" * progress))
    stdscr.addstr(y+2, x, "Minibatch Time: {:.2f}s | Remaining Time in Epoch: {:.2f}s".format(m_time, t_time))
    stdscr.refresh()

def report_progress(task_names, epoch_counts, minibatches, minibatch_counts, m_times, t_times):

    for task_idx, task in enumerate(task_names):
        task_progress(task, epoch_counts[task_idx], minibatches[task_idx], minibatch_counts[task_idx], 
                                                      m_times[task_idx], t_times[task_idx], 0 + 4 * task+task_idx, 0)

    
    

class Logger():
    def __init__(self, tasks):
        self.tasks = tasks
        stdscr = curses.initscr()
        curses.noecho()
        curses.cbreak()
  
    def refresh(self):
        names, epochs, epoch_counts, minibatches, minibatch_counts, m_times, t_times = [], [], [], [], [], [], []
        for t in tasks:
            names.append(t.name)
            epochs.append(t.epochs)
            minibatches.append(t.total_length - t.batches_remaining)
            minibatch_counts.append(t.total_length)
            m_times.append(t.total_time)
            t_times.append(t.total_time * t.batches_remaining)
            
