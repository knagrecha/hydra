import curses
import time


class Logger():
    def __init__(self, tasks):
        self.tasks = tasks
        self.stdscr = curses.initscr()
        curses.noecho()
        curses.cbreak()
        
        
    def cleanup(self):
        curses.nocbreak()
        self.stdscr.keypad(0)
        curses.echo()
        curses.endwin()
        
    def task_progress(self, name, epoch_count, minibatch, minibatch_count, m_time, t_time, loss, y, x):
        progress = int(minibatch / minibatch_count)
        self.stdscr.addstr(y, x, "{} | Epochs Remaining {}".format(name, epoch_count))
        self.stdscr.addstr(y+1, x, "Epoch Progress: [{1:10}] {0}%".format(progress * 10, "#" * progress))
        self.stdscr.addstr(y+2, x, "Minibatch Time: {:.2f}s | Remaining Time in Epoch: {:.2f}s".format(m_time, t_time))
        self.stdscr.addstr(y+3, x, "Last Loss: {}".format(loss))
        self.stdscr.refresh()
    


    def report_progress(self, task_names, epoch_counts, minibatches, minibatch_counts, m_times, t_times, losses):
        for task_idx, task in enumerate(task_names):
            self.task_progress(task, epoch_counts[task_idx], minibatches[task_idx], minibatch_counts[task_idx], 
                                                          m_times[task_idx], t_times[task_idx], losses[task_idx], 0 + 5 * task_idx, 0)
  
    def refresh(self):
        names, epoch_counts, minibatches, minibatch_counts, m_times, t_times, losses = [], [], [], [], [], [], []
        for t in self.tasks:
            names.append(t.name)
            epoch_counts.append(t.epochs)
            minibatches.append(t.total_length - t.batches_remaining)
            minibatch_counts.append(t.total_length)
            m_times.append(t.total_time)
            t_times.append(t.total_time * t.batches_remaining)
            losses.append(t.last_loss)
            

        self.report_progress(names, epoch_counts, minibatches, minibatch_counts, m_times, t_times, losses)
        
    
