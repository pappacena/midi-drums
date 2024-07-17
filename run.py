import curses
import mido

outport = mido.open_output()

stdscr = curses.initscr()
curses.cbreak()
stdscr.keypad(1)

stdscr.addstr(0,10,"Hit 'q' to quit")
stdscr.refresh()

key = ''
while key != ord('q'):
    key = stdscr.getch()
    notes = {
        ord('a'): 57,
        ord('s'): 40,
        ord('d'): 50,
        ord('f'): 60,
        ord('g'): 70,
        ord('h'): 80,
    }
    outport.send(mido.Message('note_on', note=notes[key], velocity=100))
    stdscr.addch(20, 25, key)
    stdscr.refresh()
    #if key == curses.KEY_UP:
    #    stdscr.addstr(2, 20, "Up")
    #elif key == curses.KEY_DOWN:
    #    stdscr.addstr(3, 20, "Down")

curses.endwin()
