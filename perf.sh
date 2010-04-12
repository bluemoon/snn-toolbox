#!/bin/sh
CPUPROFILE=profile python test.py
#pprof --disasm=link_out /bin/ls ls.prof
pprof --gif /usr/bin/python profile > p.gif
#pprof --callgrind /usr/bin/python profile > python.callgrind
#kcachegrind python.callgrind
