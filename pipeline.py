import build_stars_new
import time
from subprocess import Popen

if __name__ == '__main__':
	inputfn = raw_input('Input filename: ')
	big_dt = int(raw_input('Time window size: '))
	small_dt = int(raw_input('Time window slide: '))

	print ("Start building stars\n")
	start = time.time()
	star_fn = build_stars_new.build_stars(inputfn,big_dt,small_dt)
	print ("Star construction time: "+ str(time.time()-start))

	print ("Start fptree construction and lockstep extraction\n")
	start = time.time()
	nbrate = raw_input('Near_biclique rate (100 complete biclique): ')
	supplement = raw_input('Do suplement phase: ')
	Popen('python fptree_const.py %s %s %s'%(star_fn,nbrate,supplement),shell=True).wait()
	print ("Lockstep extraction time: "+ str(time.time()-start))












