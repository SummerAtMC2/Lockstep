import networkx as nx
import sys

def build_stars(inputfn,big_dt,small_dt):

	STARSIZELIMIT = 1

	edgefile = open(inputfn,'r')
	outfilefn = inputfn.replace(inputfn.split('/')[-1],'stars.csv')

	print outfilefn

	outfile = open(outfilefn,"w")

	outfile.write("sid,rhn,lhns\n")

	G = nx.Graph()

	rhns = []

	print "Load edge file\n"
	edgedata = [line.rstrip() for line in edgefile]
	print "End loading\n"
	edgefile.close()


	flhn, frhn, fts = edgedata[0].split(',') #First row

	lb = int(fts) # Start of time window
	ub = lb + big_dt # End of time window
	count = 0

	for i in range(len(edgedata)):

		lhn, rhn, ts = edgedata[i].split(',')

		if int(ts) < lb:
			continue

		for j in range(i, len(edgedata)):

			lhn, rhn, ts = edgedata[j].split(',')

			if int(ts) > ub:
				break

			rhns.append(rhn)

			G.add_edge(lhn, rhn)
		print rhns, lb, '-', ub

		for y in set(rhns):

			lhns = G.neighbors(y)

			print rhns, y, lhns

			lhns.sort()

			if len(lhns) <= STARSIZELIMIT: # Get rid of small stars
				continue       
			outfile.write('%d, %s, "%s"\n'%(count,y,str(lhns)))

		count += 1
		rhns = []
		G.clear()
		# Swift the timewindow
		lb += small_dt
		ub += small_dt

	outfile.close()

	return outfilefn


if __name__=='__main__':
	inputfn = sys.argv[1]
	big_dt_day = int(sys.argv[2])
	small_dt_day = int(sys.argv[3])

	star_fn = build_stars(inputfn,big_dt_day,small_dt_day)