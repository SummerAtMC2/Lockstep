import networkx as nx
import time
import csv
import hashlib
import sys
import operator

# csv.field_size_limit(sys.maxsize)
csv.field_size_limit(sys.maxint)

downloader_dict = {}
url_dict = {}
g = nx.Graph()


fp_tree = {}
fp_tree['root'] = []

freq = {}
freq['root'] = 0

visited = {}
visited['root'] = []

prefix = {}
prefix['root'] = []


star_ids = {}
star_ids_flip = {}
node_version = {}
locksteps = {}

superset_relationship = {}
star_buckets = {}

outfile = open('data/dlr_dom_lcks.csv','w')
outfile.write("lid,rhns,lhns,near_rate\n")

def stats(g):

    print("nodes : "+str(len(g.nodes())))
    print("edges : "+str(len(g.edges())))


def near_biclique_rate(xlen,ylen,add_x,add_y,missing_x,missing_y):

    new_edges = missing_x * (add_y + add_x) + missing_y * add_x
    new_biclique = (xlen + add_x) * (ylen + add_y)

    rate = float(new_biclique - new_edges) / float(new_biclique)

    return rate

def get_near_bicliques(node, prefix_local, visited_local, fp_tree_local, nbrate):

    trld = nbrate
    xnodes = prefix_local[node]
    ynodes = visited_local[node]

    xlen = len(xnodes)
    ylen = len(ynodes)
    missing_x = 0
    missing_y = 0
    poss_y = []

    for xidx in range(xlen-1,0,-1):
        crrnode = xnodes[xidx-1]
        missing_x += 1
        if len(visited_local[crrnode]) > ylen:
            poss_y = list(set(visited_local[crrnode]) - set(ynodes))
            break
    if len(poss_y) == 0:
        missing_x = 0

    chren = [key for key in fp_tree_local[node]]
    visited_chren = {}
    for chd in chren:
        visited_chren.update({chd:visited_local[chd]})

    sorted_chren = sorted(visited_chren.items(), key=operator.itemgetter(1), reverse = True)

    add_x = 0
    add_y = 0

    y_added = 0

    new_xnodes = xnodes
    new_ynodes = ynodes

    near_rate = 1

    for new_x in sorted_chren:
        missing_y = len(ynodes) - len(new_x[1])
        if missing_x > missing_y:
            add_x += 1
            if near_biclique_rate(xlen,ylen,add_x,add_y,missing_x,missing_y) < trld:
                add_x -= 1
                break
            else:
                near_rate = near_biclique_rate(xlen,ylen,add_x,add_y,missing_x,missing_y)
                new_xnodes.append(new_x[0])
        else:
            if y_added == 0:
                for i in range(0, len(poss_y)):
                    if near_biclique_rate(xlen,ylen,add_x,i+1,missing_x,missing_y) < trld:
                        break
                    else:
                        add_y += 1
                        near_rate = near_biclique_rate(xlen,ylen,add_x,add_y,missing_x,missing_y)
                        new_ynodes.append(poss_y[i])
                y_added = 1
            add_x += 1
            if near_biclique_rate(xlen,ylen,add_x,add_y,missing_x,missing_y) < trld:
                add_x -= 1
                break
            else:
                near_rate = near_biclique_rate(xlen,ylen,add_x,add_y,missing_x,missing_y)
                new_xnodes.append(new_x[0])


    return new_xnodes, new_ynodes, near_rate


def extract_fptree_comp(fp_tree_local, prefix_local, visited_local, nbrate):
    visit_list = ['root']

    while len(visit_list) > 0:
        curr_node = visit_list[0]
        to_visit = [key for key in fp_tree_local[curr_node]]
        noneedto = 0
        for ele in to_visit:
            if len(visited_local[curr_node]) == len(visited_local[ele]) or curr_node == 'root':
                noneedto = 1
            prefix_local[ele] = []
            prefix_local[ele].extend(prefix_local[curr_node])
            prefix_local[ele].append(ele)

        visit_list.extend(to_visit)

        #li = visited[curr_node]
        if noneedto == 1:
            del visit_list[0]
            continue

        newx, newy, near_rate = get_near_bicliques(curr_node,prefix_local,visited_local,fp_tree_local, nbrate)

        ynodes = {}
        for ynode_id in newy:
            sid, yid = eval(ynode_id)
            try:
                ynodes[yid].append(sid)
            except:
                ynodes.update({yid:[sid]})
        all_sids = ynodes.values()
        yids = ynodes.keys()
        #xids = [name for (name,version) in prefix[curr_node]]
        xids = [name for (name,version) in newx]
        yids.sort()
        xids.sort()
        lck = [yids,xids]
        hash_obj = hashlib.sha256(str(lck))
        lckSHA = hash_obj.hexdigest()

        if len(xids) > 2 and len(yids) > 2: # minimum size of the lockstep
            outfile.write("%s,'%s','%s',%s\n"%(lckSHA,yids,xids,near_rate))
        del visit_list[0]

def construct_fptree_comp(s, dwn_dict_local, url_dict_local, fp_tree_local, visited_local, freq_local, node_version_local):

    urlkeys = sorted(url_dict_local.keys(), key = s.degree, reverse = True )
    for dwn in dwn_dict_local:
        node_version_local.update({dwn:0})

    for key in urlkeys:

        curr_node = 'root'

        i=0

        ####matching common prefix

        while i < len(url_dict_local[key]):

            children = [childname for (childname,version) in fp_tree_local[curr_node]]

            if url_dict_local[key][i] not in children:

                break

            for (childname,version) in fp_tree_local[curr_node]:
                if childname == url_dict_local[key][i]:
                    curr_node = (childname,version)

            freq_local[curr_node] += 1

            visited_local[curr_node].append(key)

            i+=1

        ##insert new nodes into fp tree

        while i < len(url_dict_local[key]):
            node_name, version = url_dict_local[key][i], node_version_local[url_dict_local[key][i]]
            fp_tree_local[curr_node].append((node_name,version))
            node_version_local[node_name] += 1

            curr_node = (node_name,version)

            fp_tree_local[curr_node] = []

            freq_local[curr_node] = 1

            visited_local[curr_node] = [key]

            i+=1

    return


def sort_dicts_comp(s, dwn_dict_local, url_dict_local):

    for k in url_dict_local.keys():
        url_dict_local[k] = sorted(url_dict_local[k], key = s.degree, reverse = True )


    for k in dwn_dict_local.keys():
        dwn_dict_local[k] = sorted(dwn_dict_local[k], key = s.degree, reverse = True )


    return

def create_bipartite_graph_from_star(star_list, dwn_dict_local, url_dict_local):
    s = nx.Graph()

    for star in star_list:
        yid, xids, sid = star[0], star[1], star[2]

        ynode_id = str([sid,yid])
        s.add_node(ynode_id,bipartite=1)
        for xid in xids:
            s.add_node(xid,bipartite=0)
            s.add_edge(xid,ynode_id)


    for node in s.nodes():
        if s.node[node]['bipartite'] == 1:
            url_dict_local.update({node:s.edge[node].keys()})
        else:
            dwn_dict_local.update({node:s.edge[node].keys()})

    return s

def create_bipartite_graph(filename):

    star_id_count = 0

    edgelist = []

    star_buckets = {}

    i=0
    print filename
    with open(filename,'r') as f:

        for line in csv.DictReader(f,delimiter=',',quotechar='"',skipinitialspace=True):
            pid = line['sid']
            yid = line['rhn']
            
            xids = eval(line['lhns'])
            xids.sort()
            star = [yid,xids]
            hash_obj = hashlib.sha256(str(star))
            sha2_id = hash_obj.hexdigest()

            if sha2_id not in star_ids:
                star_ids.update({sha2_id:{'sid':star_id_count,'yid':yid,'xids':xids}})
                star_ids_flip.update({star_id_count: sha2_id})
                sid = star_id_count
                superset_relationship[sid] = []
                update_stars(star,sid)
                star_id_count += 1
            i+=1


    for sid in superset_relationship:
        yid = star_ids[star_ids_flip[sid]]['yid']
        xids = star_ids[star_ids_flip[sid]]['xids']
        ynode_id = str([sid,yid])
        g.add_node(ynode_id,bipartite=1)
        for xid in xids:
            g.add_node(xid,bipartite=0)
            g.add_edge(xid,ynode_id)

    for node in g.nodes():
        if g.node[node]['bipartite'] == 1:
            url_dict.update({node:g.edge[node].keys()})
        else:
            downloader_dict.update({node:g.edge[node].keys()})

    return g


def update_stars(star,sid):
    yid, xids = star

    if yid not in star_buckets:
        star_buckets.update({yid:{str(xids):sid}})
    else:
        all_xids = [eval(xsets) for xsets in star_buckets[yid].keys()]
        is_subset = 0
        is_superset = 0
        for comp_xids in all_xids:
            if set(xids).issubset(set(comp_xids)):
                is_subset = 1
                sid_comp_xids = star_buckets[yid][str(comp_xids)]
                superset_relationship[sid_comp_xids].append(sid)

            if set(comp_xids).issubset(set(xids)):
                is_superset = 1
                sid_comp_xids = star_buckets[yid][str(comp_xids)]
                superset_relationship[sid] = list(set(superset_relationship[sid] + superset_relationship[sid_comp_xids]))

                superset_relationship[sid].append(sid_comp_xids)

                star_buckets[yid].pop(str(comp_xids))
                superset_relationship.pop(sid_comp_xids)

        if is_subset == 0 and is_superset == 0:
            star_buckets[yid].update({str(xids):sid})
        elif is_superset == 1:
            star_buckets[yid].update({str(xids):sid})
        else:
            superset_relationship.pop(sid)

    return

def sort_dicts(g):

    for k in url_dict.keys():
        url_dict[k] = sorted(url_dict[k], key = g.degree, reverse = True )


    for k in downloader_dict.keys():
        downloader_dict[k] = sorted(downloader_dict[k], key = g.degree, reverse = True )


    return


def construct_fptree():

    urlkeys = sorted(url_dict.keys(), key = g.degree, reverse = True )
    for dwn in downloader_dict:
        node_version.update({dwn:0})

    for key in urlkeys:


        curr_node = 'root'

        i=0

        ####matching common prefix

        while i < len(url_dict[key]):

            children = [childname for (childname,version) in fp_tree[curr_node]]

            if url_dict[key][i] not in children:

                break

            for (childname,version) in fp_tree[curr_node]:
                if childname == url_dict[key][i]:
                    curr_node = (childname,version)

            freq[curr_node] += 1

            visited[curr_node].append(key)

            i+=1

        ##insert new nodes into fp tree

        while i < len(url_dict[key]):
            node_name, version = url_dict[key][i], node_version[url_dict[key][i]]
            fp_tree[curr_node].append((node_name,version))
            node_version[node_name] += 1

            curr_node = (node_name,version)

            fp_tree[curr_node] = []

            freq[curr_node] = 1

            visited[curr_node] = [key]

            i+=1

    return


def prune_fptree(sup):


    for key in fp_tree.keys():

        for val in fp_tree[key]:

            if freq[val] < sup:

                fp_tree[key].remove(val)

    return


def extract_fptree(nbrate,supplement):
    visit_list = ['root']

    visitcounter = 1
    while len(visit_list) > 0:
        #print visitcounter
        visitcounter += 1
        curr_node = visit_list[0]
        to_visit = [key for key in fp_tree[curr_node]]
        noneedto = 0
        for ele in to_visit:
            if len(visited[curr_node]) == len(visited[ele]) or curr_node == 'root':
                noneedto = 1
            prefix[ele] = []
            prefix[ele].extend(prefix[curr_node])
            prefix[ele].append(ele)

        visit_list.extend(to_visit)

        #li = visited[curr_node]
        if noneedto == 1:
            del visit_list[0]
            continue

        newx, newy, near_rate = get_near_bicliques(curr_node,prefix,visited,fp_tree,nbrate)

        ynodes = {}
        for ynode_id in newy:
            sid, yid = eval(ynode_id)
            try:
                ynodes[yid].append(sid)
            except:
                ynodes.update({yid:[sid]})
        all_sids = ynodes.values()
        yids = ynodes.keys()
        #xids = [name for (name,version) in prefix[curr_node]]
        xids = [name for (name,version) in newx]
        yids.sort()
        xids.sort()
        lck = [yids,xids]
        hash_obj = hashlib.sha256(str(lck))
        lckSHA = hash_obj.hexdigest()
        all_sids_comb = []

        if len(xids) > 2 and len(yids) > 2:
            outfile.write('%s,"%s","%s",%s\n'%(lckSHA,yids,xids,near_rate))
        del visit_list[0]

    if supplement != 1:
        print "pass supplementation step"
        return

    nodes_to_supplement = []
    for nd in node_version:
        if node_version[nd] > 0:
            nodes_to_supplement.append(nd)
    count_supplement = 0
    for ntc in nodes_to_supplement:
        print "supplement " + str(ntc) + " (" + str(count_supplement) + "/" +str(len(nodes_to_supplement)) + ")"
        supplement_fptree(ntc)
        count_supplement += 1



def supplement_fptree(nd):
    dwn_dict_lc = {}
    url_dict_lc = {}
    fp_tree_lc = {}
    visited_lc = {}
    prefix_lc = {}
    freq_lc = {}
    node_version_lc = {}
    fp_tree_lc['root'] = []
    prefix_lc['root'] = []
    visited_lc['root'] = []
    freq_lc['root'] = []

    num_ver = node_version[nd]
    all_stars = []
    for ver in range(0,num_ver):
        curr_node = (nd,ver)
        for vt in visited[curr_node]:
            sid, yid = eval(vt)
            smeta = star_ids[star_ids_flip[sid]]
            vtstar = [smeta['yid'],smeta['xids'],sid]
            all_stars.append(vtstar)
    s  = create_bipartite_graph_from_star(all_stars, dwn_dict_lc, url_dict_lc)
    sort_dicts_comp(s, dwn_dict_lc, url_dict_lc)
    start = time.time()
    construct_fptree_comp(s, dwn_dict_lc, url_dict_lc, fp_tree_lc, visited_lc, freq_lc, node_version_lc)
    print("Constuct fp-tree"+ str(time.time()-start))
    print(str(len(fp_tree_lc.keys()))+" nodes in the fp tree")
    extract_fptree_comp(fp_tree_lc, prefix_lc, visited_lc, nbrate)


def main(starfn,nbrate,supplement):


    g = create_bipartite_graph(starfn)
    sort_dicts(g)

    start = time.time()

    construct_fptree()

    print("Constuct fp-tree"+ str(time.time()-start))

    print(str(len(fp_tree.keys()))+" nodes in the fp tree")

    g.clear()
    print "graph mem cleared"
    start = time.time()

    extract_fptree(nbrate,supplement)

    print("Constuct biclique"+ str(time.time()-start))


    outfile.close()


    return


if __name__ == '__main__':
    starfn = sys.argv[1]
    nbrate = float(sys.argv[2])
    supplement = int(sys.argv[3])
    main(starfn,nbrate,supplement)