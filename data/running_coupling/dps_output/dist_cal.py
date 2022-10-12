#!/usr/bin/env python

import os

for dp in range(22, 23, 1): 
    os.system('cp /global/cscratch1/sd/td115/output/Tequila/running_coupling/AuAu200/centrality0-10/dp0/hadrons/cross_section_avepT.py /global/cscratch1/sd/td115/output/Tequila/running_coupling/AuAu200/centrality0-10/dp%d/hadrons' %dp)
    os.system('cp /global/cscratch1/sd/td115/output/Tequila/running_coupling/AuAu200/centrality0-10/dp0/hadrons/sigmaGenErr.txt /global/cscratch1/sd/td115/output/Tequila/running_coupling/AuAu200/centrality0-10/dp%d/hadrons' %dp)

    # os.system('cd /global/cscratch1/sd/td115/output/Tequila/running_coupling/AuAu200/centrality0-10/dp%d/hadrons' %dp)
    os.system('cd /global/cscratch1/sd/td115/output/Tequila/running_coupling/AuAu200/centrality0-10/dp%d/hadrons && python /global/cscratch1/sd/td115/output/Tequila/running_coupling/AuAu200/centrality0-10/dp%d/hadrons/cross_section_avepT.py --dp %d' %(dp, dp, dp))

