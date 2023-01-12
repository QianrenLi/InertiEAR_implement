import numpy as np
ref_points_slow = [1300, 4100, 7200, 9900,12600,15600,18600,21600,24600,27600,30600,34500,37500,40500,44000,47000,50500,53500,56500,59500,62750,66000,69000,72500,75500,79000,81750,84500,87750,90750,93800,96750,99750,102800,105750,109000,112000,115000,118000,121000,123750,127000,130000,133000,136000,139000,142600,145500,148500,151500,154000,157400,160500,163500,167000,170000,173000,176500,179500,182500,185500,188500,191500,195000,198000,201000,204250,207500,211000,214000,217500,220500,224000,227000,230000,233500,236500,239750,242750,246000,249500,252500,255500,259000,261750,264750,268000,271000,274500,277750,281000,284200,287250,290250,293500,296500,299750,303000,306000,309000,312000,315000,317750,321000,327000,329800,332700,335500,338550,341500,344250,347500,350120,353150,356150,359100,362000,364750,367750,371000,374000,376760,379600,382500,385400,388380,391250,394240,397250,400000,403050,405870,408820,412000,414800,417600,420600,423600,426400,429500,432400,435400,438000,441100,443900,447000,450000,453000,456000,459200,462000,465000,467750,470750,473600,476800,479900,482780,486000,488750,492000,495000,497800,500800,503800,506800,509900,512780,515600,518600,521500,524500,527700,530500,533500,536500,539600,542500,545500,548500,551500,554500,557500,560500,563500,566500,569300,572400,575500,578500,581250,584500,587250,590600,593600,596500,599500,602800]
ref_points_np = np.array(ref_points_slow)

initial_medium = 2800
width_reduction_medium = 900

ref_medium_np = np.cumsum(np.diff(ref_points_np) - width_reduction_medium) + initial_medium

print(ref_medium_np)
# print(np.argmin(np.diff(ref_points_np)))
# print(ref_points_np[46])
# print(np.diff(ref_points_np))
# print(len(ref_points_slow))