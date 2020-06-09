import nmap
import socket
#routerIP='192.168.0.1'
"""
	This code assumes that the aggregator know's it's own ip and scans for other devices on the router. 
	it returns its own ip followed by all other ips connected to the router.
"""

def run(routerIP):

	#latency speed recording is possible with nmap.
	nMappy =nmap.PortScanner()

	#scanning the router's ip address, on specified ports.
	nMappy.scan(hosts='192.168.0.0/24', arguments= '-n -sn ') #-sP -PE -PA21,23,24,80,3389
	#scanner results are put into a matrix. Status is ignored, only the ip.
	hosts_list = [(x) for x in nMappy.all_hosts()]
	#print(hosts_list)
	#removing the router's IP address.
	hosts_list.remove(routerIP)

	#identify aggregator IP address
	aggIPList = socket.gethostbyname_ex(socket.gethostname())[2]
	#aggIP = aggIPList[-1]
	#removing aggregator IP address
	aggIP = '192.168.0.105'

	hosts_list.remove(aggIP)
	return aggIP, hosts_list

print(run('192.168.0.1'))
print(' ')