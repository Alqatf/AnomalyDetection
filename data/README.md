NSL-KDD99
=========
[NSL-KDD dataset](http://nsl.cs.unb.ca/NSL-KDD/)

Files
------
KDDTrain+.ARFF  The full NSL-KDD train set with binary labels in ARFF format
KDDTrain+.TXT   The full NSL-KDD train set including attack-type labels and difficulty level in CSV format
KDDTrain+20Percent.ARFF    A 20% subset of the KDDTrain+.arff file
KDDTrain+20Percent.TXT A 20% subset of the KDDTrain+.txt file
KDDTest+.ARFF   The full NSL-KDD test set with binary labels in ARFF format
KDDTest+.TXT    The full NSL-KDD test set including attack-type labels and difficulty level in CSV format
KDDTest-21.ARFF A subset of the KDDTest+.arff file which does not include records with difficulty level of 21 out of 21
KDDTest-21.TXT  A subset of the KDDTest+.txt file which does not include records with difficulty level of 21 out of 21

Attributes
----------
duration,protocol-type,service,flag,src-bytes,dst-bytes,land,wrong-fragment,urgent,hot,num-failed-logins,logged-in,num-compromised,root-shell,su-attempted,num-root,num-file-creations,num-shells,num-access-files,num-outbound-cmds,is-host-login,is-guest-login,count,srv-count,serror-rate,srv-serror-rate,rerror-rate,srv-rerror-rate,same-srv-rate,diff-srv-rate,srv-diff-host-rate,dst-host-count,dst-host-srv-count,dst-host-same-srv-rate,dst-host-diff-srv-rate,dst-host-same-src-port-rate,dst-host-srv-diff-host-rate,dst-host-serror-rate,dst-host-srv-serror-rate,dst-host-rerror-rate,dst-host-srv-rerror-rate,class

attribute 'duration' real
attribute 'protocol-type' {'tcp','udp', 'icmp'} 
attribute 'service' {'aol', 'auth', 'bgp', 'courier', 'csnet-ns', 'ctf', 'daytime', 'discard', 'domain', 'domain-u', 'echo', 'eco-i', 'ecr-i', 'efs', 'exec', 'finger', 'ftp', 'ftp-data', 'gopher', 'harvest', 'hostnames', 'http', 'http-2784', 'http-443', 'http-8001', 'imap4', 'IRC', 'iso-tsap', 'klogin', 'kshell', 'ldap', 'link', 'login', 'mtp', 'name', 'netbios-dgm', 'netbios-ns', 'netbios-ssn', 'netstat', 'nnsp', 'nntp', 'ntp-u', 'other', 'pm-dump', 'pop-2', 'pop-3', 'printer', 'private', 'red-i', 'remote-job', 'rje', 'shell', 'smtp', 'sql-net', 'ssh', 'sunrpc', 'supdup', 'systat', 'telnet', 'tftp-u', 'tim-i', 'time', 'urh-i', 'urp-i', 'uucp', 'uucp-path', 'vmnet', 'whois', 'X11', 'Z39-50'} 
attribute 'flag' { 'OTH', 'REJ', 'RSTO', 'RSTOS0', 'RSTR', 'S0', 'S1', 'S2', 'S3', 'SF', 'SH' }

OTH: Other, a state not contemplated here. 
REJ: Connection rejected, Initial SYN elicited a RST in reply 
RSTO: Connection reset by the originator 
RSTOS0: Originator sent a SYN followed by a RST, we never saw a SYN
ACK from the responder.
RSTR: Connection reset by the responder 
S0 State 0: initial SYN seen but no reply 
S1 State 1: connection established (SYN's exchanged), nothing 
further seen 
S2 State 2: connection established, initiator has closed their side 
S3 State 3: connection established, responder has closed their side 
SF: Normal SYN/FIN completion 
SH: Originator sent a SYN followed by a FIN, we never saw a SYN ACK
from the responder (hence the connection was “half” open).

[source 1](http://www.iaeng.org/publication/WCECS2012/WCECS2012_pp30-35.pdf)
[source 2](http://www.takakura.com/Kyoto_data/BenchmarkData-Description-v3.pdf)

attribute 'src-bytes' real
attribute 'dst-bytes' real
attribute 'land' {'0', '1'}
attribute 'wrong-fragment' real
attribute 'urgent' real
attribute 'hot' real
attribute 'num-failed-logins' real
attribute 'logged-in' {'0', '1'}
attribute 'num-compromised' real
attribute 'root-shell' real
attribute 'su-attempted' real
attribute 'num-root' real
attribute 'num-file-creations' real
attribute 'num-shells' real
attribute 'num-access-files' real
attribute 'num-outbound-cmds' real
attribute 'is-host-login' {'0', '1'}
attribute 'is-guest-login' {'0', '1'}
attribute 'count' real
attribute 'srv-count' real
attribute 'serror-rate' real
attribute 'srv-serror-rate' real
attribute 'rerror-rate' real
attribute 'srv-rerror-rate' real
attribute 'same-srv-rate' real
attribute 'diff-srv-rate' real
attribute 'srv-diff-host-rate' real
attribute 'dst-host-count' real
attribute 'dst-host-srv-count' real
attribute 'dst-host-same-srv-rate' real
attribute 'dst-host-diff-srv-rate' real
attribute 'dst-host-same-src-port-rate' real
attribute 'dst-host-srv-diff-host-rate' real
attribute 'dst-host-serror-rate' real
attribute 'dst-host-srv-serror-rate' real
attribute 'dst-host-rerror-rate' real
attribute 'dst-host-srv-rerror-rate' real
attribute 'class' {'normal', 'anomaly'}

