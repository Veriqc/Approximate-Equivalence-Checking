OPENQASM 2.0;
include "qelib1.inc";
qreg q4[5];
creg c4[5];
u3(1.46591949407576,2.45594958402062,-1.74539633597193) q4[1];
u3(0.689540190320858,1.20968676361446,-2.40352679695779) q4[2];
cx q4[2],q4[1];
u1(1.77165686723653) q4[1];
u3(-2.10601722588071,0.0,0.0) q4[2];
cx q4[1],q4[2];
u3(0.143036176496447,0.0,0.0) q4[2];
cx q4[2],q4[1];
u3(2.86275042002748,-0.0532609396704569,-2.48764527254775) q4[1];
u3(1.24375991501351,3.83943966009430,1.47589466754961) q4[2];
u3(1.89864677896345,1.60591079702547,-0.0505837999497549) q4[4];
u3(2.21167469165439,0.627461394008116,-1.76567487691976) q4[3];
cx q4[3],q4[4];
u1(1.91414836502858) q4[4];
u3(-2.49714136957900,0.0,0.0) q4[3];
cx q4[4],q4[3];
u3(0.0499365022142744,0.0,0.0) q4[3];
cx q4[3],q4[4];
u3(0.663501741358494,-1.98597840518492,0.780365803772669) q4[4];
u3(2.27353582734376,-1.44926183666680,-4.31072115898583) q4[3];
u3(1.17265764773261,-1.45283400073720,-0.161348564097211) q4[1];
u3(1.12554374679816,-2.20326949584533,0.528256866947237) q4[4];
cx q4[4],q4[1];
u1(0.861921834300037) q4[1];
u3(-1.61282048459303,0.0,0.0) q4[4];
cx q4[1],q4[4];
u3(2.62334288888810,0.0,0.0) q4[4];
cx q4[4],q4[1];
u3(0.434292069327810,1.86245398767102,-4.02797055198996) q4[1];
u3(2.45880830535544,1.44634966781390,4.30187070497382) q4[4];
u3(0.890192255745184,-0.815171896454099,1.55252503304690) q4[0];
u3(1.71989403896402,-1.94880564118704,-2.24383252356406) q4[3];
cx q4[3],q4[0];
u1(2.35747954944101) q4[0];
u3(-3.02754696090328,0.0,0.0) q4[3];
cx q4[0],q4[3];
u3(1.22667006546371,0.0,0.0) q4[3];
cx q4[3],q4[0];
u3(0.594720954814969,-0.600472752942080,0.498336958511212) q4[0];
u3(2.72510077233322,0.496529596563687,-1.68925994528993) q4[3];
u3(2.38166685149797,0.456223413868142,0.168197051216620) q4[3];
u3(1.06713312350035,-0.547316002458407,-3.83737410442567) q4[1];
cx q4[1],q4[3];
u1(1.51992809638127) q4[3];
u3(-2.68199213205223,0.0,0.0) q4[1];
cx q4[3],q4[1];
u3(0.327030532670185,0.0,0.0) q4[1];
cx q4[1],q4[3];
u3(2.36558323208728,-1.79595752573243,-0.159598129542778) q4[3];
u3(1.08821919777050,-0.859783438199879,-0.905017625141656) q4[1];
u3(1.87451006179773,4.01351165895380,-1.39929515084246) q4[4];
u3(1.31719320176742,2.09883804144215,-0.326211090654265) q4[2];
cx q4[2],q4[4];
u1(1.61285481018370) q4[4];
u3(-0.352404406102992,0.0,0.0) q4[2];
cx q4[4],q4[2];
u3(2.13066330073936,0.0,0.0) q4[2];
cx q4[2],q4[4];
u3(1.42985434033523,-1.87284344484461,1.69672234038195) q4[4];
u3(1.21030814049415,1.49787608426554,2.82500416520181) q4[2];
barrier q4[0],q4[1],q4[2],q4[3],q4[4];
measure q4[0] -> c4[0];
measure q4[1] -> c4[1];
measure q4[2] -> c4[2];
measure q4[3] -> c4[3];
measure q4[4] -> c4[4];