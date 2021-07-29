OPENQASM 2.0;
include "qelib1.inc";
qreg q5[5];
creg c5[5];
u3(0.858713078139674,-0.902081617682068,0.944507764103263) q5[0];
u3(0.494212588719947,2.15997844132294,-2.77148022334507) q5[3];
cx q5[3],q5[0];
u1(1.09716802430385) q5[0];
u3(-0.406202803565922,0.0,0.0) q5[3];
cx q5[0],q5[3];
u3(1.84909992939079,0.0,0.0) q5[3];
cx q5[3],q5[0];
u3(1.72758736162678,1.66373246385959,0.244639904916478) q5[0];
u3(1.85794236094111,-4.87836586570177,-0.760242896754066) q5[3];
u3(0.628756831938898,-0.259259071968556,1.23260460407296) q5[2];
u3(0.577807577729503,-1.47756603476898,-0.534427538678117) q5[4];
cx q5[4],q5[2];
u1(0.840471499407081) q5[2];
u3(0.0768901668672914,0.0,0.0) q5[4];
cx q5[2],q5[4];
u3(2.01467228245153,0.0,0.0) q5[4];
cx q5[4],q5[2];
u3(0.732900037226540,1.20529163021660,0.393625443725726) q5[2];
u3(0.768815705796674,-1.76406073912995,-4.24886329350704) q5[4];
u3(2.32326600288941,0.407892810796966,0.119048891888269) q5[3];
u3(0.983026808125432,-3.80526265267414,-1.12318934127172) q5[4];
cx q5[4],q5[3];
u1(1.78251747178956) q5[3];
u3(-3.09106772079883,0.0,0.0) q5[4];
cx q5[3],q5[4];
u3(0.761280043368228,0.0,0.0) q5[4];
cx q5[4],q5[3];
u3(1.02905662321870,2.70849613467213,-2.35345392876964) q5[3];
u3(1.87717934222728,-0.264574906140136,-2.31456418391774) q5[4];
u3(1.24088646413842,0.235531202591684,1.60967192765585) q5[0];
u3(2.34487380278260,-2.42624033561739,-0.705492832628579) q5[2];
cx q5[2],q5[0];
u1(1.73517000199425) q5[0];
u3(-2.83177598906838,0.0,0.0) q5[2];
cx q5[0],q5[2];
u3(0.791871395399319,0.0,0.0) q5[2];
cx q5[2],q5[0];
u3(0.956569876403668,1.88806285638272,-2.64777144958244) q5[0];
u3(1.15397638003235,3.62122578179927,2.55078828995023) q5[2];
u3(0.587878954919190,-2.61677203216931,2.86202893280719) q5[3];
u3(1.26383000985472,0.0973041061598896,-2.12649123243527) q5[0];
cx q5[0],q5[3];
u1(3.03687969475460) q5[3];
u3(-2.64395159960897,0.0,0.0) q5[0];
cx q5[3],q5[0];
u3(1.21990780807383,0.0,0.0) q5[0];
cx q5[0],q5[3];
u3(1.02594012421873,-2.11409652483203,3.03355874479514) q5[3];
u3(2.61393243324750,2.99559185937631,0.636403621854072) q5[0];
u3(1.57727278133247,-2.28487239533224,3.56490476821174) q5[1];
u3(0.354724566391358,3.40919991445182,-1.67461138649603) q5[4];
cx q5[4],q5[1];
u1(-0.464165887136636) q5[1];
u3(0.137735849346107,0.0,0.0) q5[4];
cx q5[1],q5[4];
u3(4.28843966230764,0.0,0.0) q5[4];
cx q5[4],q5[1];
u3(2.47055184720376,-2.57977066830705,2.14911284467680) q5[1];
u3(2.91919470007013,0.927825651973451,2.48519431611446) q5[4];
u3(0.794468015850656,-1.46831813434691,0.828469640075511) q5[0];
u3(0.826631790593711,-2.37193591743099,-0.341132631359333) q5[3];
cx q5[3],q5[0];
u1(0.913878928616460) q5[0];
u3(0.0799935080626575,0.0,0.0) q5[3];
cx q5[0],q5[3];
u3(2.86710007783752,0.0,0.0) q5[3];
cx q5[3],q5[0];
u3(1.97806746381232,3.53729738981042,-0.989044910426094) q5[0];
u3(1.46234490348025,-3.40196297726301,-0.700135369194713) q5[3];
u3(0.687366238597798,1.10331212112405,-1.43025990978376) q5[1];
u3(0.0141269058989462,-0.593266817147686,-2.17669889696521) q5[2];
cx q5[2],q5[1];
u1(1.64700616521060) q5[1];
u3(-2.45276953662416,0.0,0.0) q5[2];
cx q5[1],q5[2];
u3(0.373328826654425,0.0,0.0) q5[2];
cx q5[2],q5[1];
u3(0.430558941455295,-0.905374086008988,1.87839353681852) q5[1];
u3(2.21003837612286,4.52364017287998,0.215143922085437) q5[2];
barrier q5[0],q5[1],q5[2],q5[3],q5[4];
measure q5[0] -> c5[0];
measure q5[1] -> c5[1];
measure q5[2] -> c5[2];
measure q5[3] -> c5[3];
measure q5[4] -> c5[4];