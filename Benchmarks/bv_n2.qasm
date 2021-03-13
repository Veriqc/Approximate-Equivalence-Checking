//Bernstein-Vazirani with 2 qubits.
//Hidden string is 1
OPENQASM 2.0;
include "qelib1.inc";
qreg qr[2];
creg cr[1];
h qr[0];
x qr[1];
h qr[1];
barrier qr[0],qr[1];
cx qr[0],qr[1];
barrier qr[0],qr[1];
h qr[0];
measure qr[0] -> cr[0];
