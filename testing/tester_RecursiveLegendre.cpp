#include "NjetModel.hpp"

int main(int argCount, char** argVec)
{
	RecursiveLegendre<double, 1024> Pl_computer;
	
	for(size_t i = 0; i < Pl_computer.z.size(); ++i)
		Pl_computer.z[i] = -1. + 1./double(Pl_computer.z.size()) + 
			2.*double(i)/double(Pl_computer.z.size());
		
	Pl_computer.Reset();	
		
	auto P1 = Pl_computer.Next();
	auto P2 = Pl_computer.Next();
	auto P3 = Pl_computer.Next();
	auto P4 = Pl_computer.Next();
	auto P5 = Pl_computer.Next();
	
	for(size_t i = 0; i < Pl_computer.z.size(); ++i)
		printf("%.3e  %.3e  %.3e  %.3e  %.3e  %.3e\n", 
			Pl_computer.z[i], P1[i], P2[i], P3[i], P4[i], P5[i]);
		
	return 0;
}
