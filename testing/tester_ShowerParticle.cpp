#include "NjetModel.hpp"
//~ #include "kdp/kdpVectors.hpp"

void PrintVec4(kdp::Vec4 const& p4)
{
	printf("[%.16e (%.16e, %.16e, %.16e)]\n", p4.x0, p4.x1, p4.x2, p4.x3);
}

int main()
{
	std::vector<double> params = {0.4, 0.9, 
			0.7, 0.8, 0.99, 
			0.1, 0.5, 0.99, 0.9, 
			0.1, 0.1, 0.99, -1.2};
	
	std::vector<std::vector<bool>> addresses = {{false}, {true}, {true, true}};
	
	ShowerParticle root(params, addresses);
	
	auto jets = root.GetJets();
	
	for(auto& jet : jets)
		PrintVec4(jet.p4);
	
	printf("\n");
	printf("inexact: %s\n", root.isShowerInexact() ? "yes" : "no");
	printf("absElost: %.16e\n", root.Total_absElost());
	PrintVec4(root.Total_p4());
	
	return 0;
}
